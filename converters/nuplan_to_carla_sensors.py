#!/usr/bin/env python3
"""
nuPlan / NAVSIM rig  ->  CARLA 0.9.16 sensor-config YAML.

OUTPUT = 8 RGB cameras + EXACTLY ONE merged LiDAR (LIDAR_MERGED).  The 5 physical LiDARs are never emitted
(one ray-caster instead of five = ~5x cheaper).

Pipeline
--------
    python extract_rig.py   --logs L --blobs B --lidar --vehicle veh-35 --out rig.yaml
    python fit_lidar_rig.py --logs L --blobs B --vehicle veh-35 --out lidar_fit.yaml
    python nuplan_to_carla_sensors.py rig.yaml --vehicle veh-35 --lidar-fit lidar_fit.yaml -o carla_sensors.yaml

How the single merged LiDAR is built
------------------------------------
* position : fitted mounting point of the TOP LiDAR (fit_lidar_rig.py), or --lidar-mount X Y Z (nuPlan ego frame)
* channels / vertical FOV / horizontal FOV : paper Table VII top Pandar40P: 40 ch, [-25, +15] deg, 360 deg
* points_per_second : summed measured points/frame of all 5 LiDARs x rotation_frequency, capped (--lidar-pps-cap)
* range : 100 m by default (paper: 200 m; range is the main ray-cast cost)
CARLA spaces channels uniformly; the real Pandar40P does not. Bumper / A-pillar near-field coverage is lost.

Coordinate conventions
----------------------
nuPlan ego/lidar : right-handed, x forward, y LEFT, z up.  p_ego = R @ p_sensor + t   (origin = REAR AXLE)
nuPlan camera    : OpenCV, x right, y down, z forward.
CARLA (Unreal)   : left-handed, x forward, y RIGHT, z up. Angles in degrees, yaw > 0 = turn right, pitch > 0 = up.
CARLA camera     : looks along local +x, local y = right, local z = up.
cameras: R_carla = S @ R_ego @ M ;  lidar: R_carla = S @ R @ S ;  t_carla = S @ t + offset ;  S = diag(1,-1,1)

Loading the result in CARLA (keyword arguments, the YAML order is roll/pitch/yaw, carla.Rotation wants pitch/yaw/roll):
    tf = carla.Transform(carla.Location(x=t['x'], y=t['y'], z=t['z']),
                         carla.Rotation(pitch=t['pitch'], yaw=t['yaw'], roll=t['roll']))
Run the world synchronous with fixed_delta_seconds = simulation.fixed_delta_seconds from the output file
(LiDAR rotation_frequency must equal 1/fixed_delta_seconds, otherwise you get partial sweeps).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

# CARLA camera local axes (x fwd, y right, z up) expressed in OpenCV camera axes (x right, y down, z fwd)
M_CARLACAM_TO_CV = np.array([[0.0, 1.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [1.0, 0.0, 0.0]])
S_FLIP_Y = np.diag([1.0, -1.0, 1.0])

# nuPlan paper Table VII, top Pandar40P
TOP_CHANNELS, TOP_LOWER, TOP_UPPER = 40, -25.0, 15.0
PAPER_PPS_TOTAL = 3 * 720_000 + 2 * 360_000          # 3x Pandar40P + 2x Pandar20 (only used if nothing was measured)
PAPER_HZ = 20.0
# rear axle -> vehicle centre of the nuPlan Pacifica (nuplan-devkit get_pacifica_parameters; please verify)
REAR_AXLE_TO_CENTER = 1.461


# --------------------------------------------------------------------------- math helpers
def carla_matrix(roll, pitch, yaw):
    """Rotation matrix exactly as carla.Transform.get_matrix() builds it (angles in degrees)."""
    r, p, y = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp,      -cp * sr,               cp * cr],
    ])


def matrix_to_rpy(R):
    """Inverse of carla_matrix(); returns (roll, pitch, yaw) in degrees."""
    sp = float(np.clip(R[2, 0], -1.0, 1.0))
    pitch = math.asin(sp)
    if abs(sp) > 0.999999:  # gimbal lock
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(-R[2, 1], R[2, 2])
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def orthonormalize(R):
    """Snap to the nearest proper rotation (inputs are rounded to ~9 decimals)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def rnd(x, n=6):
    return round(float(x), n) + 0.0  # +0.0 removes "-0.0"


class Flow(dict):
    """dict that is dumped in YAML flow style ({a: 1, b: 2}) for compact transforms."""


yaml.SafeDumper.add_representer(
    Flow, lambda d, data: d.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True))


# --------------------------------------------------------------------------- shared context
class Ctx:
    """Conversion options shared by all sensors."""

    def __init__(self, args, ground_z):
        self.offset = np.array(args.origin_offset, float)  # CARLA frame (x fwd, y right, z up)
        if ground_z is not None:
            # nuPlan ego origin sits |ground_z| above the road; the CARLA vehicle origin is at ground level.
            self.offset[2] += -float(ground_z)
        self.notes = []
        self.pp_offsets = []


def ego_pose_to_carla(R_ego, t_ego, ctx, left_handed_local=False):
    """right-handed ego pose -> CARLA pose.
    left_handed_local=True: the sensor's own axes are already CARLA's left-handed (fwd, right, up) basis
    (cameras), so only the parent frame is flipped: Rc = S @ R. Otherwise Rc = S @ R @ S."""
    Rc = S_FLIP_Y @ R_ego if left_handed_local else S_FLIP_Y @ R_ego @ S_FLIP_Y
    tc = S_FLIP_Y @ np.asarray(t_ego, float) + ctx.offset
    return Rc, tc


def transform_dict(Rc, tc):
    roll, pitch, yaw = matrix_to_rpy(Rc)
    err = np.abs(carla_matrix(roll, pitch, yaw) - Rc).max()  # self-check
    if err > 1e-6:
        raise RuntimeError(f"rotation round-trip error {err:.2e}")
    return Flow(x=rnd(tc[0]), y=rnd(tc[1]), z=rnd(tc[2]),
                roll=rnd(wrap180(roll)), pitch=rnd(pitch), yaw=rnd(wrap180(yaw)))


def get_lidar2ego(calib, ctx):
    """-> (R, t). The extractor writes lidar2ego: None when the logs have none -> identity."""
    l2e = calib.get("lidar2ego")
    if not l2e:
        ctx.notes.append("lidar2ego missing in the YAML: assuming LiDAR frame == ego frame.")
        return np.eye(3), np.zeros(3)
    return orthonormalize(l2e.get("R", np.eye(3))), np.array(l2e.get("t", [0, 0, 0]), float)


# --------------------------------------------------------------------------- cameras
def convert_camera(name, cam, args, ctx, l2e):
    W, H = cam["image_size_wh"]
    if "intrinsics" in cam:
        fx, fy = cam["intrinsics"]["fx"], cam["intrinsics"]["fy"]
        cx, cy = cam["intrinsics"]["cx"], cam["intrinsics"]["cy"]
    else:
        K = np.array(cam["K"], float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # --- FOV (CARLA fov is the HORIZONTAL fov; vertical follows from the aspect ratio)
    fov = None
    if args.fov_source == "distortion_aware":
        fov = (cam.get("fov_deg") or {}).get("hfov_distortion_aware_deg")
        if fov is None:
            ctx.notes.append(f"{name}: distortion-aware FOV not available in the YAML, used pinhole FOV.")
    if fov is None:
        fov = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    if abs(fx - fy) / fx > 1e-3:
        ctx.notes.append(f"{name}: fx != fy; CARLA assumes square pixels (vertical FOV will differ).")

    # --- pose: sensor -> lidar -> ego (identity for NAVSIM, composed anyway so a non-identity lidar2ego stays correct)
    Rle, tle = l2e
    R = Rle @ orthonormalize(cam["sensor2lidar"]["R"])
    t = Rle @ np.array(cam["sensor2lidar"]["t"], float) + tle
    Rc, tc = ego_pose_to_carla(R @ M_CARLACAM_TO_CV, t, ctx, left_handed_local=True)

    # --- principal point: CARLA's principal point is always the image centre.
    dx_right = (W / 2.0 - cx) / fx
    dz_up = (cy - H / 2.0) / fy
    if abs(cx - W / 2.0) > 0.5 or abs(cy - H / 2.0) > 0.5:
        if args.compensate_principal_point:
            Rc = Rc @ carla_matrix(0.0, math.degrees(math.atan(dz_up)), math.degrees(math.atan(dx_right)))
        else:
            ctx.pp_offsets.append((name, cx - W / 2.0, cy - H / 2.0))

    attrs = {
        "image_size_x": int(round(W * args.scale)),
        "image_size_y": int(round(H * args.scale)),
        "fov": rnd(fov, 4),
        "sensor_tick": rnd(1.0 / args.camera_hz, 4),
    }
    if args.lens_k is not None:       # optional manual passthrough, see the notes: NOT derived from OpenCV k1..k3
        attrs["lens_k"] = args.lens_k
    if args.lens_kcube is not None:
        attrs["lens_kcube"] = args.lens_kcube
    return {"id": name, "type": "sensor.camera.rgb", "transform": transform_dict(Rc, tc), "attributes": attrs}


# --------------------------------------------------------------------------- the ONE merged LiDAR
def _lidar_attrs(channels, rng, pps, hz, upper, lower, hfov):
    return {
        "channels": int(channels),
        "range": rnd(rng, 1),
        "points_per_second": int(round(pps)),
        "rotation_frequency": rnd(hz, 3),
        "upper_fov": rnd(upper, 3),
        "lower_fov": rnd(lower, 3),
        "horizontal_fov": rnd(hfov, 3),
        "atmosphere_attenuation_rate": 0.0,
        "dropoff_general_rate": 0.0,
        "dropoff_intensity_limit": 1.0,
        "dropoff_zero_intensity": 0.0,
        "noise_stddev": 0.0,
        "sensor_tick": 0.0,
    }


def pick_top_lidar(fit):
    """-> (id, entry) of the top LiDAR in a lidar_fit.yaml: the roles written by the fit, else highest central 40-ch unit."""
    L = (fit or {}).get("lidars", {})
    tid = (fit or {}).get("top_lidar_id")
    if tid is not None and str(tid) in L:
        return str(tid), L[str(tid)]
    cand = [(v["mount_xyz_ego_m"][2], k, v) for k, v in L.items()
            if v.get("n_rings", 0) >= 30 and abs(v["mount_xyz_ego_m"][1]) < 0.5]
    if not cand:
        return None, None
    _, k, v = max(cand, key=lambda c: c[0])
    return str(k), v


def resolve_pps(args, fit, lm, hz, ctx):
    """Rays per second of the single merged sensor."""
    if args.lidar_pps:
        return float(args.lidar_pps), "--lidar-pps"
    ppf, src = None, None
    L = (fit or {}).get("lidars", {})
    if fit and fit.get("total_points_per_frame_mean"):
        ppf, src = float(fit["total_points_per_frame_mean"]), "fit (sum of 5 LiDARs, measured)"
    elif L:
        ppf, src = sum(v.get("points_per_frame_mean", 0.0) for v in L.values()), "fit (sum of 5 LiDARs, measured)"
    elif isinstance(lm, dict) and lm.get("points_per_frame"):
        ppf, src = float(lm["points_per_frame"]["median"]), "extractor lidar_merged median"
    if not ppf:
        ppf, src = PAPER_PPS_TOTAL / PAPER_HZ, "paper Table VII sum"
    pps = ppf * hz
    if pps > args.lidar_pps_cap:
        ctx.notes.append(f"LIDAR_MERGED points_per_second capped {pps:,.0f} -> {args.lidar_pps_cap:,.0f} "
                         f"(--lidar-pps-cap) to keep the ray-casting cheap.")
        pps = float(args.lidar_pps_cap)
    return pps, src


def convert_lidar_merged(args, ctx, fit, lm):
    """ONE sensor.lidar.ray_cast named LIDAR_MERGED."""
    hz = args.lidar_hz
    pos, src, top_id = None, None, None
    if args.lidar_mount:
        pos, src = list(args.lidar_mount), "--lidar-mount"
    else:
        top_id, top = pick_top_lidar(fit)
        if top is not None:
            pos, src = top["mount_xyz_ego_m"], f"fitted top LiDAR (lidar_id {top_id})"
            q = top.get("quality", {})
            if q.get("height_well_determined") is False:
                ctx.notes.append("[CHECK] the fit says the top LiDAR height is NOT well determined "
                                 "(z_profile_ratio <= 2): verify the z of LIDAR_MERGED by hand.")
            pc = top.get("paper_check", {})
            if pc.get("consistent_with_paper") is False:
                ctx.notes.append(f"[CHECK] fitted elevation range {top.get('elevation_min_deg')}.."
                                 f"{top.get('elevation_max_deg')} deg is not consistent with the paper "
                                 f"({pc.get('paper_fov_deg')}): the fit or the lidar_id selection may be wrong.")
            if q.get("level_assumption_ok") is False:
                ctx.notes.append(f"[CHECK] top LiDAR looks tilted ({q.get('tilt_amplitude_deg')} deg): the fit assumed a "
                                 "level sensor, apex/elevations may be biased.")
    if pos is None:
        sys.exit("No LiDAR mounting position available. Run fit_lidar_rig.py and pass --lidar-fit lidar_fit.yaml, "
                 "or pass --lidar-mount X Y Z (nuPlan ego frame, metres), or use --no-lidar. "
                 "(Refusing to place the LiDAR at the ego origin: it would sit inside the car body.)")
    pps, pps_src = resolve_pps(args, fit, lm, hz, ctx)
    Rc, tc = ego_pose_to_carla(np.eye(3), pos, ctx)   # fit points are already in the ego frame; sensor assumed level
    ctx.notes.append(
        f"LIDAR_MERGED: the 5 physical LiDARs are represented by ONE ray_cast sensor at the {src}; "
        f"{TOP_CHANNELS} ch, [{TOP_LOWER:+.0f}, {TOP_UPPER:+.0f}] deg, 360 deg (paper Table VII top Pandar40P); "
        f"points_per_second from {pps_src}. Bumper and A-pillar viewpoints / near-field blind spots are NOT modelled; "
        "CARLA spaces channels uniformly (real Pandar40P: denser near the horizon); rays with no hit are dropped.")
    return {"id": "LIDAR_MERGED", "type": "sensor.lidar.ray_cast", "transform": transform_dict(Rc, tc),
            "attributes": _lidar_attrs(TOP_CHANNELS, args.lidar_range, pps, hz, TOP_UPPER, TOP_LOWER, 360.0)}


# --------------------------------------------------------------------------- main
def select_calibration(doc, vehicle, index):
    vehicles = doc["vehicles"]
    if vehicle is None:
        vehicle = next(iter(vehicles))
    if vehicle not in vehicles:
        sys.exit(f"vehicle '{vehicle}' not found; available: {', '.join(vehicles)}")
    calibs = vehicles[vehicle]["calibrations"]
    if index >= len(calibs):
        sys.exit(f"calibration index {index} out of range (vehicle has {len(calibs)})")
    return vehicle, calibs[index], vehicles[vehicle]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="rig YAML written by extract_rig.py")
    ap.add_argument("-o", "--output", type=Path, default=Path("carla_sensors.yaml"))
    ap.add_argument("--vehicle", help="vehicle key (default: first, e.g. veh-35)")
    ap.add_argument("--calibration", type=int, default=0, help="index in the vehicle's calibrations list")
    ap.add_argument("--cameras", nargs="*", help="only these cameras (default: all 8)")
    # --- LiDAR (always ONE merged sensor)
    ap.add_argument("--lidar-fit", type=Path, help="lidar_fit.yaml from fit_lidar_rig.py (position + point budget)")
    ap.add_argument("--lidar-mount", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="explicit LiDAR position in the nuPlan ego frame (x fwd, y LEFT, z up, metres); overrides the fit")
    ap.add_argument("--no-lidar", action="store_true", help="cameras only")
    ap.add_argument("--lidar-hz", type=float, default=20.0,
                    help="rotation frequency = 1/fixed_delta_seconds (nuPlan LiDARs: 20 Hz; 10 halves the per-tick cost)")
    ap.add_argument("--lidar-range", type=float, default=100.0, help="metres (paper 200; range drives the cost)")
    ap.add_argument("--lidar-pps", type=float, help="override points_per_second")
    ap.add_argument("--lidar-pps-cap", type=float, default=1_000_000.0, help="upper bound for points_per_second")
    # --- cameras
    ap.add_argument("--camera-hz", type=float, default=10.0, help="camera rate (nuPlan: 10 Hz) -> sensor_tick = 1/hz")
    ap.add_argument("--scale", type=float, default=1.0, help="image size scale (FOV unchanged), e.g. 0.5")
    ap.add_argument("--fov-source", choices=["pinhole", "distortion_aware"], default="pinhole")
    ap.add_argument("--compensate-principal-point", action="store_true",
                    help="approximate the principal-point offset by a small camera rotation")
    ap.add_argument("--lens-k", type=float, help="CARLA lens_k for every camera (manual, no OpenCV mapping is derived)")
    ap.add_argument("--lens-kcube", type=float, help="CARLA lens_kcube for every camera (manual)")
    # --- frame
    ap.add_argument("--origin-offset", nargs=3, type=float, default=[-REAR_AXLE_TO_CENTER, 0.0, 0.0],
                    metavar=("X", "Y", "Z"),
                    help="shift ADDED to every sensor in the CARLA vehicle frame (x fwd, y right, z up). nuPlan origin is "
                         "the rear axle, CARLA's is the vehicle centre ~1.461 m AHEAD of it, so the shift is NEGATIVE "
                         "(default -1.461 0 0). Pass 0 0 0 to disable.")
    ap.add_argument("--ground-z", type=float,
                    help="ego-frame z of the road (default: fit ground_z_m, else extractor ground_z_estimate_m)")
    ap.add_argument("--no-ground-offset", action="store_true",
                    help="do not raise sensors so that heights are above the road")
    ap.add_argument("--vehicle-blueprint", default="vehicle.chrysler.pacifica")
    args = ap.parse_args()

    doc = yaml.safe_load(args.input.read_text())
    vehicle, calib, vdata = select_calibration(doc, args.vehicle, args.calibration)

    # extract_rig.py writes lidar_merged at VEHICLE level (sibling of `calibrations`); accept both layouts
    lm = calib.get("lidar_merged") or vdata.get("lidar_merged")
    if isinstance(lm, str):
        print(f"[warn] lidar_merged in the YAML is a message, not data: {lm}")
        lm = None
    fit = None
    if args.lidar_fit:
        fit = yaml.safe_load(args.lidar_fit.read_text()) or {}
        fv = fit.get("vehicle")
        if fv and vehicle and str(fv).lower().replace("-", "") != str(vehicle).lower().replace("-", ""):
            print(f"[warn] lidar fit was made for {fv}, converting {vehicle}")

    # ---- ground height: ONE source, no silent default
    ground_z, gsrc = None, None
    if not args.no_ground_offset:
        if args.ground_z is not None:
            ground_z, gsrc = args.ground_z, "--ground-z"
        elif fit and fit.get("ground_z_m") is not None:
            ground_z, gsrc = fit["ground_z_m"], "lidar fit ground_z_m"
        elif isinstance(lm, dict) and lm.get("ground_z_estimate_m") is not None:
            ground_z, gsrc = lm["ground_z_estimate_m"], "extractor ground_z_estimate_m"
        else:
            sys.exit("No ground height available (need --lidar-fit, an extractor run with --lidar, or --ground-z). "
                     "Use --no-ground-offset to skip the road-level shift.")
    ctx = Ctx(args, ground_z)
    l2e = get_lidar2ego(calib, ctx)

    sensors = []
    for name, cam in calib["cameras"].items():
        if args.cameras and name not in args.cameras:
            continue
        sensors.append(convert_camera(name, cam, args, ctx, l2e))
    if not args.no_lidar:
        sensors.append(convert_lidar_merged(args, ctx, fit, lm))

    # ---- notes
    if ctx.pp_offsets:
        ctx.notes.append(
            "principal point is not at the image centre (CARLA cannot model that, ignored; "
            "--compensate-principal-point approximates it by a rotation): "
            + ", ".join(f"{n} ({dx:+.0f},{dy:+.0f}) px" for n, dx, dy in ctx.pp_offsets))
    if ground_z is not None:
        ctx.notes.append(f"z raised by {-ground_z:.3f} m ({gsrc} = {ground_z}) so heights are above the road.")
    ctx.notes.append(f"x/y/z shift in the CARLA frame: {ctx.offset.round(3).tolist()} "
                     "(rear axle -> CARLA vehicle origin, plus road level). The CARLA Pacifica body differs slightly "
                     "from the real car: check that the sensors sit on the body.")
    ctx.notes.append(
        "Lens distortion: the OpenCV k1,k2,p1,p2,k3 model is not exported. CARLA's RGB camera has lens_k / lens_kcube / "
        "lens_x_size / lens_y_size / lens_circle_* (a different model, no verified mapping), settable via --lens-k / "
        "--lens-kcube. The nominal k1 = -0.356 is strong barrel distortion: either undistort the dataset images before "
        "comparing, or accept a pinhole mismatch towards the image borders.")

    n_lidar = sum(1 for s in sensors if s["type"] == "sensor.lidar.ray_cast")
    assert n_lidar <= 1, "internal error: more than one LiDAR in the output"

    out = {
        "carla_version": "0.9.16",
        "source": {"file": args.input.name, "vehicle": vehicle,
                   "lidar_fit": args.lidar_fit.name if args.lidar_fit else None,
                   "frame": "vehicle-relative, CARLA left-handed (x fwd, y right, z up), degrees"},
        "simulation": {"synchronous_mode": True, "fixed_delta_seconds": rnd(1.0 / args.lidar_hz, 6),
                       "note": "LiDAR rotation_frequency == 1/fixed_delta_seconds; cameras use sensor_tick"},
        "vehicle": {"blueprint": args.vehicle_blueprint},
        "sensors": sensors,
        "notes": ctx.notes,
    }
    with open(args.output, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False, width=140)
    print(f"wrote {args.output}  ({len(sensors) - n_lidar} cameras + {n_lidar} merged LiDAR)")
    for n in ctx.notes:
        print("  note:", n)


if __name__ == "__main__":
    main()