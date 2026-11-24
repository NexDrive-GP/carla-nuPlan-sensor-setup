#!/usr/bin/env python3
"""
Convert a nuPlan / NAVSIM sensor-calibration YAML (the "veh-35" style dump) into a
CARLA 0.9.16 sensor-config YAML that maps 1:1 onto blueprint attributes + transforms.

Usage
-----
    python nuplan_to_carla_sensors.py calib.yaml -o carla_sensors.yaml
    python nuplan_to_carla_sensors.py calib.yaml --lidar per-id --scale 0.5
    python nuplan_to_carla_sensors.py --help

Output schema
-------------
    carla_version: 0.9.16
    vehicle: {blueprint: vehicle.chrysler.pacifica}
    sensors:
      - id: CAM_F0
        type: sensor.camera.rgb
        transform: {x: .., y: .., z: .., roll: .., pitch: .., yaw: ..}   # relative to the ego vehicle, CARLA frame
        attributes: {image_size_x: .., image_size_y: .., fov: .., sensor_tick: 0.0}
      - id: LIDAR_MERGED
        type: sensor.lidar.ray_cast
        transform: {...}
        attributes: {channels: .., range: .., ...}

Coordinate conventions handled here
-----------------------------------
nuPlan ego/lidar : right-handed, x forward, y LEFT,  z up.   p_ego = R @ p_sensor + t
nuPlan camera    : OpenCV, x right, y down, z forward.
CARLA (Unreal)   : left-handed,  x forward, y RIGHT, z up. Rotation in degrees (roll/pitch/yaw),
                   yaw > 0 = turn right, pitch > 0 = look up.
CARLA camera     : looks along its local +x, local y = right, local z = up.

Conversion:  B = R_sensor2lidar @ M  (columns = CARLA-cam fwd/right/up in the right-handed ego frame)
             cameras: R_carla = S @ B;  lidar: R_carla = S @ R @ S;  t_carla = S @ t,  S = diag(1, -1, 1)
             (roll, pitch, yaw) extracted with CARLA's own Transform.get_matrix() convention.
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


# --------------------------------------------------------------------------- conversion
class Ctx:
    """Conversion options shared by all sensors."""

    def __init__(self, args, ground_z):
        self.offset = np.array(args.origin_offset, float)  # CARLA frame (x fwd, y right, z up)
        if not args.no_ground_offset and ground_z is not None:
            # nuPlan ego origin sits ~|ground_z| above the road; CARLA vehicle origin is at ground level.
            self.offset[2] += -float(ground_z)
        self.notes = []


def ego_pose_to_carla(R_ego, t_ego, ctx, left_handed_local=False):
    """right-handed ego pose -> CARLA pose.
    left_handed_local=True: the sensor's own axes are already CARLA's left-handed (fwd, right, up) basis
    (cameras), so only the parent frame is flipped: Rc = S @ R. Otherwise Rc = S @ R @ S."""
    Rc = S_FLIP_Y @ R_ego if left_handed_local else S_FLIP_Y @ R_ego @ S_FLIP_Y
    tc = S_FLIP_Y @ np.asarray(t_ego, float) + ctx.offset
    return Rc, tc


def transform_dict(Rc, tc):
    roll, pitch, yaw = matrix_to_rpy(Rc)
    # self-check: reconstruct and compare
    err = np.abs(carla_matrix(roll, pitch, yaw) - Rc).max()
    if err > 1e-6:
        raise RuntimeError(f"rotation round-trip error {err:.2e}")
    return Flow(x=rnd(tc[0]), y=rnd(tc[1]), z=rnd(tc[2]),
                roll=rnd(wrap180(roll)), pitch=rnd(pitch), yaw=rnd(wrap180(yaw)))


def convert_camera(name, cam, args, ctx):
    W, H = cam["image_size_wh"]
    K = np.array(cam["K"], float) if "K" in cam else None
    fx = cam["intrinsics"]["fx"] if "intrinsics" in cam else K[0, 0]
    fy = cam["intrinsics"]["fy"] if "intrinsics" in cam else K[1, 1]
    cx = cam["intrinsics"]["cx"] if "intrinsics" in cam else K[0, 2]
    cy = cam["intrinsics"]["cy"] if "intrinsics" in cam else K[1, 2]

    # --- FOV (CARLA fov is the HORIZONTAL fov; vertical follows from the aspect ratio)
    if args.fov_source == "distortion_aware" and "fov_deg" in cam:
        fov = cam["fov_deg"]["hfov_distortion_aware_deg"]
    else:
        fov = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    if abs(fx - fy) / fx > 1e-3:
        ctx.notes.append(f"{name}: fx != fy; CARLA assumes square pixels (vertical FOV will differ).")

    # --- pose
    R = orthonormalize(cam["sensor2lidar"]["R"])
    t = cam["sensor2lidar"]["t"]
    R_ego = R @ M_CARLACAM_TO_CV
    Rc, tc = ego_pose_to_carla(R_ego, t, ctx, left_handed_local=True)

    # --- principal point: CARLA's principal point is always the image centre.
    dx_right = (W / 2.0 - cx) / fx
    dz_up = (cy - H / 2.0) / fy
    if abs(cx - W / 2.0) > 0.5 or abs(cy - H / 2.0) > 0.5:
        if args.compensate_principal_point:
            R_local = carla_matrix(0.0, math.degrees(math.atan(dz_up)), math.degrees(math.atan(dx_right)))
            Rc = Rc @ R_local
        else:
            ctx.notes.append(
                f"{name}: principal point offset ({cx - W / 2:+.1f}, {cy - H / 2:+.1f}) px cannot be modelled by "
                f"CARLA; ignored (use --compensate-principal-point to approximate it by a small rotation).")

    attrs = {
        "image_size_x": int(round(W * args.scale)),
        "image_size_y": int(round(H * args.scale)),
        "fov": rnd(fov, 4),
        "sensor_tick": 0.0,
    }
    return {"id": name, "type": "sensor.camera.rgb",
            "transform": transform_dict(Rc, tc), "attributes": attrs}


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


def convert_lidars(calib, args, ctx):
    lm = calib.get("lidar_merged")
    if not lm:
        ctx.notes.append("No lidar_merged section found; no LiDAR emitted.")
        return []
    l2e = calib.get("lidar2ego", {})
    R_l2e = np.array(l2e.get("R", np.eye(3)), float)
    t_l2e = np.array(l2e.get("t", [0, 0, 0]), float)
    Rc, tc = ego_pose_to_carla(R_l2e, t_l2e, ctx)
    per = lm.get("per_lidar_id", {})
    hz = args.lidar_hz

    def keyf(d, k):  # YAML keys like p0.5 are strings
        return d[k]

    if args.lidar == "merged":
        ppf = lm["points_per_frame"]["median"]
        chans = max(v["rings"]["distinct_ring_ids_overall"] for v in per.values())
        lower = min(keyf(v["elevation_from_ego_origin_deg"], "p0.5") for v in per.values())
        upper = max(keyf(v["elevation_from_ego_origin_deg"], "p99.5") for v in per.values())
        rng = args.lidar_range or math.ceil(max(v["xy_range_m"]["p99"] for v in per.values()) / 5.0) * 5.0
        ctx.notes.append(
            "LIDAR_MERGED: the 5 physical LiDARs are fused into one 360 deg ray_cast sensor at the lidar2ego pose; "
            "channels = max rings (40), vertical FOV = widest p0.5..p99.5 elevation measured from the ego origin.")
        return [{"id": "LIDAR_MERGED", "type": "sensor.lidar.ray_cast",
                 "transform": transform_dict(Rc, tc),
                 "attributes": _lidar_attrs(chans, rng, ppf * hz, hz, upper, lower, 360.0)}]

    # per-id
    out = []
    for lid, v in sorted(per.items(), key=lambda kv: int(kv[0])):
        arcs = v["azimuth_coverage"].get("empty_arcs_deg", [])
        if arcs:
            a, b = arcs[0]
            width = (b - a) % 360.0
            hfov = 360.0 - width
            covered_center_ego = wrap180(a + width / 2.0 + 180.0)   # right-handed, ccw+
            yaw_extra = -covered_center_ego                           # CARLA: y flipped, yaw cw+
        else:
            hfov, yaw_extra = 360.0, 0.0
        Rl = Rc @ carla_matrix(0.0, 0.0, yaw_extra)
        el = v["elevation_from_ego_origin_deg"]
        rng = args.lidar_range or math.ceil(v["xy_range_m"]["p99"] / 5.0) * 5.0
        out.append({"id": f"LIDAR_{lid}", "type": "sensor.lidar.ray_cast",
                    "transform": transform_dict(Rl, tc),
                    "attributes": _lidar_attrs(v["rings"]["distinct_ring_ids_overall"], rng,
                                               v["points_per_frame"]["median"] * hz, hz,
                                               el["p99.5"], el["p0.5"], hfov)})
    ctx.notes.append(
        "per-id LiDARs: logs contain NO per-sensor mounting poses, so all are placed at the lidar2ego origin and "
        "aimed at the centre of their observed azimuth coverage. Use 'merged' unless you know the real mounts.")
    return out


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
    return vehicle, calibs[index]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="nuPlan/NAVSIM calibration YAML")
    ap.add_argument("-o", "--output", type=Path, default=Path("carla_sensors.yaml"))
    ap.add_argument("--vehicle", help="vehicle key (default: first, e.g. veh-35)")
    ap.add_argument("--calibration", type=int, default=0, help="index in the vehicle's calibrations list")
    ap.add_argument("--cameras", nargs="*", help="only these cameras (default: all)")
    ap.add_argument("--lidar", choices=["merged", "per-id", "none"], default="merged")
    ap.add_argument("--lidar-hz", type=float, default=10.0, help="LiDAR rotation frequency (nuPlan: 10 Hz)")
    ap.add_argument("--lidar-range", type=float, help="override LiDAR range in metres")
    ap.add_argument("--scale", type=float, default=1.0, help="image size scale (FOV unchanged), e.g. 0.5")
    ap.add_argument("--fov-source", choices=["pinhole", "distortion_aware"], default="pinhole")
    ap.add_argument("--compensate-principal-point", action="store_true")
    ap.add_argument("--origin-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="extra shift in the CARLA vehicle frame (x fwd, y right, z up), e.g. rear-axle -> actor origin")
    ap.add_argument("--no-ground-offset", action="store_true",
                    help="do not raise sensors by -ground_z_estimate_m (nuPlan ego origin is above the road)")
    ap.add_argument("--vehicle-blueprint", default="vehicle.chrysler.pacifica")
    args = ap.parse_args()

    doc = yaml.safe_load(args.input.read_text())
    vehicle, calib = select_calibration(doc, args.vehicle, args.calibration)
    ground_z = calib.get("lidar_merged", {}).get("ground_z_estimate_m")
    ctx = Ctx(args, ground_z)

    sensors = []
    for name, cam in calib["cameras"].items():
        if args.cameras and name not in args.cameras:
            continue
        sensors.append(convert_camera(name, cam, args, ctx))
    if args.lidar != "none":
        sensors += convert_lidars(calib, args, ctx)

    if not args.no_ground_offset and ground_z is not None:
        ctx.notes.append(f"z raised by {-ground_z:.3f} m (ground_z_estimate_m = {ground_z}) so heights are above road.")
    if ctx.offset[0] == 0 and ctx.offset[1] == 0:
        ctx.notes.append("nuPlan origin is the REAR AXLE; CARLA actor origin is near the vehicle centre. "
                         "Pass --origin-offset X 0 0 (X = rear axle -> CARLA origin, metres, + forward) if needed.")
    ctx.notes.append("Lens distortion is not exported: CARLA has no OpenCV k1,k2,p1,p2,k3 model (pinhole only).")

    out = {
        "carla_version": "0.9.16",
        "source": {"file": args.input.name, "vehicle": vehicle,
                   "frame": "ego/vehicle-relative, CARLA left-handed (x fwd, y right, z up), degrees"},
        "vehicle": {"blueprint": args.vehicle_blueprint},
        "sensors": sensors,
        "notes": ctx.notes,
    }
    with open(args.output, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False, width=140)
    print(f"wrote {args.output}  ({len(sensors)} sensors)")
    for n in ctx.notes:
        print("  note:", n)


if __name__ == "__main__":
    main()
