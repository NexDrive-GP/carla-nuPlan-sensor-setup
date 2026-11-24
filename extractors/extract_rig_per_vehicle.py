#!/usr/bin/env python3
"""
Extract the NAVSIM / OpenScene sensor rig from the log pickles -> rig.yaml

ONE ENTRY PER VEHICLE, nothing is merged or averaged across vehicles.
Inside a vehicle, logs that carry the *same* calibration (<= 1e-6) share one entry
(their names are listed under `logs`); a log whose calibration differs gets its own entry.

Needs only numpy + Pillow (+ PyYAML if available, else writes JSON) for the camera part.
The optional LiDAR part (--lidar) needs the navsim devkit environment.

Usage:
  python extract_rig.py \
      --logs  /work/dataset/navsim_logs/test \
      --blobs /work/dataset/sensor_blobs/test \
      --lidar [--vehicle veh-35] [--lidar-samples 20] [--out rig.yaml]
      [--one-log-per-vehicle] [--max-logs N]

--vehicle accepts "veh-35", "35", "veh-08" or "8" and processes only that vehicle's logs.
"""
import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CAMS = ["CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"]
EYE_TOL = 1e-9   # lidar2ego is called "identity" below this
SAME_TOL = 1e-6  # two logs of one vehicle share a calibration entry below this
# the intrinsics set that shows up identically on every camera of most vehicles (see intrinsics_check)
NOMINAL_INTR = (1545.0, 1545.0, 960.0, 560.0)  # fx, fy, cx, cy
NOMINAL_DIST = (-0.356123, 0.172545, -0.00213, 0.000464, -0.05231)


# ----------------------------------------------------------------------------- math
def quat_to_R(q):
    """(w, x, y, z) quaternion -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def as_R(r):
    r = np.asarray(r, float)
    return r if r.shape == (3, 3) else quat_to_R(r.reshape(4))


def lst(a, nd=9):
    return np.round(np.asarray(a, float), nd).tolist()


def r4(x):
    return None if x is None else round(float(x), 4)


def to_py(o):
    """Recursively convert numpy scalars/arrays to plain Python so YAML/JSON can dump them."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


# ----------------------------------------------------------------------------- frame / calibration access
def vnum(name):
    """Vehicle number from a file name ('..._veh-35_...' -> 35) or None."""
    m = re.search(r"veh-?(\d+)", name)
    return int(m.group(1)) if m else None


def vkey(p):
    n = vnum(p.name)
    return n if n is not None else p.name


def get_lidar2ego(fr):
    """-> (R, t, source) or None. Prefers the 4x4 'lidar2ego', falls back to the rotation/translation keys."""
    T = fr.get("lidar2ego")
    if T is not None:
        T = np.asarray(T, float)
        if T.shape == (4, 4):
            return T[:3, :3], T[:3, 3], "lidar2ego(4x4)"
    r, t = fr.get("lidar2ego_rotation"), fr.get("lidar2ego_translation")
    if r is not None and t is not None:
        return as_R(r), np.asarray(t, float).reshape(3), "lidar2ego_rotation/translation"
    return None


def calib_vec(fr):
    """Every calibration number of one frame as a flat vector (used to compare frames / logs)."""
    parts = []
    for cam in CAMS:
        c = fr["cams"][cam]
        parts += [np.asarray(c["cam_intrinsic"], float).ravel(),
                  np.asarray(c["distortion"], float).ravel(),
                  as_R(c["sensor2lidar_rotation"]).ravel(),
                  np.asarray(c["sensor2lidar_translation"], float).reshape(3)]
    l2e = get_lidar2ego(fr)
    if l2e is not None:
        parts += [l2e[0].ravel(), l2e[1]]
    return np.concatenate(parts)


def within_log_dev(frames):
    """Max abs deviation of any calibration number between the first frame and every other frame of a log."""
    v0 = calib_vec(frames[0])
    dev = 0.0
    for fr in frames[1:]:
        v = calib_vec(fr)
        if v.shape != v0.shape:
            return float("inf")
        dev = max(dev, float(np.abs(v - v0).max()))
    return dev


def vehicle_of(frame0, pkl_path):
    v = frame0.get("vehicle_name")
    if v:
        return str(v)
    n = vnum(pkl_path.name)
    return f"veh-{n:02d}" if n is not None else pkl_path.stem


# ----------------------------------------------------------------------------- camera model
def _pad5(dist):
    d = np.zeros(5)
    d[: min(5, len(dist))] = np.asarray(dist, float)[:5]
    return d


def distort_norm(x, y, dist):
    """OpenCV model [k1, k2, p1, p2, k3] on normalised coordinates."""
    k1, k2, p1, p2, k3 = _pad5(dist)
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def undistort_norm(xd, yd, dist, iters=100):
    """Fixed-point inverse of distort_norm (same scheme as cv2.undistortPoints)."""
    k1, k2, p1, p2, k3 = _pad5(dist)
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    return x, y


def pixels_to_rays(u, v, K, dist):
    """Raw-image pixels -> undistorted normalised ray (x, y, 1) + worst re-projection residual in px."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xd, yd = (u - cx) / fx, (v - cy) / fy
    with np.errstate(all="ignore"):
        x, y = undistort_norm(xd, yd, dist)
        xr, yr = distort_norm(x, y, dist)
        res = float(np.max(np.hypot((xr - xd) * fx, (yr - yd) * fy)))  # NaN propagates -> "not ok"
    return x, y, res


def fov_block(K, dist, W, H, R):
    """Pinhole FOV (ignores distortion) and distortion-aware FOV of the RAW image."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    deg = math.degrees
    out = {
        # principal-point-aware pinhole formulas (valid only if the image were undistorted)
        "hfov_pinhole_deg": r4(deg(math.atan((W - cx) / fx) + math.atan(cx / fx))),
        "vfov_pinhole_deg": r4(deg(math.atan((H - cy) / fy) + math.atan(cy / fy))),
        "hfov_distortion_aware_deg": None,
        "vfov_distortion_aware_deg": None,
        "max_angle_from_axis_deg": None,   # half-angle of the smallest cone around the optical axis holding the whole image
        "undistort_max_residual_px": None,
    }
    az_edges = (None, None)
    n = 401
    us, vs = np.linspace(0, W, n), np.linspace(0, H, n)
    u = np.concatenate([[0.0, W, cx, cx], us, us, np.zeros(n), np.full(n, W)])
    v = np.concatenate([[cy, cy, 0.0, H], np.zeros(n), np.full(n, H), vs, vs])
    x, y, res = pixels_to_rays(u, v, K, dist)
    out["undistort_max_residual_px"] = r4(res) if np.isfinite(res) else None
    if len(dist) <= 5 and np.all(np.isfinite(x)) and np.isfinite(res) and res < 1.0:
        out["hfov_distortion_aware_deg"] = r4(deg(math.atan(x[1]) - math.atan(x[0])))   # row v = cy
        out["vfov_distortion_aware_deg"] = r4(deg(math.atan(y[3]) - math.atan(y[2])))   # column u = cx
        out["max_angle_from_axis_deg"] = r4(deg(math.atan(float(np.hypot(x[4:], y[4:]).max()))))
        edges = []
        for i in (0, 1):  # left edge (u=0), right edge (u=W) of the principal row
            d = R @ np.array([x[i], y[i], 1.0])
            edges.append(r4(deg(math.atan2(d[1], d[0]))))
        az_edges = tuple(edges)
    return out, az_edges


def find_image(blobs, frames, cam, max_scan=400):
    """First existing image file of a camera in this log (works with partially downloaded blobs)."""
    for fr in frames[:max_scan]:
        p = blobs / fr["cams"][cam]["data_path"]
        if p.exists():
            return p
    return None


def image_size(blobs, frames, cam, K, size_cache):
    p = find_image(blobs, frames, cam)
    if p is not None:
        from PIL import Image
        with Image.open(p) as im:
            W, H = im.size
        size_cache[cam] = (W, H)
        return (W, H), "image_file"
    if cam in size_cache:
        return size_cache[cam], "borrowed_from_other_log"
    return (int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))), "ESTIMATED_from_principal_point_UNRELIABLE"


def cam_spec(c, size, size_src):
    K = np.asarray(c["cam_intrinsic"], float).reshape(3, 3)
    dist = np.asarray(c["distortion"], float).ravel()
    R = as_R(c["sensor2lidar_rotation"])
    t = np.asarray(c["sensor2lidar_translation"], float).reshape(3)
    W, H = size
    fov, (az_l, az_r) = fov_block(K, dist, W, H, R)
    fwd = R @ np.array([0.0, 0.0, 1.0])  # assumes OpenCV-style camera: z forward, x right, y down
    return {
        "image_size_wh": [W, H],
        "image_size_source": size_src,
        "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
        "principal_point_offset_px": [r4(K[0, 2] - W / 2), r4(K[1, 2] - H / 2)],
        "K": lst(K),
        "distortion": lst(dist),
        "fov_deg": fov,
        "sensor2lidar": {"R": lst(R), "t": lst(t)},
        # optical axis in the LiDAR frame (azimuth CCW from +x if y is left)
        "optical_axis_in_lidar": {
            "azimuth_deg": r4(math.degrees(math.atan2(fwd[1], fwd[0]))),
            "elevation_deg": r4(math.degrees(math.asin(float(np.clip(fwd[2], -1, 1))))),
            # tilt of the camera x-axis (image right) out of the horizontal plane; + = right side up
            "roll_deg": r4(math.degrees(math.asin(float(np.clip(R[2, 0], -1, 1))))),
        },
        # azimuth in the LiDAR frame of the left (u=0) / right (u=W) image edge on the principal row, wrapped to [-180, 180]
        "azimuth_left_edge_in_lidar_deg": az_l,
        "azimuth_right_edge_in_lidar_deg": az_r,
    }


# ----------------------------------------------------------------------------- per-calibration checks
def intrinsics_check(f0):
    """Are the intrinsics really per-camera? Groups cameras whose K + distortion are identical."""
    groups, nominal = {}, True
    for cam in CAMS:
        c = f0["cams"][cam]
        K = np.asarray(c["cam_intrinsic"], float).reshape(3, 3)
        d = np.asarray(c["distortion"], float).ravel()
        groups.setdefault(tuple(np.round(np.concatenate([K.ravel(), d]), 9)), []).append(cam)
        is_nom = (np.abs(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) - NOMINAL_INTR).max() < 1e-9
                  and len(d) == 5 and np.abs(d - NOMINAL_DIST).max() < 1e-9)
        nominal = nominal and bool(is_nom)
    g = list(groups.values())
    return {"all_cameras_equal_nominal_1545_960_560_set": nominal,
            "num_distinct_intrinsics_sets": len(g),
            "cameras_sharing_identical_intrinsics": [x for x in g if len(x) > 1]}


def circular_empty_arcs(occ, bin_deg, min_gap_deg):
    """occ: bool per bin over [-180, 180). Runs of False >= min_gap_deg as [start, end] degrees, CCW."""
    nb = len(occ)
    if occ.all():
        return []
    if not occ.any():
        return [[-180.0, 180.0]]
    start = int(np.argmax(occ))
    empty = np.roll(~occ, -start)
    arcs, i = [], 0
    while i < nb:
        if empty[i]:
            j = i
            while j < nb and empty[j]:
                j += 1
            if (j - i) * bin_deg >= min_gap_deg:
                arcs.append([r4(((i + start) % nb) * bin_deg - 180), r4(((j + start) % nb) * bin_deg - 180)])
            i = j
        else:
            i += 1
    return arcs


def camera_coverage(cams):
    """Azimuth coverage / neighbour overlap of the 8 cameras from their principal-row edge azimuths.
    Parallax between camera positions is ignored."""
    res, n = 0.1, 3600
    grid = (np.arange(n) + 0.5) * res - 180
    masks = {}
    for cam in CAMS:
        r, l = cams[cam]["azimuth_right_edge_in_lidar_deg"], cams[cam]["azimuth_left_edge_in_lidar_deg"]
        if r is None or l is None:
            continue
        masks[cam] = ((grid - r) % 360) <= ((l - r) % 360)  # CCW from right edge to left edge
    if not masks:
        return None
    count = sum(m.astype(int) for m in masks.values())
    pairs = {}
    names = list(masks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = float((masks[a] & masks[b]).sum() * res)
            if ov > 0.05:
                pairs[f"{a}|{b}"] = r4(ov)
    return {"basis": "principal-row edge azimuths in the lidar frame, parallax ignored",
            "covered_deg": r4((count > 0).sum() * res),
            "uncovered_arcs_deg": circular_empty_arcs(count > 0, res, 0.5),
            "pair_overlap_deg": pairs}


def build_entry(f0, blobs, frames, size_cache):
    """Calibration entry for one distinct calibration of one vehicle (taken from frame 0 of the first log that has it)."""
    entry, warnings = {}, []
    l2e = get_lidar2ego(f0)
    if l2e is None:
        entry["lidar2ego"] = None
        entry["frame_note"] = "no lidar2ego in the frame dict: ego-frame offset unknown, sensor2ego not written"
    else:
        Rle, tle, src = l2e
        ident = bool(np.abs(Rle - np.eye(3)).max() < EYE_TOL and np.abs(tle).max() < EYE_TOL)
        entry["lidar2ego"] = {"R": lst(Rle), "t": lst(tle), "is_identity": ident, "source_key": src}
        if src == "lidar2ego(4x4)" and f0.get("lidar2ego_rotation") is not None \
                and f0.get("lidar2ego_translation") is not None:
            R2, t2 = as_R(f0["lidar2ego_rotation"]), np.asarray(f0["lidar2ego_translation"], float).reshape(3)
            if np.abs(R2 - Rle).max() > 1e-6 or np.abs(t2 - tle).max() > 1e-6:
                warnings.append("lidar2ego 4x4 and lidar2ego_rotation/translation disagree")
        if ident:
            entry["frame_note"] = ("lidar2ego is identity: LiDAR frame == ego frame, "
                                   "so sensor2lidar is also sensor2ego (not duplicated)")
    entry["intrinsics_check"] = intrinsics_check(f0)
    cams = {}
    for cam in CAMS:
        c = f0["cams"][cam]
        K = np.asarray(c["cam_intrinsic"], float).reshape(3, 3)
        size, src = image_size(blobs, frames, cam, K, size_cache)
        spec = cam_spec(c, size, src)
        R = as_R(c["sensor2lidar_rotation"])
        if np.abs(R @ R.T - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1) > 1e-3:
            warnings.append(f"{cam}: sensor2lidar rotation is not a proper rotation matrix")
        if src != "image_file":
            warnings.append(f"{cam}: image size is {src}")
        if l2e is not None and not entry["lidar2ego"]["is_identity"]:
            t = np.array(spec["sensor2lidar"]["t"])
            spec["sensor2ego"] = {"R": lst(Rle @ R), "t": lst(Rle @ t + tle)}
        cams[cam] = spec
    entry["camera_azimuth_coverage"] = camera_coverage(cams)
    entry["cameras"] = cams
    if warnings:
        entry["warnings"] = warnings
    return entry


# ----------------------------------------------------------------------------- lidar
def azimuth_coverage(counts, min_gap_deg=5):
    """counts: points per 1-degree azimuth bin (accumulated over the sampled frames)."""
    counts = np.asarray(counts)
    nz = counts[counts > 0]
    if nz.size == 0:
        return {"covered_deg": 0, "empty_arcs_deg": [[-180.0, 180.0]]}
    thr = max(3.0, 0.05 * float(np.median(nz)))  # ignore stray returns in blind sectors
    occ = counts >= thr
    return {"covered_deg": int(occ.sum()), "empty_arcs_deg": circular_empty_arcs(occ, 1.0, min_gap_deg),
            "occupancy_threshold_points": r4(thr)}


def ground_estimate(rng, z):
    """Densest 5 cm z-bin among points 3..30 m away, and the nearest return to that ground height."""
    m = (rng > 3) & (rng < 30) & (z > -4) & (z < 4)
    if not m.any():
        return None, None
    hist, edges = np.histogram(z[m], bins=np.arange(-4, 4.05, 0.05))
    k = int(hist.argmax())
    gz = float((edges[k] + edges[k + 1]) / 2)
    g = np.abs(z - gz) < 0.15
    return gz, (float(rng[g].min()) if g.any() else None)


def _mmm(v):
    v = np.asarray(v, float)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


class LidarAcc:
    """Accumulates merged-LiDAR statistics over several frames of ONE vehicle (points are in the ego frame)."""
    SUB = 20000  # max points kept per frame and lidar id for the pooled percentiles

    def __init__(self):
        self.rs = np.random.default_rng(0)
        self.files, self.n_points, self.ground, self.closest, self.errors = [], [], [], [], []
        self.inten = [np.inf, -np.inf]
        self.ids = {}

    def _sub(self, a):
        return a[self.rs.choice(a.size, self.SUB, replace=False)] if a.size > self.SUB else a

    def add(self, rel, pc):
        pc = np.asarray(pc)
        x, y, z, inten, ring, lid = pc[:6]
        rng = np.hypot(x, y)
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arctan2(z, np.maximum(rng, 1e-6)))
        gz, cg = ground_estimate(rng, z)
        if gz is not None:
            self.ground.append(gz)
        if cg is not None:
            self.closest.append(cg)
        self.files.append(str(rel))
        self.n_points.append(int(pc.shape[1]))
        self.inten = [min(self.inten[0], float(inten.min())), max(self.inten[1], float(inten.max()))]
        for i in np.unique(lid):
            s = lid == i
            d = self.ids.setdefault(int(i), {"n": [], "rings": [], "ring_union": set(), "az": np.zeros(360, np.int64),
                                             "el": [], "rng": [], "z": [], "zmin": np.inf, "zmax": -np.inf,
                                             "rmax": 0.0})
            d["n"].append(int(s.sum()))
            d["rings"].append(int(len(np.unique(ring[s]))))
            d["ring_union"] |= set(np.unique(ring[s]).astype(int).tolist())
            d["az"] += np.histogram(az[s], bins=360, range=(-180, 180))[0]
            far = s & (rng > 10.0)  # elevation from the ego origin is only meaningful for far points
            d["el"].append(self._sub(el[far]).astype(np.float32))
            d["rng"].append(self._sub(rng[s]).astype(np.float32))
            d["z"].append(self._sub(z[s]).astype(np.float32))
            d["zmin"], d["zmax"] = min(d["zmin"], float(z[s].min())), max(d["zmax"], float(z[s].max()))
            d["rmax"] = max(d["rmax"], float(rng[s].max()))

    def result(self):
        per_id = {}
        for i, d in sorted(self.ids.items()):
            el, rg, zz = np.concatenate(d["el"]), np.concatenate(d["rng"]), np.concatenate(d["z"])
            per_id[i] = {
                "present_in_frames": len(d["n"]),
                "points_per_frame": _mmm(d["n"]),
                "rings": {"max_per_frame": int(max(d["rings"])), "distinct_ring_ids_overall": len(d["ring_union"])},
                "xy_range_m": {"p50": float(np.percentile(rg, 50)), "p90": float(np.percentile(rg, 90)),
                               "p99": float(np.percentile(rg, 99)), "max": d["rmax"]},
                "z_m": {"min": d["zmin"], "p0.1": float(np.percentile(zz, 0.1)),
                        "p99.9": float(np.percentile(zz, 99.9)), "max": d["zmax"]},
                # angles seen from the EGO ORIGIN (not the sensor), points > 10 m only: approximate vertical extent
                "elevation_from_ego_origin_deg": (
                    {"min": float(el.min()), "p0.5": float(np.percentile(el, 0.5)),
                     "p99.5": float(np.percentile(el, 99.5)), "max": float(el.max())} if el.size else None),
                "azimuth_coverage": azimuth_coverage(d["az"]),
            }
        return {
            "num_frames_sampled": len(self.files),
            "sampled_files": self.files,
            "points_per_frame": _mmm(self.n_points),
            "intensity_range": self.inten,
            "ground_z_estimate_m": (r4(np.median(self.ground)) if self.ground else None),
            "ground_z_estimate_spread_m": ([r4(min(self.ground)), r4(max(self.ground))] if self.ground else None),
            # nearest ground return to the ego origin: includes the vehicle body footprint, NOT a sensor blind zone
            "closest_ground_return_from_ego_origin_m": (r4(np.median(self.closest)) if self.closest else None),
            "per_lidar_id": per_id,
        }


def load_pc(blobs, rel):
    from navsim.common.dataclasses import Lidar  # needs the navsim env
    return Lidar.from_paths(blobs, Path(rel), ["lidar_pc"]).lidar_pc


def pick_lidar_frames(blobs, frames, quota):
    """`quota` evenly spaced frames of this log whose LiDAR file exists on disk."""
    cand = [fr for fr in frames if (blobs / fr["lidar_path"]).exists()]
    if not cand or quota <= 0:
        return []
    idx = np.unique(np.linspace(0, len(cand) - 1, min(quota, len(cand))).round().astype(int))
    return [cand[i] for i in idx]


# ----------------------------------------------------------------------------- reporting
def log_info(p, frames):
    ts = [fr.get("timestamp") for fr in frames]
    dt = None
    if len(ts) > 1 and all(t is not None for t in ts):
        dt = float(np.median(np.diff(np.asarray(ts, float)))) / 1e6  # timestamps are microseconds
    return {"name": p.name, "num_frames": len(frames), "map_location": frames[0].get("map_location"),
            "median_frame_dt_s_assuming_us": r4(dt)}


def print_entry(veh, k, entry):
    logs = entry["logs"]
    l2e = entry["lidar2ego"]
    tag = "none" if l2e is None else ("identity" if l2e["is_identity"] else "NON-IDENTITY")
    print(f"\n=== {veh}  calibration #{k}  ({len(logs)} log(s): {logs[0]['name']}"
          f"{' ...' if len(logs) > 1 else ''})  lidar2ego={tag}")
    devs = [l["within_log_max_abs_dev"] for l in logs]
    if max(devs) > 0:
        print(f"    [note] calibration varies INSIDE a log: max abs dev {max(devs):.3g}")
    ic = entry["intrinsics_check"]
    print(f"    intrinsics: {ic['num_distinct_intrinsics_sets']} distinct set(s) over 8 cameras"
          + ("  [info] the nominal 1545/960/560 set on every camera: probably a placeholder, not a per-camera "
             "calibration" if ic["all_cameras_equal_nominal_1545_960_560_set"] else ""))
    print(f"{'cam':7s} {'WxH':>10s} {'fx':>8s} {'hFOV pin/dist':>14s} {'vFOV pin/dist':>14s} "
          f"{'az':>7s} {'el':>6s} {'roll':>6s}   t_lidar(x,y,z)")
    for cam in CAMS:
        c = entry["cameras"][cam]
        W, H = c["image_size_wh"]
        f, a = c["fov_deg"], c["optical_axis_in_lidar"]
        hd, vd = f["hfov_distortion_aware_deg"], f["vfov_distortion_aware_deg"]
        hs = f"{f['hfov_pinhole_deg']:.1f}/" + (f"{hd:.1f}" if hd is not None else "n/a")
        vs = f"{f['vfov_pinhole_deg']:.1f}/" + (f"{vd:.1f}" if vd is not None else "n/a")
        print(f"{cam:7s} {W:>5d}x{H:<4d} {c['intrinsics']['fx']:8.1f} {hs:>14s} {vs:>14s} "
              f"{a['azimuth_deg']:7.1f} {a['elevation_deg']:6.1f} {a['roll_deg']:6.2f}   {c['sensor2lidar']['t']}")
    cc = entry.get("camera_azimuth_coverage")
    if cc:
        print(f"    camera azimuth coverage {cc['covered_deg']} deg, uncovered arcs {cc['uncovered_arcs_deg']}")
    f0_az = entry["cameras"]["CAM_F0"]["optical_axis_in_lidar"]["azimuth_deg"]
    if abs(f0_az) > 10:
        print(f"    [CHECK] CAM_F0 optical axis azimuth is {f0_az:.1f} deg, expected ~0. "
              "Camera/lidar axis convention differs from the assumption - fix before using this entry.")
    for w in entry.get("warnings", []):
        print(f"    [warn] {w}")


def print_lidar(veh, L):
    if not isinstance(L, dict):
        print(f"\n--- {veh} LiDAR: {L}")
        return
    print(f"\n--- {veh} LiDAR ({L['num_frames_sampled']} frames sampled, "
          f"{L['points_per_frame']['median']:.0f} pts/frame median)")
    print(f"{'id':>3s} {'rings(max)':>10s} {'az covered':>10s} {'empty arcs':<24s} {'elev p0.5..p99.5 (deg)':>24s} "
          f"{'range p99':>10s}")
    for i, d in L["per_lidar_id"].items():
        e = d["elevation_from_ego_origin_deg"]
        es = f"{e['p0.5']:.1f}..{e['p99.5']:.1f}" if e else "n/a"
        print(f"{i:>3d} {d['rings']['max_per_frame']:>10d} {d['azimuth_coverage']['covered_deg']:>10d} "
              f"{str(d['azimuth_coverage']['empty_arcs_deg']):<24s} {es:>24s} {d['xy_range_m']['p99']:>10.1f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="dir with navsim log *.pkl (e.g. .../navsim_logs/test)")
    ap.add_argument("--blobs", required=True, help="sensor_blobs root for that split")
    ap.add_argument("--out", default=None, help="default: rig.yaml, or rig_veh-XX.yaml with --vehicle")
    ap.add_argument("--vehicle", default=None, help='only this vehicle: "veh-35", "35", "veh-08" or "8"')
    ap.add_argument("--max-logs", type=int, default=0, help="0 = all logs")
    ap.add_argument("--one-log-per-vehicle", action="store_true",
                    help="use only the first log of each vehicle (veh-XX in the file name)")
    ap.add_argument("--lidar", action="store_true", help="also compute merged-LiDAR statistics per vehicle")
    ap.add_argument("--lidar-samples", type=int, default=20,
                    help="LiDAR frames per vehicle, evenly spread over its logs (default 20)")
    args = ap.parse_args()

    logs_dir, blobs = Path(args.logs).expanduser(), Path(args.blobs).expanduser()
    if not blobs.is_dir():
        sys.exit(f"sensor blobs dir not found: {blobs}")
    pkls = sorted(logs_dir.glob("*.pkl"))
    if not pkls:
        sys.exit(f"No .pkl logs in {logs_dir}")

    target = None
    if args.vehicle is not None:
        m = re.fullmatch(r"(?:veh-?)?0*(\d+)", args.vehicle.strip())
        if not m:
            sys.exit(f'--vehicle "{args.vehicle}" not understood: use e.g. veh-35 or 35')
        target = int(m.group(1))
        avail = sorted({vnum(p.name) for p in pkls if vnum(p.name) is not None})
        pkls = [p for p in pkls if vnum(p.name) == target]
        if not pkls:
            sys.exit(f"no logs for vehicle {target} in {logs_dir}. Available: "
                     + ", ".join(f"veh-{v:02d}" for v in avail))
    if args.one_log_per_vehicle:
        seen, sel = set(), []
        for p in pkls:
            if vkey(p) not in seen:
                seen.add(vkey(p))
                sel.append(p)
        pkls = sel
    if args.max_logs:
        pkls = pkls[: args.max_logs]

    out = Path(args.out) if args.out else Path(f"rig_veh-{target:02d}.yaml" if target is not None else "rig.yaml")
    logs_left = Counter(vkey(p) for p in pkls)  # logs still to read per vehicle (for the LiDAR sampling quota)

    vehicles, vecs, accs = {}, {}, {}
    size_cache = {}
    first = True
    for p in pkls:
        with open(p, "rb") as fh:
            frames = pickle.load(fh)
        vk = vkey(p)
        if not frames:
            print(f"[warn] {p.name}: empty log, skipped")
            logs_left[vk] -= 1
            continue
        f0 = frames[0]
        if first:
            print(f"Keys of first frame dict:\n  {sorted(f0.keys())}")
            print(f"Keys of f0['cams']['CAM_F0']:\n  {sorted(f0['cams']['CAM_F0'].keys())}")
            first = False
        veh = vehicle_of(f0, p)
        V = vehicles.setdefault(veh, {"num_logs": 0, "calibrations": []})
        vecs.setdefault(veh, [])
        V["num_logs"] += 1

        info = log_info(p, frames)
        info["within_log_max_abs_dev"] = within_log_dev(frames)  # 0.0 = calibration constant inside the log
        v0 = calib_vec(f0)
        constant = info["within_log_max_abs_dev"] <= SAME_TOL
        match = None
        if constant:  # a log whose calibration drifts inside it is never merged with another log
            match = next((k for k, vv in enumerate(vecs[veh])
                          if vv is not None and vv.shape == v0.shape and np.abs(vv - v0).max() <= SAME_TOL), None)
        if match is None:
            entry = {"logs": [info], **build_entry(f0, blobs, frames, size_cache)}
            if not constant:
                entry["note"] = "calibration changes inside this log: values below are from its first frame"
            V["calibrations"].append(entry)
            vecs[veh].append(v0 if constant else None)
        else:
            V["calibrations"][match]["logs"].append(info)

        if args.lidar:
            acc = accs.setdefault(veh, LidarAcc())
            remaining = args.lidar_samples - len(acc.files)
            if remaining > 0 and not acc.errors:
                quota = math.ceil(remaining / max(logs_left[vk], 1))
                for fr in pick_lidar_frames(blobs, frames, quota):
                    try:
                        acc.add(fr["lidar_path"], load_pc(blobs, fr["lidar_path"]))
                    except Exception as e:  # noqa: BLE001
                        acc.errors.append(repr(e))
                        break
        logs_left[vk] -= 1
        del frames

    for veh, V in vehicles.items():
        if args.lidar:
            acc = accs.get(veh)
            if acc is not None and acc.files:
                V["lidar_merged"] = acc.result()
            elif acc is not None and acc.errors:
                V["lidar_merged"] = f"FAILED: {acc.errors[0]}"
            else:
                V["lidar_merged"] = "no lidar file found on disk for this vehicle (download a lidar shard)"
        for k, entry in enumerate(V["calibrations"]):
            print_entry(veh, k, entry)
        if args.lidar:
            print_lidar(veh, V["lidar_merged"])

    rig = {
        "meta": {
            "logs_dir": str(logs_dir),
            "vehicle_filter": None if target is None else f"veh-{target:02d}",
            "num_logs_loaded": sum(V["num_logs"] for V in vehicles.values()),
            "num_vehicles": len(vehicles),
            "lidar_samples_per_vehicle": args.lidar_samples if args.lidar else None,
            "conventions": {
                "sensor2lidar": "p_lidar = R @ p_sensor + t",
                "camera_axes_assumed": "OpenCV: x right, y down, z forward (validated by the CAM_F0 azimuth check)",
                "distortion_assumed": "OpenCV [k1, k2, p1, p2, k3] on the RAW image",
                "lidar_frame": "x forward, y left, z up (assumed); equals the ego frame when lidar2ego.is_identity",
                "fov": "*_pinhole ignores distortion; *_distortion_aware uses undistorted rays of the raw-image border. "
                       "Which one applies depends on whether the stored images are raw or undistorted "
                       "(not recorded in the logs)",
                "intrinsics_check": "cameras sharing identical K + distortion, and whether all 8 equal the nominal "
                                    "1545/960/560 set; a vehicle with the nominal set on all cameras has no individual "
                                    "per-camera intrinsics in these logs",
                "lidar_angles": "elevation/azimuth are measured from the ego origin, not from each sensor; "
                                "per-sensor mounting poses are not in the logs",
                "lidar_sampling": "evenly spaced frames across the vehicle's logs; azimuth coverage uses accumulated "
                                  "1-degree bins with a relative occupancy threshold",
            },
        },
        "vehicles": vehicles,
    }
    rig = to_py(rig)
    try:
        import yaml
        out.write_text(yaml.safe_dump(rig, sort_keys=False, default_flow_style=None, width=200))
    except ImportError:
        out = out.with_suffix(".json")
        out.write_text(json.dumps(rig, indent=2))
    print(f"\nwrote {out}  ({len(vehicles)} vehicle(s))")


if __name__ == "__main__":
    main()
