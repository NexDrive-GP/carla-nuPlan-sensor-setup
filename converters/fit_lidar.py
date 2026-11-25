#!/usr/bin/env python3
"""
Recover the mounting position and beam table of each LiDAR inside the merged NAVSIM/nuPlan cloud.

Why this works
--------------
Every point of the merged cloud carries (lidar_id, ring). A ring is a fixed vertical beam, so ALL points of one
ring of one LiDAR lie on a cone whose apex is that sensor:

    tan(elevation) = (z - z_sensor) / hypot(x - x_sensor, y - y_sensor)   is constant per ring.

We search for the apex (x, y, z) that makes every ring a perfect cone. The cost is METRIC: the vertical distance
(metres) of each point from its ring's cone, median over the ring, weighted by ring size. (An angular cost over-weights
near points.) The search is box-bounded and its z range is relative to the measured road height. Ground points alone
cannot fix the height (degenerate), but walls, cars and poles at different ranges can, so ALL points are used.

Added checks
------------
* tilt check   : amplitude of the first azimuth harmonic of the residual elevation. A pitched / rolled sensor makes the
                 elevation of a ring vary with azimuth; the fit assumes a level sensor, so a large amplitude is flagged.
* roles        : lidar_id -> top_center / a_pillar_left / a_pillar_right / front_bumper / rear_bumper from the FITTED
                 positions (ring count + position), written as `roles` and `top_lidar_id`.
* paper check  : fitted lowest / highest ring vs the paper Table VII vertical FOV (channel count +-2).
* ring table   : {ring_id: elevation_deg}, comparable with a datasheet.

Output (lidar_fit.yaml) is consumed by `nuplan_to_carla_sensors.py --lidar-fit lidar_fit.yaml`
(position of the top LiDAR + measured points per frame; the CARLA output contains ONE merged LiDAR).

Usage (needs the navsim environment):
    python fit_lidar_rig.py --logs /work/dataset/navsim_logs/test --blobs /work/dataset/sensor_blobs/test \
        --vehicle veh-35 --frames 20 --out lidar_fit.yaml
"""
import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

# nuPlan paper Table VII: (model, lower_deg, upper_deg, channels)
PAPER_MODELS = [
    ("Pandar40P (top / A-pillar)", -25.0, 15.0, 40),
    ("Pandar20PA (front bumper)", -25.0, 8.0, 20),
    ("Pandar20PB (rear bumper)", -19.0, 3.0, 20),
]
CH_TOL = 2          # observed rings may exceed the datasheet channel count by at most this many
CH_MISSING = 8      # ... and may fall short by this many (upper rings often return nothing: sky)
BIG_RINGS = 30      # >= this many rings = a 40-channel unit (top / A-pillar), else a 20-channel bumper unit
ROLE_MODEL = {"top_center": 0, "a_pillar_left": 0, "a_pillar_right": 0, "front_bumper": 1, "rear_bumper": 2}
BOUND_TOL = 0.02    # m: a coordinate this close to the search box edge means the optimiser ran away
TILT_OK_DEG = 0.5   # first-harmonic elevation amplitude below which the level-sensor assumption is accepted
Z_RATIO_OK = 1.2    # cost at (z +-15 cm) / cost at the optimum. 1.0 = height unconstrained (ground-only scene);
                    # the cost is a metric residual with a noise floor, so values are small (calibrated on synthetic data)


# ----------------------------------------------------------------------------- core fit
def _prep(x, y, z, ring, per_ring, rs):
    """Subsample per ring and group points by ring. -> (x, y, z, segments) or None."""
    keep = []
    for r in np.unique(ring):
        idx = np.flatnonzero(ring == r)
        if idx.size < 20:
            continue
        if idx.size > per_ring:
            idx = rs.choice(idx, per_ring, replace=False)
        keep.append(idx)
    if not keep:
        return None
    keep = np.concatenate(keep)
    r = ring[keep]
    cut = np.flatnonzero(np.diff(r)) + 1
    starts, ends = np.concatenate([[0], cut]), np.concatenate([cut, [len(r)]])
    return x[keep], y[keep], z[keep], list(zip(starts.tolist(), ends.tolist()))


def _elev(c, x, y, z):
    return np.degrees(np.arctan2(z - c[2], np.hypot(x - c[0], y - c[1])))


def _cost(c, x, y, z, segs):
    """Mean over points of the per-ring median |distance from the ring's cone| in METRES. Smaller = better apex."""
    rho = np.hypot(x - c[0], y - c[1]) + 1e-6
    t = (z - c[2]) / rho                                   # tan(elevation)
    tot = 0.0
    for a, b in segs:
        tb = t[a:b]
        tot += float(np.median(np.abs(rho[a:b] * (tb - np.median(tb))))) * (b - a)
    return tot / x.size


def _refine(c0, data, lo, hi, step0=0.25, step_min=0.004, max_iter=600):
    """Coordinate pattern search inside the box [lo, hi]."""
    c = np.clip(np.array(c0, float), lo, hi)
    f = _cost(c, *data)
    step = step0
    for _ in range(max_iter):
        if step <= step_min:
            break
        improved = False
        for dim in range(3):
            for sgn in (1.0, -1.0):
                t = c.copy()
                t[dim] = np.clip(t[dim] + sgn * step, lo[dim], hi[dim])
                if t[dim] == c[dim]:
                    continue
                ft = _cost(t, *data)
                if ft < f - 1e-9:
                    c, f, improved = t, ft, True
        if not improved:
            step /= 2.0
    return c, f


def azimuth_coverage(az_deg, min_gap=5.0):
    """-> (hfov_deg, center_deg) of the populated arc; center is the middle of the covered arc (ego frame, CCW+)."""
    h = np.histogram(az_deg, bins=360, range=(-180, 180))[0]
    nz = h[h > 0]
    if nz.size == 0:
        return 0.0, 0.0
    occ = h >= max(3.0, 0.05 * float(np.median(nz)))
    if occ.all():
        return 360.0, 0.0
    e2 = np.concatenate([~occ, ~occ])
    best_len, best_start, i = 0, 0, 0
    while i < 360:
        if e2[i]:
            j = i
            while j < i + 360 and e2[j]:
                j += 1
            if j - i > best_len:
                best_len, best_start = j - i, i
            i = j
        else:
            i += 1
    if best_len < min_gap:
        return 360.0, 0.0
    empty_center = -180.0 + best_start + best_len / 2.0
    covered_center = (empty_center + 180.0 + 180.0) % 360.0 - 180.0
    return float(360 - best_len), float(covered_center)


def tilt_amplitude(c, x, y, z, ring):
    """Amplitude (deg) of A*cos(az)+B*sin(az) fitted to the per-ring residual elevation. ~0 for a level sensor."""
    el = _elev(c, x, y, z)
    az = np.arctan2(y - c[1], x - c[0])
    res = np.zeros_like(el)
    ok = np.zeros(el.size, bool)
    for r in np.unique(ring):
        s = ring == r
        if s.sum() < 20:
            continue
        res[s] = el[s] - np.median(el[s])
        ok |= s
    ok &= np.abs(res) < 2.0
    if ok.sum() < 50:
        return float("nan")
    coef = np.linalg.lstsq(np.c_[np.cos(az[ok]), np.sin(az[ok])], res[ok], rcond=None)[0]
    return float(np.hypot(coef[0], coef[1]))


def paper_check(n_rings, lo, up, only=None):
    """Compare the fitted ring range with the datasheet. The LOWER edge must match (ground returns always see it);
    the UPPER edge may be under-observed (rings above the horizon only return from tall objects), so it is only
    required not to exceed the datasheet value."""
    best = None
    for mi, (name, plo, pup, ch) in enumerate(PAPER_MODELS):
        if only is not None and mi != only:
            continue
        if not (-CH_MISSING <= n_rings - ch <= CH_TOL):
            continue
        err = max(abs(lo - plo), abs(up - pup))
        if best is None or err < best["max_edge_error_deg"]:
            best = {"closest_paper_model": name, "paper_fov_deg": [plo, pup], "max_edge_error_deg": round(err, 2),
                    "lower_edge_error_deg": round(abs(lo - plo), 2), "upper_edge_error_deg": round(up - pup, 2)}
    if best is None:
        return {"closest_paper_model": None, "consistent_with_paper": False}
    best["consistent_with_paper"] = bool(best["lower_edge_error_deg"] <= 3.0 and best["upper_edge_error_deg"] <= 3.0)
    best["upper_edge_under_observed"] = bool(best["upper_edge_error_deg"] < -3.0)
    return best


def fit_lidar(x, y, z, ring, n_frames, ground_z=None, seed=0):
    """Fit one LiDAR. x, y, z in the ego frame (nuPlan: x fwd, y left, z up). Returns a dict or None."""
    rs = np.random.default_rng(seed)
    x, y, z, ring = (np.asarray(a, float) for a in (x, y, z, ring))
    ring = ring.astype(int)
    n_total = x.size
    d0 = np.hypot(x, y)
    m = (d0 > 1.0) & (d0 < 80.0) & (z > -3.0) & (z < 15.0)
    x, y, z, ring = x[m], y[m], z[m], ring[m]
    coarse, fine = _prep(x, y, z, ring, 120, rs), _prep(x, y, z, ring, 600, rs)
    if coarse is None or fine is None:
        return None

    # search box: z relative to the measured road height (the ego origin is NOT at ground level)
    zlo, zhi = (ground_z + 0.15, ground_z + 2.8) if ground_z is not None else (-0.2, 2.6)
    lo = np.array([-4.0, -2.0, zlo - 0.05])
    hi = np.array([5.5, 2.0, zhi + 0.4])

    cands = []
    for cx in np.arange(-3.0, 5.01, 0.5):
        for cy in np.arange(-1.5, 1.51, 0.5):
            for zs in np.arange(zlo, zhi + 1e-9, 0.2):
                cands.append((_cost((cx, cy, zs), *coarse), cx, cy, zs))
    cands.sort()
    best = None
    for _, cx, cy, zs in cands[:6]:
        c, f = _refine((cx, cy, zs), fine, lo, hi)
        if best is None or f < best[1]:
            best = (c, f)
    c, f = best

    # z-profile check: how sharply does the cost rise if the height is wrong by 15 cm?
    up = _cost(c + [0, 0, 0.15], *fine)
    dn = _cost(c - [0, 0, 0.15], *fine)
    ratio = (up + dn) / 2.0 / max(f, 1e-6)
    axis_ratio = {"z": ratio}
    for dim, name in ((0, "x"), (1, "y")):          # same test along x and y (+-30 cm): bumper units see a half-space
        e = np.zeros(3)                              # and trade apex position along the view axis against height
        e[dim] = 0.30
        axis_ratio[name] = (_cost(c + e, *fine) + _cost(c - e, *fine)) / 2.0 / max(f, 1e-6)
    at_bound = [n for n, i in (("x", 0), ("y", 1), ("z", 2)) if min(c[i] - lo[i], hi[i] - c[i]) < BOUND_TOL]

    el = _elev(c, x, y, z)
    rings = []
    for r in np.unique(ring):
        s = ring == r
        if s.sum() < 20:
            continue
        v = el[s]
        rings.append((float(np.median(v)), int(r), float(np.median(np.abs(v - np.median(v))))))
    if not rings:
        return None
    rings.sort()
    elev = [round(e, 3) for e, _, _ in rings]
    hfov, center = azimuth_coverage(np.degrees(np.arctan2(y - c[1], x - c[0])))
    tilt = tilt_amplitude(c, x, y, z, ring)
    return {
        "mount_xyz_ego_m": [round(float(v), 3) for v in c],
        "n_rings": len(rings),
        "ring_elevation_deg": {r: round(e, 3) for e, r, _ in rings},     # {ring_id: elevation}, ordered by elevation
        "elevation_min_deg": elev[0],
        "elevation_max_deg": elev[-1],
        "azimuth": {"hfov_deg": hfov, "center_deg_ego": round(center, 1)},
        "points_per_frame_mean": round(n_total / n_frames, 1),
        "quality": {
            "cone_residual_m": round(f, 4),
            "ring_elevation_mad_deg": round(float(np.mean([mad for _, _, mad in rings])), 4),
            "z_profile_ratio": round(float(ratio), 2),
            "x_profile_ratio": round(float(axis_ratio["x"]), 2),
            "y_profile_ratio": round(float(axis_ratio["y"]), 2),
            "height_well_determined": bool(ratio > Z_RATIO_OK),
            "xy_well_determined": bool(axis_ratio["x"] > Z_RATIO_OK and axis_ratio["y"] > Z_RATIO_OK),
            "search_bound_hit": at_bound,
            "fit_ok": bool(not at_bound),
            "tilt_amplitude_deg": None if not np.isfinite(tilt) else round(tilt, 3),
            "level_assumption_ok": bool(np.isfinite(tilt) and tilt < TILT_OK_DEG),
        },
        "paper_check": paper_check(len(rings), elev[0], elev[-1]),
    }


def ground_z_estimate(x, y, z):
    """The ONE ground estimate: densest 5 cm z-bin among points 3..30 m from the ego origin."""
    rng = np.hypot(x, y)
    m = (rng > 3) & (rng < 30) & (z > -4) & (z < 4)
    if not m.any():
        return None
    hist, edges = np.histogram(z[m], bins=np.arange(-4, 4.05, 0.05))
    k = int(hist.argmax())
    return float((edges[k] + edges[k + 1]) / 2)


def assign_roles(lidars):
    """lidar_id -> role from the FITTED positions. -> (roles dict, top_lidar_id or None)."""
    roles, top = {}, None
    good = {k: v for k, v in lidars.items() if v["quality"]["fit_ok"]}
    for k in lidars:
        if k not in good:
            roles[k] = "unreliable_fit"          # apex hit the search box: position is meaningless
    big = {k: v for k, v in good.items() if v["n_rings"] >= BIG_RINGS}
    small = {k: v for k, v in good.items() if v["n_rings"] < BIG_RINGS}
    centre = [k for k, v in big.items() if abs(v["mount_xyz_ego_m"][1]) < 0.5]
    if centre:
        top = max(centre, key=lambda k: big[k]["mount_xyz_ego_m"][2])
        roles[top] = "top_center"
    for k, v in big.items():
        if k != top:
            roles[k] = "a_pillar_left" if v["mount_xyz_ego_m"][1] > 0 else "a_pillar_right"
    if len(small) >= 2:
        order = sorted(small, key=lambda k: small[k]["mount_xyz_ego_m"][0])
        roles[order[-1]] = "front_bumper"
        roles[order[0]] = "rear_bumper"
    else:
        for k in small:
            roles[k] = "bumper_unknown"
    return roles, top


def fit_rig(per_lidar, n_frames):
    """per_lidar: {lidar_id: (x, y, z, ring)} -> fit dict."""
    allx = np.concatenate([v[0] for v in per_lidar.values()])
    ally = np.concatenate([v[1] for v in per_lidar.values()])
    allz = np.concatenate([v[2] for v in per_lidar.values()])
    gz = ground_z_estimate(allx, ally, allz)
    out = {}
    for lid, (x, y, z, ring) in sorted(per_lidar.items()):
        r = fit_lidar(x, y, z, ring, n_frames, ground_z=gz)
        if r is None:
            print(f"[warn] lidar {lid}: not enough points to fit")
            continue
        if gz is not None:
            r["height_above_ground_m"] = round(r["mount_xyz_ego_m"][2] - gz, 3)
        out[str(int(lid))] = r
    roles, top = assign_roles(out)
    for k, role in roles.items():
        out[k]["role"] = role
        if role in ROLE_MODEL:
            r = out[k]
            r["paper_check"] = paper_check(r["n_rings"], r["elevation_min_deg"], r["elevation_max_deg"],
                                           only=ROLE_MODEL[role])
    return {"ground_z_m": gz, "num_frames": n_frames,
            "top_lidar_id": top, "roles": roles,
            "total_points_per_frame_mean": round(sum(v["points_per_frame_mean"] for v in out.values()), 1),
            "lidars": out}


# ----------------------------------------------------------------------------- data loading (navsim env)
def _vnum(name):
    m = re.search(r"veh-?(\d+)", name)
    return int(m.group(1)) if m else None


def collect_lidar_paths(logs_dir, blobs, vehicle, n_frames):
    """`n_frames` lidar files spread evenly over ALL logs of the vehicle (Bresenham quota per log)."""
    target = int(re.fullmatch(r"(?:veh-?)?0*(\d+)", vehicle.strip()).group(1))
    pkls = [p for p in sorted(Path(logs_dir).glob("*.pkl")) if _vnum(p.name) == target]
    if not pkls:
        sys.exit(f"no logs for vehicle {vehicle} in {logs_dir}")
    L, paths = len(pkls), []
    for k, p in enumerate(pkls):
        q = ((k + 1) * n_frames) // L - (k * n_frames) // L
        if q == 0:
            continue
        with open(p, "rb") as fh:
            frames = pickle.load(fh)
        cand = [fr["lidar_path"] for fr in frames if (blobs / fr["lidar_path"]).exists()]
        if not cand:
            continue
        idx = np.unique(np.linspace(0, len(cand) - 1, min(q, len(cand))).round().astype(int))
        paths += [cand[i] for i in idx]
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--vehicle", required=True, help='e.g. "veh-35"')
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default="lidar_fit.yaml")
    a = ap.parse_args()

    import yaml
    from navsim.common.dataclasses import Lidar  # navsim env

    blobs = Path(a.blobs).expanduser()
    paths = collect_lidar_paths(Path(a.logs).expanduser(), blobs, a.vehicle, a.frames)
    if not paths:
        sys.exit("no lidar files found on disk")
    acc = {}
    for rel in paths:
        pc = Lidar.from_paths(blobs, Path(rel), ["lidar_pc"]).lidar_pc
        x, y, z, _, ring, lid = pc[:6]
        for i in np.unique(lid):
            s = lid == i
            d = acc.setdefault(int(i), [[], [], [], []])
            for lst, arr in zip(d, (x[s], y[s], z[s], ring[s])):
                lst.append(arr)
    per = {k: tuple(np.concatenate(v) for v in vs) for k, vs in acc.items()}
    res = fit_rig(per, len(paths))
    res["vehicle"] = a.vehicle
    Path(a.out).write_text(yaml.safe_dump(res, sort_keys=False, default_flow_style=None, width=200))
    print(f"ground z = {res['ground_z_m']}   top lidar id = {res['top_lidar_id']}   "
          f"total pts/frame = {res['total_points_per_frame_mean']}")
    for k, r in res["lidars"].items():
        q = r["quality"]
        print(f"lidar {k} [{r.get('role', '?')}]: mount {r['mount_xyz_ego_m']}  rings {r['n_rings']}  "
              f"elev {r['elevation_min_deg']:.1f}..{r['elevation_max_deg']:.1f}  "
              f"az {r['azimuth']['hfov_deg']:.0f}deg@{r['azimuth']['center_deg_ego']}  "
              f"cone {q['cone_residual_m']} m  ratios x/y/z={q['x_profile_ratio']}/{q['y_profile_ratio']}/{q['z_profile_ratio']}  "
              f"fit_ok={q['fit_ok']}{' BOUND:' + ','.join(q['search_bound_hit']) if q['search_bound_hit'] else ''}  "
              f"tilt={q['tilt_amplitude_deg']}deg level_ok={q['level_assumption_ok']}  "
              f"paper_ok={r['paper_check'].get('consistent_with_paper')}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()