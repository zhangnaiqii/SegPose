import os
import re
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
from scipy.optimize import least_squares


PLY_TYPE_TO_DTYPE = {
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}

NUM_SUFFIX_RE = re.compile(r"(\d+)$", re.IGNORECASE)


@dataclass
class Trio:
    folder: str
    depth_path: str
    ply_path: str
    color_path: Optional[str] = None
    suffix: str = ""


def _read_depth_png(depth_path: str) -> np.ndarray:
    d = np.array(Image.open(depth_path), dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"Depth PNG must be single-channel 16-bit, got shape={d.shape}, path={depth_path}")
    return d


def _parse_ply_header(f) -> Tuple[str, int, List[Tuple[str, str]], int]:
    fmt = None
    vcount = None
    props: List[Tuple[str, str]] = []
    in_vertex = False

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF in PLY header")
        s = line.decode("utf-8", errors="ignore").strip()
        if s.startswith("format "):
            fmt = s.split()[1]
        elif s.startswith("element "):
            parts = s.split()
            elem = parts[1]
            count = int(parts[2])
            in_vertex = (elem == "vertex")
            if elem == "vertex":
                vcount = count
        elif s.startswith("property ") and in_vertex:
            parts = s.split()
            if parts[1] == "list":
                continue
            ptype, pname = parts[1], parts[2]
            props.append((ptype, pname))
        elif s == "end_header":
            header_end = f.tell()
            break

    if fmt is None or vcount is None:
        raise ValueError("PLY header missing format or vertex count")
    return fmt, vcount, props, header_end


def sample_ply_vertices_xyz(ply_path: str, n_samples: int, stride_factor: int = 8,
                            z_min: float = 1.0, z_max: float = 30000.0) -> np.ndarray:
    with open(ply_path, "rb") as f:
        fmt, vcount, props, header_end = _parse_ply_header(f)

        name_to_idx = {n: i for i, (_, n) in enumerate(props)}
        for k in ("x", "y", "z"):
            if k not in name_to_idx:
                raise ValueError(f"PLY missing vertex property '{k}', path={ply_path}")
        ix, iy, iz = name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]

        target = max(n_samples * stride_factor, 1)
        stride = max(1, vcount // target)

        if fmt == "ascii":
            f.seek(header_end)
            pts = []
            kept = 0
            for i in range(vcount):
                line = f.readline()
                if not line:
                    break
                if i % stride != 0:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x = float(parts[ix]); y = float(parts[iy]); z = float(parts[iz])
                except Exception:
                    continue
                if z <= z_min or z >= z_max:
                    continue
                pts.append((x, y, z))
                kept += 1
                if kept >= n_samples:
                    break
            if not pts:
                raise ValueError(f"No valid vertices sampled from {ply_path}")
            return np.asarray(pts, dtype=np.float64)

        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {fmt}, path={ply_path}")

        fields = []
        for ptype, pname in props:
            if ptype not in PLY_TYPE_TO_DTYPE:
                raise ValueError(f"Unsupported PLY property type '{ptype}' in {ply_path}")
            fields.append((pname, PLY_TYPE_TO_DTYPE[ptype]))
        dtype = np.dtype(fields)
        dtype = dtype.newbyteorder("<" if fmt == "binary_little_endian" else ">")

        f.seek(header_end)
        verts = np.fromfile(f, dtype=dtype, count=vcount)
        idx = np.arange(0, vcount, stride, dtype=np.int64)
        if idx.size > n_samples:
            idx = idx[:n_samples]
        x = verts["x"][idx].astype(np.float64)
        y = verts["y"][idx].astype(np.float64)
        z = verts["z"][idx].astype(np.float64)
        m = (z > z_min) & (z < z_max)
        xyz = np.stack([x[m], y[m], z[m]], axis=1)
        if xyz.shape[0] == 0:
            raise ValueError(f"No valid vertices sampled from {ply_path}")
        return xyz


def _sample_depth_bilinear_safe(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth.shape
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1
    valid = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)

    d = np.full(u.shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return d, valid

    idx = np.where(valid)[0]
    u0v = u0[idx]; v0v = v0[idx]; u1v = u1[idx]; v1v = v1[idx]
    du = (u[idx] - u0v); dv = (v[idx] - v0v)

    d00 = depth[v0v, u0v]
    d10 = depth[v0v, u1v]
    d01 = depth[v1v, u0v]
    d11 = depth[v1v, u1v]

    ok = (d00 > 0) & (d10 > 0) & (d01 > 0) & (d11 > 0)
    if not np.any(ok):
        return d, valid & False

    idx2 = idx[ok]
    du = du[ok]; dv = dv[ok]
    d00 = d00[ok]; d10 = d10[ok]; d01 = d01[ok]; d11 = d11[ok]

    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv

    d[idx2] = d00 * w00 + d10 * w10 + d01 * w01 + d11 * w11
    valid2 = np.zeros_like(valid)
    valid2[idx2] = True
    return d, valid2


def _residuals_fixed(p: np.ndarray, pts: np.ndarray, depth: np.ndarray, sx: float, sy: float, penalty: float) -> np.ndarray:
    fx, fy, cx, cy, s = p
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    u = sx * fx * (x / z) + cx
    v = sy * fy * (y / z) + cy
    d, valid = _sample_depth_bilinear_safe(depth, u, v)
    r = z - s * d
    r = np.where(valid, r, penalty)
    return np.clip(r, -5000.0, 5000.0).astype(np.float64)


def _build_inliers(p0: np.ndarray, pts: np.ndarray, depth: np.ndarray, sx: float, sy: float,
                   thresh: float = 80.0) -> np.ndarray:
    fx, fy, cx, cy, s = p0
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    u = sx * fx * (x / z) + cx
    v = sy * fy * (y / z) + cy
    d, valid = _sample_depth_bilinear_safe(depth, u, v)
    r = z - s * d
    m = valid & (np.abs(r) < thresh)
    in_pts = pts[m]
    if in_pts.shape[0] < 500:
        raise ValueError(f"Too few inliers ({in_pts.shape[0]}). Your depth/ply likely don't correspond.")
    return in_pts


def _coarse_choose_sign_and_init(depth: np.ndarray, pts: np.ndarray) -> Tuple[float, float, float, float]:
    H, W = depth.shape
    cx = W / 2.0
    cy = H / 2.0

    fx_grid = np.linspace(0.25 * W, 1.2 * W, 8)
    fy_grid = np.linspace(0.25 * H, 2.0 * H, 8)

    best = (-1, None)  # (inlier_count, (sy, fx, fy))
    sx = 1.0

    if pts.shape[0] > 20000:
        idx = np.random.choice(pts.shape[0], 20000, replace=False)
        pts0 = pts[idx]
    else:
        pts0 = pts

    for sy in (-1.0, 1.0):
        for fx in fx_grid:
            for fy in fy_grid:
                p0 = np.array([fx, fy, cx, cy, 1.0], dtype=np.float64)
                r = _residuals_fixed(p0, pts0, depth, sx=sx, sy=sy, penalty=2000.0)
                inl = np.sum((np.abs(r) < 50.0) & (np.abs(r - 2000.0) > 1e-9))
                if inl > best[0]:
                    best = (int(inl), (sy, float(fx), float(fy)))

    if best[1] is None:
        raise ValueError("Coarse init failed. Depth and PLY may not correspond.")
    sy, fx0, fy0 = best[1]
    return sx, sy, fx0, fy0


def estimate_intrinsics_from_one(depth_path: str, ply_path: str,
                                 n_ply_samples: int = 80000,
                                 inlier_thresh_mm: float = 80.0) -> Dict:
    depth = _read_depth_png(depth_path)
    H, W = depth.shape

    pts = sample_ply_vertices_xyz(ply_path, n_samples=n_ply_samples, stride_factor=8)
    if pts.shape[0] < 2000:
        raise ValueError(f"Too few sampled points ({pts.shape[0]}) from {ply_path}")

    sx, sy, fx0, fy0 = _coarse_choose_sign_and_init(depth, pts)
    cx0, cy0 = W / 2.0, H / 2.0

    p0 = np.array([fx0, fy0, cx0, cy0, 1.0], dtype=np.float64)
    in_pts = _build_inliers(p0, pts, depth, sx=sx, sy=sy, thresh=inlier_thresh_mm)

    lb = np.array([50.0, 50.0, 0.0, 0.0, 0.5], dtype=np.float64)
    ub = np.array([50000.0, 50000.0, float(W), float(H), 2.0], dtype=np.float64)

    def fun(p):
        return _residuals_fixed(p, in_pts, depth, sx=sx, sy=sy, penalty=200.0)

    res = least_squares(fun, p0, bounds=(lb, ub), loss="huber", f_scale=30.0, max_nfev=120)

    p = res.x
    r = _residuals_fixed(p, in_pts, depth, sx=sx, sy=sy, penalty=200.0)
    m = np.abs(r - 200.0) > 1e-9
    r_valid = r[m]

    return {
        "fx": float(p[0]),
        "fy": float(p[1]),
        "cx": float(p[2]),
        "cy": float(p[3]),
        "depth_scale": float(p[4]),
        "sx": float(sx),
        "sy": float(sy),
        "inlier_points": int(in_pts.shape[0]),
        "median_abs_res_mm": float(np.median(np.abs(r_valid))) if r_valid.size else float("nan"),
        "p95_abs_res_mm": float(np.percentile(np.abs(r_valid), 95)) if r_valid.size else float("nan"),
        "success": bool(res.success),
        "cost": float(res.cost),
    }


def _collect_trios(root: str) -> List[Trio]:
    if not os.path.isdir(root):
        raise FileNotFoundError(root)

    trios: List[Trio] = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue

        files = os.listdir(folder)
        depth_pngs = [f for f in files if f.lower().endswith(".png") and f.lower().startswith("depth")]
        color_pngs = [f for f in files if f.lower().endswith(".png") and f.lower().startswith("color")]
        plys = [f for f in files if f.lower().endswith(".ply") and ("depthpoints" in f.lower() or "rgbdpoints" in f.lower())]

        if not depth_pngs or not plys:
            continue

        def suffix_of(fn: str) -> str:
            stem = os.path.splitext(fn)[0]
            m = NUM_SUFFIX_RE.search(stem)
            return m.group(1) if m else stem

        depth_map: Dict[str, str] = {suffix_of(f): f for f in depth_pngs}
        ply_map: Dict[str, str] = {suffix_of(f): f for f in plys}
        color_map: Dict[str, str] = {suffix_of(f): f for f in color_pngs}

        common = sorted(set(depth_map.keys()) & set(ply_map.keys()))
        if common:
            for suf in common:
                trios.append(Trio(
                    folder=folder,
                    depth_path=os.path.join(folder, depth_map[suf]),
                    ply_path=os.path.join(folder, ply_map[suf]),
                    color_path=os.path.join(folder, color_map[suf]) if suf in color_map else None,
                    suffix=suf,
                ))
        else:
            if len(depth_pngs) == 1 and len(plys) == 1:
                trios.append(Trio(
                    folder=folder,
                    depth_path=os.path.join(folder, depth_pngs[0]),
                    ply_path=os.path.join(folder, plys[0]),
                    color_path=os.path.join(folder, color_pngs[0]) if len(color_pngs) == 1 else None,
                    suffix="",
                ))

    if not trios:
        raise ValueError(f"No valid (depth*.png + *DepthPoints*.ply) trios found under {root}")
    return trios


def _robust_median(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return med, mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"Z:\datasets\pointCloud\ori")
    ap.add_argument("--max_trios", default=0, type=int, help="0 means process all")
    ap.add_argument("--n_ply_samples", default=80000, type=int)
    ap.add_argument("--inlier_thresh_mm", default=80.0, type=float)
    ap.add_argument("--topk_refine", default=8, type=int, help="use top-K best trios for final joint refine; 0 disables")
    args = ap.parse_args()

    trios = _collect_trios(args.root)
    if args.max_trios > 0:
        trios = trios[:args.max_trios]

    results = []
    for i, t in enumerate(trios, 1):
        print(f"[{i}/{len(trios)}] {t.folder}  suffix={t.suffix}")
        r = estimate_intrinsics_from_one(
            depth_path=t.depth_path,
            ply_path=t.ply_path,
            n_ply_samples=args.n_ply_samples,
            inlier_thresh_mm=args.inlier_thresh_mm,
        )
        r["depth_path"] = t.depth_path
        r["ply_path"] = t.ply_path
        results.append(r)
        print(f"  fx={r['fx']:.3f}  fy={r['fy']:.3f}  cx={r['cx']:.3f}  cy={r['cy']:.3f}  scale={r['depth_scale']:.6f}  sy={r['sy']:+.0f}")
        print(f"  inliers={r['inlier_points']}  med|res|={r['median_abs_res_mm']:.3f}mm  p95|res|={r['p95_abs_res_mm']:.3f}mm  success={r['success']}")
        print()

    good = [r for r in results if r["success"] and math.isfinite(r["median_abs_res_mm"]) and r["median_abs_res_mm"] < 120.0]

    if len(good) == 0:
        raise RuntimeError("No good fits found. Depth/PLY pairs likely do not correspond, or thresholds are too strict.")

    if len(good) < 3:
        best = sorted(good, key=lambda r: (r["median_abs_res_mm"], -r["inlier_points"]))[0]
        print("=== Not enough frames for robust aggregation; using best single fit ===")
        print(f"fx={best['fx']:.6f}, fy={best['fy']:.6f}, cx={best['cx']:.6f}, cy={best['cy']:.6f}, depth_scale={best['depth_scale']:.9f}")
        print("K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]")
        return

    fx_med, fx_sig = _robust_median([r["fx"] for r in good])
    fy_med, fy_sig = _robust_median([r["fy"] for r in good])
    cx_med, cx_sig = _robust_median([r["cx"] for r in good])
    cy_med, cy_sig = _robust_median([r["cy"] for r in good])
    sc_med, sc_sig = _robust_median([r["depth_scale"] for r in good])

    print("=== Robust aggregate (median ± MAD*1.4826) ===")
    print(f"fx = {fx_med:.6f} ± {fx_sig:.6f}")
    print(f"fy = {fy_med:.6f} ± {fy_sig:.6f}")
    print(f"cx = {cx_med:.6f} ± {cx_sig:.6f}")
    print(f"cy = {cy_med:.6f} ± {cy_sig:.6f}")
    print(f"depth_scale = {sc_med:.9f} ± {sc_sig:.9f}")
    print()

    if args.topk_refine > 0:
        good_sorted = sorted(good, key=lambda r: (r["median_abs_res_mm"], -r["inlier_points"]))
        sel = good_sorted[:min(args.topk_refine, len(good_sorted))]

        frames = []
        for r in sel:
            depth = _read_depth_png(r["depth_path"])
            pts = sample_ply_vertices_xyz(r["ply_path"], n_samples=args.n_ply_samples, stride_factor=8)
            p_init = np.array([fx_med, fy_med, cx_med, cy_med, sc_med], dtype=np.float64)
            in_pts = _build_inliers(p_init, pts, depth, sx=1.0, sy=r["sy"], thresh=args.inlier_thresh_mm)
            if in_pts.shape[0] > 15000:
                idx = np.random.choice(in_pts.shape[0], 15000, replace=False)
                in_pts = in_pts[idx]
            frames.append((depth, in_pts, float(r["sy"])))

        H0, W0 = frames[0][0].shape
        for d, _, _ in frames[1:]:
            if d.shape != (H0, W0):
                raise RuntimeError("Depth resolutions differ across frames; joint refine would be ill-posed. Split by resolution.")

        def fun_joint(p):
            rs = []
            for depth, pts, sy in frames:
                rs.append(_residuals_fixed(p, pts, depth, sx=1.0, sy=sy, penalty=200.0))
            return np.concatenate(rs, axis=0)

        lb = np.array([50.0, 50.0, 0.0, 0.0, 0.5], dtype=np.float64)
        ub = np.array([50000.0, 50000.0, float(W0), float(H0), 2.0], dtype=np.float64)
        p0 = np.array([fx_med, fy_med, cx_med, cy_med, sc_med], dtype=np.float64)

        res = least_squares(fun_joint, p0, bounds=(lb, ub), loss="huber", f_scale=30.0, max_nfev=160)
        p = res.x

        print("=== Final joint refine (top-K) ===")
        print(f"fx={p[0]:.6f}, fy={p[1]:.6f}, cx={p[2]:.6f}, cy={p[3]:.6f}, depth_scale={p[4]:.9f}")
        print("K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]")
    else:
        print("K = [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]  (use robust aggregate above)")


if __name__ == "__main__":
    main()
