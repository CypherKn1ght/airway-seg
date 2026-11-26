"""
airway_seg.py -- airway segmentation for light-sheet images of cleared whole mouse lungs.

Reads one channel of an Imaris (.ims) file, computes an airway probability P for every voxel
from local image shape (multiscale Hessian analysis) and darkness, thresholds P, repairs short
gaps and cleans the result. No machine learning and no airway stain needed: the airways are
found in tissue autofluorescence.

    python airway_seg.py --lungs <folder with .ims files> --out <output folder>
    python airway_seg.py --ims <one file.ims> --out <output folder>

Writes <lung>_mask.npz (keys: mask, iso = voxel size in um) per lung, and with --save-prob
also <lung>_prob.npz (airway_prob, prob_mask, iso). The masks feed airway_tree.py.
"""
import os
import glob
import time
import argparse
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.graph import MCP_Geometric

# ================================================================== SETTINGS ==
LEVEL          = 2          # Imaris pyramid level to read (level 2 ~ 8 x 6.5 x 6.5 um)
MAX_VOX        = 60_000_000 # voxel budget for the isotropic volume (sets the voxel size)

# lung mask
SMOOTH_UM      = 50.0       # smoothing before Otsu
CLOSE_ITERS    = 3          # closing of the lung mask
RIM_ERODE_VOX  = 3          # erode the lung mask before applying it to P (bright surface band)

# intensity normalisation
BG_UM          = 300.0      # window for local mean / standard deviation

# Hessian detectors
MIN_SIGMA_VOX   = 0.95      # scales finer than this (in voxels) are skipped
SCALE_RATIO     = 1.35
PLATE_SIGMAS_UM = [3.25, 6.5, 13.0, 26.0]     # wall (bright sheets)
TUBE_MIN_UM, TUBE_MAX_UM = 6.5, 250.0         # lumen (dark tubes)
BLOB_MIN_UM, BLOB_MAX_UM = 10.0, 150.0        # bright compact structures

# smooth darkness
S_SMOOTH_UM    = 4.0
TEX_UM         = 6.0
TEX_W          = 1.0
DARK_SLOPE     = 1.5        # logistic slope for p_dark

# fusion
MAP_PCT        = 99.9       # wall / lumen / blob maps scaled by this in-lung percentile
ENC_BARRIER_PCT = 85        # wall voxels above this in-lung percentile block the enclosure field
ENC_ITERS      = 8
W_BLOB         = 0.7
W_ENC          = 0.8

# binary mask
T_HIGH, T_LOW  = 0.74, 0.68 # hysteresis thresholds on P (same for every lung)
MIN_OBJ_UM3    = 5.0e4      # smallest object / hole kept after thresholding

# gap repair
BRIDGE_ON      = True
BRIDGE_P       = 0.55       # lowest P a repair path may cross
BRIDGE_MAX_UM  = 120.0      # longest gap repaired
BRIDGE_WIDTH   = 1          # repair path widened by this many voxels

# cleanup
CLEAN_CLOSE_ITERS = 1
MIN_PIECE_UM3  = 3.0e5      # pieces smaller than this are removed (~ 60 um airway, 100 um long)
COMPACT_MIN_UM3 = 5.0e6     # compact-piece filter only judges pieces at least this big
COMPACT_ELONG  = 3.0        # longest bounding-box side / max thickness below this ...
COMPACT_FILL   = 0.25       # ... and filling more than this of the box = compact, removed
# ==============================================================================

N26 = np.ones((3, 3, 3), bool)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- Imaris I/O --
def _attr(group, key):
    """Imaris stores attributes as arrays of single characters."""
    v = group.attrs.get(key)
    if v is None:
        return None
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("ascii", "ignore").strip()
    a = np.asarray(v).ravel()
    if a.dtype.kind in "SU":
        return b"".join(np.char.encode(a.astype(str))).decode("ascii", "ignore").strip()
    return str(v).strip()


def _num(group, key):
    try:
        return float(_attr(group, key))
    except (TypeError, ValueError):
        return None


def read_ims(path, level, channel):
    """Whole pyramid level of one channel, as float32 (Z, Y, X), plus voxel size (Z, Y, X) in um."""
    import h5py
    with h5py.File(path, "r") as f:
        img = f["DataSetInfo/Image"]
        n = [int(round(_num(img, k))) for k in ("Z", "Y", "X")]
        ext = [abs(_num(img, f"ExtMax{i}") - _num(img, f"ExtMin{i}")) for i in (2, 1, 0)]   # Z, Y, X
        g = f[f"DataSet/ResolutionLevel {level}/TimePoint 0/Channel {channel}"]
        shape = [_num(g, k) for k in ("ImageSizeZ", "ImageSizeY", "ImageSizeX")]
        shape = [int(round(s)) for s in shape] if None not in shape else list(g["Data"].shape)
        data = g["Data"][:shape[0], :shape[1], :shape[2]]                   # stored arrays are padded
    spacing = np.round([e / s for e, s in zip(ext, shape)], 4)          # um, rounded as in Imaris readers
    log(f"read level {level} channel {channel}: {tuple(shape)} at {np.round(spacing, 3)} um")
    return np.ascontiguousarray(data, np.float32), spacing


# ------------------------------------------------------------ preprocessing --
def to_isotropic(img, sc):
    iso = max(sc.min(), (np.prod(np.array(img.shape) * sc) / MAX_VOX) ** (1.0 / 3.0))
    zoom = sc / iso
    pre = [max(0.0, (1.0 / zf - 1.0) / 2.0) for zf in zoom]          # anti-aliasing
    if any(p > 0 for p in pre):
        img = ndi.gaussian_filter(img, pre)
    out = ndi.zoom(img, zoom, order=1).astype(np.float32)
    log(f"isotropic {iso:.2f} um: {out.shape}")
    return out, iso


def lung_mask(img, iso):
    sm = ndi.gaussian_filter(img, SMOOTH_UM / iso)
    t = float(np.expm1(threshold_otsu(np.log1p(np.maximum(sm, 0))[::2, ::2, ::2])))
    m = sm > t
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    if CLOSE_ITERS:
        m = ndi.binary_closing(m, iterations=CLOSE_ITERS)
    return ndi.binary_fill_holes(m)


def normalise(img, iso):
    s = BG_UM / iso
    m = ndi.gaussian_filter(img, s)
    sd = np.sqrt(np.maximum(ndi.gaussian_filter(img * img, s) - m * m, 0.0))
    floor = max(float(np.percentile(sd, 25)), 1e-6)
    return ((img - m) / np.maximum(sd, floor)).astype(np.float32)


# ------------------------------------------------------------------ Hessian --
def _fit(a, shape):
    out = np.zeros(shape, a.dtype)
    sl = tuple(slice(0, min(s, t)) for s, t in zip(a.shape, shape))
    out[sl] = a[sl]
    for ax in range(3):
        n = a.shape[ax]
        if n < shape[ax]:
            dst, src = [slice(None)] * 3, [slice(None)] * 3
            dst[ax], src[ax] = slice(n, None), slice(n - 1, n)
            out[tuple(dst)] = out[tuple(src)]
    return out


def _up(a, shape):
    if a.shape == tuple(shape):
        return a
    zf = np.array(shape, float) / np.array(a.shape, float)
    return _fit(ndi.zoom(a, zf, order=1).astype(np.float32), shape)


def _sym3_eigvals(a, b, c, d, e, f):
    """Eigenvalues of the symmetric 3x3 [[a,d,e],[d,b,f],[e,f,c]] per voxel (closed form)."""
    a, b, c, d, e, f = (x.astype(np.float64) for x in (a, b, c, d, e, f))
    q = (a + b + c) / 3.0
    p1 = d * d + e * e + f * f
    aq, bq, cq = a - q, b - q, c - q
    p = np.sqrt((aq * aq + bq * bq + cq * cq + 2.0 * p1) / 6.0)
    ps = np.where(p > 0, p, 1.0)
    detB = (aq * (bq * cq - f * f) - d * (d * cq - f * e) + e * (d * f - bq * e)) / ps ** 3
    phi = np.arccos(np.clip(detB / 2.0, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    return np.stack([e1, 3.0 * q - e1 - e3, e3])


def _eigs(z, s_vox, chunk=4_000_000):
    """Scale-normalised Hessian eigenvalues sorted by magnitude; large scales on a coarse grid."""
    f = max(1, int(s_vox // 3))
    if f > 1:
        pre = f / 2.0
        zc = ndi.gaussian_filter(z, pre)[::f, ::f, ::f]
        s_c = np.sqrt(max(s_vox ** 2 - pre ** 2, 1.0)) / f
    else:
        zc, s_c = z, s_vox
    k = (s_vox / f) ** 2
    H = [(k * ndi.gaussian_filter(zc, s_c, order=o)).astype(np.float32).ravel()
         for o in ([2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1])]
    N = H[0].size
    lam = np.empty((3, N), np.float32)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        w = _sym3_eigvals(*(h[s:e] for h in H))
        lam[:, s:e] = np.take_along_axis(w, np.argsort(np.abs(w), axis=0), axis=0)
    return lam[0], lam[1], lam[2], zc.shape


def _scales(lo, hi):
    n = int(np.ceil(np.log(hi / lo) / np.log(SCALE_RATIO))) + 1
    return list(lo * SCALE_RATIO ** np.arange(n))


def _usable(sigmas_um, iso):
    return [(s, s / iso) for s in sigmas_um if s / iso >= MIN_SIGMA_VOX]


def detect_wall(z, iso):
    wall = np.zeros(z.shape, np.float32)
    for _, sv in _usable(PLATE_SIGMAS_UM, iso):
        l1, l2, l3, cs = _eigs(z, sv)
        S2 = l1 * l1 + l2 * l2 + l3 * l3
        c2 = 2 * (0.5 * float(np.percentile(np.sqrt(S2), 99.5))) ** 2 + 1e-12
        r = np.exp(-((l1 * l1 + l2 * l2) / (l3 * l3 + 1e-12)) / 0.5) * (1 - np.exp(-S2 / c2))
        r[l3 >= 0] = 0
        np.maximum(wall, _up(r.reshape(cs), z.shape), out=wall)
    log("wall detector done")
    return wall


def detect_lumen(z, iso):
    lumen, c2 = np.zeros(z.shape, np.float32), None
    for _, sv in _usable(_scales(TUBE_MIN_UM, TUBE_MAX_UM), iso):
        l1, l2, l3, cs = _eigs(z, sv)
        S2 = l1 * l1 + l2 * l2 + l3 * l3
        if c2 is None:                                     # one constant shared by all scales
            c2 = 2 * (0.5 * float(np.percentile(np.sqrt(S2), 99.5))) ** 2 + 1e-12
        Ra2 = (l2 * l2) / (l3 * l3 + 1e-12)
        Rb2 = (l1 * l1) / (np.abs(l2 * l3) + 1e-12)
        r = (1 - np.exp(-Ra2 / 0.5)) * np.exp(-Rb2 / 0.5) * (1 - np.exp(-S2 / c2))
        r[(l2 <= 0) | (l3 <= 0)] = 0
        np.maximum(lumen, _up(r.reshape(cs), z.shape), out=lumen)
    log("lumen detector done")
    return lumen


def detect_blobs(z, iso):
    blobs = np.zeros(z.shape, np.float32)
    for _, sv in _usable(_scales(BLOB_MIN_UM, BLOB_MAX_UM), iso):
        l1, l2, l3, cs = _eigs(z, sv)
        S2 = l1 * l1 + l2 * l2 + l3 * l3
        sph = (3.0 * np.abs(l1 * l2 * l3) ** (1 / 3)) / (np.abs(l1 + l2 + l3) + 1e-12)
        r = sph ** 2 * np.sqrt(S2)
        r[(l1 >= 0) | (l2 >= 0) | (l3 >= 0)] = 0
        np.maximum(blobs, _up(r.reshape(cs), z.shape), out=blobs)
    log("blob detector done")
    return blobs


def smooth_darkness(z, iso, lung):
    zs = ndi.gaussian_filter(z, S_SMOOTH_UM / iso)
    s = TEX_UM / iso
    m = ndi.gaussian_filter(z, s)
    tex = np.sqrt(np.maximum(ndi.gaussian_filter(z * z, s) - m * m, 0))
    med = float(np.median(tex[lung])) + 1e-6
    return (-zs - TEX_W * (tex / med - 1)).astype(np.float32)


def _scale(a, lung):
    ref = np.percentile(a[lung][::7], MAP_PCT) if lung.any() else a.max()
    return np.minimum(a / (ref + 1e-12), 1.0).astype(np.float32)


def enclosure(p_wall, lung):
    barrier = p_wall > np.percentile(p_wall[lung][::7], ENC_BARRIER_PCT)
    u_cur = barrier.astype(np.float32)
    for _ in range(ENC_ITERS):                             # 6-neighbour averaging, walls held at 1
        u = np.pad(u_cur, 1, mode="edge")
        sm = (u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]
              + u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]
              + u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]) / np.float32(7.0)
        del u
        sm[barrier] = 1.0
        u_cur = sm
    g = np.sqrt(sum(x * x for x in np.gradient(u_cur)))
    return ((1.0 / (1.0 + g)) * (1.0 - p_wall)).astype(np.float32)


def airway_probability(path, channel=0):
    """Returns P (airway probability), the eroded lung mask and the isotropic voxel size."""
    raw, sc = read_ims(path, LEVEL, channel)
    img, iso = to_isotropic(raw, sc)
    del raw
    lung = lung_mask(img, iso)
    z = normalise(img, iso)
    wall, lumen, blob = detect_wall(z, iso), detect_lumen(z, iso), detect_blobs(z, iso)
    S = smooth_darkness(z, iso, lung)
    del z, img
    p_lumen, p_blob, p_wall = _scale(lumen, lung), _scale(blob, lung), _scale(wall, lung)
    del lumen, blob, wall
    p_dark = (1.0 / (1.0 + np.exp(-DARK_SLOPE * S))).astype(np.float32)
    del S
    p_enc = enclosure(p_wall, lung)
    P = np.maximum(np.maximum(p_lumen, W_BLOB * p_blob), W_ENC * p_enc) * p_dark * (1.0 - p_wall)
    prob_mask = ndi.binary_erosion(lung, iterations=RIM_ERODE_VOX) if RIM_ERODE_VOX else lung
    P *= prob_mask
    log(f"P: 99th percentile in lung {np.percentile(P[prob_mask][::7], 99):.3f}")
    return P.astype(np.float32), prob_mask, iso


# ---------------------------------------------------------------- binary mask --
def threshold(P, iso, t_low=T_LOW, t_high=T_HIGH):
    min_vox = int(MIN_OBJ_UM3 / iso ** 3)
    m = apply_hysteresis_threshold(P, t_low, t_high)
    m = morphology.remove_small_objects(m, min_size=min_vox)
    return morphology.remove_small_holes(m, area_threshold=min_vox)


def bridge_gaps(m, P, iso):
    """From every piece except the largest, the cheapest path (cost 1/P) to any other piece
    through voxels with P >= BRIDGE_P; paths crossing <= BRIDGE_MAX_UM of tissue are added."""
    lab, n = ndi.label(m, structure=N26)
    if n < 2:
        return m
    size = np.bincount(lab.ravel()); size[0] = 0
    main = int(np.argmax(size))
    reach = int(np.ceil(BRIDGE_MAX_UM / iso))
    out, sh, done = m.copy(), np.array(m.shape), 0
    ball = ndi.generate_binary_structure(3, 1)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or i == main:
            continue
        box = tuple(slice(a, b) for a, b in zip(np.maximum([s.start - reach for s in sl], 0),
                                                np.minimum([s.stop + reach for s in sl], sh)))
        L, Pb = lab[box], P[box]
        start, target = L == i, (L > 0) & (L != i)
        if not target.any():
            continue
        cost = np.where((Pb >= BRIDGE_P) | (L > 0), 1.0 / np.maximum(Pb, 0.05), np.inf)
        cost[start | target] = 1e-3
        mcp = MCP_Geometric(cost)
        acc, _ = mcp.find_costs(np.argwhere(start), ends=np.argwhere(target), find_all_ends=False)
        tc = np.where(target, acc, np.inf)
        j = np.unravel_index(np.argmin(tc), tc.shape)
        if not np.isfinite(tc[j]):
            continue
        path = np.array(mcp.traceback(j))
        gap = path[~(L[tuple(path.T)] > 0)]
        if len(gap) == 0 or len(gap) * iso > BRIDGE_MAX_UM:
            continue
        pm = np.zeros(L.shape, bool); pm[tuple(gap.T)] = True
        if BRIDGE_WIDTH:
            pm = ndi.binary_dilation(pm, ball, iterations=BRIDGE_WIDTH) & (Pb >= BRIDGE_P)
            pm[tuple(gap.T)] = True
        out[box] |= pm
        done += 1
    log(f"gap repair: {done} gaps bridged")
    return out


def clean(m, iso):
    if CLEAN_CLOSE_ITERS:
        m = ndi.binary_closing(m, iterations=CLEAN_CLOSE_ITERS)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m, structure=N26)
    if n == 0:
        return m
    size = np.bincount(lab.ravel()); size[0] = 0
    main = int(np.argmax(size))
    v_um3 = size * iso ** 3
    keep = v_um3 >= MIN_PIECE_UM3
    keep[0] = False
    thick = 2 * ndi.maximum(ndi.distance_transform_edt(m), lab, np.arange(n + 1))
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or not keep[i] or i == main or v_um3[i] < COMPACT_MIN_UM3:
            continue
        sides = np.array([s.stop - s.start for s in sl], float)
        if sides.max() / max(thick[i], 1.0) < COMPACT_ELONG and size[i] / sides.prod() > COMPACT_FILL:
            keep[i] = False
    keep[main] = True
    out = keep[lab]
    log(f"cleanup: {n} pieces -> {int(keep.sum())}")
    return out


def segment(path, channel=0, t_low=T_LOW, t_high=T_HIGH, bridge=BRIDGE_ON):
    P, prob_mask, iso = airway_probability(path, channel)
    m = threshold(P, iso, t_low, t_high)
    if bridge:
        m = bridge_gaps(m, P, iso)
    return clean(m, iso), P, prob_mask, iso


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lungs", help="folder with .ims files")
    src.add_argument("--ims", help="a single .ims file")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--channel", type=int, default=0, help="channel to segment (default 0)")
    ap.add_argument("--t-low", type=float, default=T_LOW)
    ap.add_argument("--t-high", type=float, default=T_HIGH)
    ap.add_argument("--no-bridge", action="store_true", help="skip gap repair")
    ap.add_argument("--match", default="", help="only files whose name contains this text")
    ap.add_argument("--save-prob", action="store_true", help="also save the probability map")
    a = ap.parse_args()

    paths = [a.ims] if a.ims else sorted(glob.glob(os.path.join(a.lungs, f"*{a.match}*.ims")))
    if not paths:
        raise SystemExit("no .ims files found")
    os.makedirs(a.out, exist_ok=True)
    for k, path in enumerate(paths, 1):
        name = os.path.splitext(os.path.basename(path))[0].replace(".ome", "")
        log(f"[{k}/{len(paths)}] {name}")
        mask, P, prob_mask, iso = segment(path, a.channel, a.t_low, a.t_high, not a.no_bridge)
        np.savez_compressed(os.path.join(a.out, f"{name}_mask.npz"), mask=mask, iso=iso)
        if a.save_prob:
            np.savez_compressed(os.path.join(a.out, f"{name}_prob.npz"),
                                airway_prob=P, prob_mask=prob_mask, iso=iso)
        log(f"  airway = {100 * mask.sum() / max(prob_mask.sum(), 1):.2f}% of lung, saved")


if __name__ == "__main__":
    main()
