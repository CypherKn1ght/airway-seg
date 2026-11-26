"""
airway_tree.py -- airway tree and branch measurements from the masks made by airway_seg.py.

    python airway_tree.py --masks <folder with *_mask.npz> --out <output folder>

Per lung:
    <lung>_branches.csv       one row per branch: tree, parent, generation, Horsfield and Strahler
                              order, length and diameter (um), terminal, in_loop, centre position
    <lung>_branch_labels.npz  branch-id volume (key: labels) on the mask grid
and tree_summary.csv for all lungs.

Steps: closing + hole fill, 3D skeleton, branches split at junctions, short terminal spurs pruned,
pieces linked across small gaps (open end pointing at another piece), loops broken by a maximum
spanning tree weighted by branch diameter, orders counted from the tips.
Generation (counted from the widest branch) is reported but is only meaningful when the large
central airways are segmented as one connected piece; Horsfield order and diameter do not
depend on that.
"""
import os
import csv
import glob
import time
import argparse
import numpy as np
import scipy.ndimage as ndi
import networkx as nx
from scipy.spatial import cKDTree
from skimage import morphology

# ================================================================== SETTINGS ==
CLOSE_ITERS  = 2         # closing before thinning
SPUR_FACTOR  = 1.0       # prune terminal branches shorter than this x their diameter
PRUNE_PASSES = 3
JOIN_ON      = True      # link pieces across small gaps (topology only, no voxels added)
JOIN_FACTOR  = 3.0       # reach = this x branch diameter ...
JOIN_MIN_UM  = 60.0      # ... at least this ...
JOIN_MAX_UM  = 600.0     # ... and at most this
JOIN_ANGLE   = 60.0      # max angle between the branch direction and the gap (degrees)
MIN_TREE_UM  = 1000.0    # pieces with less centreline than this are fragments (tree = 0)
# ==============================================================================

N26 = np.ones((3, 3, 3), bool)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def skeletonize3d(mask):
    try:
        return morphology.skeletonize(mask).astype(bool)          # scikit-image >= 0.23
    except Exception:
        return morphology.skeletonize_3d(mask).astype(bool)       # older scikit-image


def neighbour_count(skel):
    k = N26.astype(np.uint8).copy()
    k[1, 1, 1] = 0
    return ndi.convolve(skel.astype(np.uint8), k, mode="constant") * skel


def split_branches(skel):
    """Junction voxels (>= 3 neighbours) separate branches."""
    nb = neighbour_count(skel)
    junc = skel & (nb >= 3)
    junc_lab, n_junc = ndi.label(junc, structure=N26)
    br_lab, n_br = ndi.label(skel & ~junc, structure=N26)
    return br_lab, n_br, junc_lab, n_junc, nb


OFFSETS = [(dz, dy, dx) for dz in (0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
           if (dz, dy, dx) > (0, 0, 0)]                        # 13 forward neighbour steps


def branch_lengths(br_lab, n_br, iso):
    """Sum of centre-to-centre steps (1, sqrt2 or sqrt3 voxels) along each branch."""
    L = np.zeros(n_br + 1)
    Z, Y, X = br_lab.shape
    for dz, dy, dx in OFFSETS:
        a = br_lab[0:Z - dz, max(0, -dy):Y - max(0, dy), max(0, -dx):X - max(0, dx)]
        b = br_lab[dz:Z, max(0, dy):Y - max(0, -dy) if dy < 0 else Y, max(0, dx):X - max(0, -dx) if dx < 0 else X]
        same = (a > 0) & (a == b)
        L += np.bincount(a[same], minlength=n_br + 1) * np.sqrt(dz * dz + dy * dy + dx * dx)
    return L * iso


def branch_table(skel, edt_um, iso):
    br_lab, n_br, junc_lab, n_junc, nb = split_branches(skel)
    idx = np.arange(1, n_br + 1)
    n_vox = ndi.sum(np.ones_like(br_lab), br_lab, idx).astype(int)
    lengths = branch_lengths(br_lab, n_br, iso)
    radius = ndi.mean(edt_um, br_lab, idx)
    centres = ndi.center_of_mass(skel, br_lab, idx)
    grown = ndi.grey_dilation(junc_lab, footprint=N26)            # junction ids spread by 1 voxel
    touch = {b: set() for b in idx}
    zz, yy, xx = np.nonzero((br_lab > 0) & (grown > 0))
    for b, j in zip(br_lab[zz, yy, xx], grown[zz, yy, xx]):
        touch[int(b)].add(int(j))
    rows = []
    for k, b in enumerate(idx):
        rows.append(dict(branch_id=int(b), n_vox=int(n_vox[k]),
                         length_um=float(max(lengths[b], iso)),
                         diameter_um=float(2 * radius[k]),
                         junctions=touch[int(b)],
                         terminal=len(touch[int(b)]) <= 1,
                         centre_zyx_um=tuple(float(c * iso) for c in centres[k])))
    return rows, br_lab


def prune_spurs(skel, edt_um, iso):
    """Remove terminal branches shorter than SPUR_FACTOR x their diameter."""
    for p in range(PRUNE_PASSES):
        rows, br_lab = branch_table(skel, edt_um, iso)
        spurs = [r["branch_id"] for r in rows
                 if r["terminal"] and len(r["junctions"]) == 1
                 and r["length_um"] < SPUR_FACTOR * r["diameter_um"]]
        if not spurs:
            break
        skel = skel & ~np.isin(br_lab, spurs)
        skel = skeletonize3d(skel)
        log(f"  prune pass {p + 1}: removed {len(spurs)} spurs")
    return skel


def branch_voxels(br_lab):
    idx = np.argwhere(br_lab > 0)
    lab = br_lab[tuple(idx.T)]
    o = np.argsort(lab, kind="stable")
    idx, lab = idx[o], lab[o]
    cuts = np.flatnonzero(np.diff(lab)) + 1
    return {int(l[0]): v for l, v in zip(np.split(lab, cuts), np.split(idx, cuts))}


def open_end(vox, ref):
    """Voxel of the branch farthest from ref, plus the outward direction there."""
    d = np.linalg.norm(vox - ref, axis=1)
    e = vox[np.argmax(d)]
    near = vox[np.linalg.norm(vox - e, axis=1) <= 5]
    v = e - near.mean(0)
    n = np.linalg.norm(v)
    return e, (v / n if n > 0 else v)


def build(name, mask, iso):
    mask = ndi.binary_closing(mask, structure=ndi.generate_binary_structure(3, 1), iterations=CLOSE_ITERS)
    mask = ndi.binary_fill_holes(mask)
    edt_um = ndi.distance_transform_edt(mask) * iso
    skel = prune_spurs(skeletonize3d(mask), edt_um, iso)
    rows, br_lab = branch_table(skel, edt_um, iso)
    _, _, junc_lab, n_junc, _ = split_branches(skel)
    jc = np.array(ndi.center_of_mass(skel, junc_lab, range(1, n_junc + 1))) if n_junc else np.zeros((0, 3))
    vox = branch_voxels(br_lab)
    by = {r["branch_id"]: r for r in rows}

    G = nx.MultiGraph()
    pos, ends = {}, {}                       # vertex -> coord ; open-end vertex -> (branch, direction)
    for r in rows:
        b, v = r["branch_id"], vox[r["branch_id"]]
        js = sorted(r["junctions"])[:2]
        verts = [("j", j) for j in js]
        for j in js:
            pos[("j", j)] = jc[j - 1]
        if len(js) == 1:
            e, dvec = open_end(v, jc[js[0] - 1])
            verts.append(("e", b, 0)); pos[("e", b, 0)] = e; ends[("e", b, 0)] = (b, dvec)
        elif len(js) == 0:
            e1, d1 = open_end(v, v.mean(0))
            e2, d2 = open_end(v, e1)
            for k, (e, dv) in enumerate([(e1, d1), (e2, d2)]):
                verts.append(("e", b, k)); pos[("e", b, k)] = e; ends[("e", b, k)] = (b, dv)
        r["vertices"] = verts
        if verts[0] == verts[1]:
            r["self_loop"] = True
            continue
        G.add_edge(verts[0], verts[1], key=b, weight=r["diameter_um"], branch=b, virtual=False)

    # ---- join pieces ------------------------------------------------------------
    comp_of = {}
    for ci, c in enumerate(nx.connected_components(G)):
        for v in c:
            comp_of[v] = ci
    n_pieces_before = len(set(comp_of.values()))
    links = []
    if JOIN_ON and ends:
        all_b = np.concatenate([np.full(len(v), b) for b, v in vox.items()])
        all_v = np.concatenate(list(vox.values()))
        tree = cKDTree(all_v)
        b_comp = {r["branch_id"]: comp_of[r["vertices"][0]] for r in rows if not r.get("self_loop")}
        cand = []
        for ev, (b, dvec) in ends.items():
            if b not in b_comp:
                continue
            reach = min(max(JOIN_FACTOR * by[b]["diameter_um"], JOIN_MIN_UM), JOIN_MAX_UM) / iso
            p = pos[ev]
            for k in tree.query_ball_point(p, reach):
                tb = int(all_b[k])
                if tb not in b_comp or b_comp[tb] == b_comp[b]:
                    continue
                gap = all_v[k] - p
                gl = np.linalg.norm(gap)
                if gl == 0 or np.dot(gap / gl, dvec) < np.cos(np.radians(JOIN_ANGLE)):
                    continue
                tv = min(by[tb]["vertices"], key=lambda u: np.linalg.norm(pos[u] - all_v[k]))
                cand.append((gl, ev, tv, b))
        cand.sort(key=lambda c: c[0])
        uf = {c: c for c in set(b_comp.values())}

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]; x = uf[x]
            return x
        for gl, ev, tv, b in cand:
            a, t = find(comp_of[ev]), find(comp_of[tv])
            if a == t:
                continue
            uf[a] = t
            G.add_edge(ev, tv, key=f"v{len(links)}", weight=by[b]["diameter_um"], branch=None,
                       virtual=True, gap_um=gl * iso)
            links.append(gl * iso)

    # ---- loops: maximum spanning tree ---------------------------------------------
    T = nx.Graph()
    T.add_nodes_from(G.nodes)
    for u, v, k, dd in nx.maximum_spanning_edges(G, algorithm="kruskal", weight="weight", keys=True, data=True):
        T.add_edge(u, v, **dd)
    kept = {dd["branch"] for _, _, dd in T.edges(data=True) if dd["branch"] is not None}
    for r in rows:
        r["in_loop"] = r["branch_id"] not in kept
        r.update(tree=0, generation=None, horsfield=None, strahler=None, parent_id=None)

    # ---- root each tree, generations and orders ---------------------------------
    # main tree = the longest piece; every tree is rooted at its widest branch
    comps = [c for c in nx.connected_components(T)]

    def brs_of(c):
        return [d["branch"] for _, _, d in T.subgraph(c).edges(data=True) if d["branch"] is not None]
    comps.sort(key=lambda c: -sum(by[b]["length_um"] for b in brs_of(c)))
    trunk_c = comps[0]
    trees = [trunk_c] + [c for c in comps if c is not trunk_c]
    tid = 0
    for comp in trees:
        sub = T.subgraph(comp)
        brs = brs_of(comp)
        if not brs:
            continue
        length = sum(by[b]["length_um"] for b in brs)
        is_tree = length >= MIN_TREE_UM or comp is trunk_c
        tid += is_tree
        # root: the widest branch; start from its open end if it has one, otherwise from
        # the end farther from the tree's centre (the hilum side)
        wb = max(brs, key=lambda b: by[b]["diameter_um"])
        cand = [v for v in by[wb]["vertices"] if v in comp]
        centre = np.mean([pos[v] for v in comp], axis=0)
        open_c = [v for v in cand if v in ends and sub.degree(v) == 1]
        root = open_c[0] if open_c else max(cand, key=lambda v: np.linalg.norm(pos[v] - centre))
        parent_edge = {}
        order_v = [root]
        for u, v in nx.bfs_edges(sub, root):
            d = sub[u][v]
            pe = parent_edge.get(u)
            # +1 per real branch; a branch reached across a virtual link (a repaired cut)
            # continues the airway before the cut, so it keeps that generation
            step = 0 if d["virtual"] or (pe is not None and pe["virtual"]) else 1
            depth = (pe["gen"] if pe is not None else 1) + step if pe is not None else (0 if d["virtual"] else 1)
            rec = dict(gen=depth, virtual=d["virtual"], branch=d["branch"], parent=pe, kids=[])
            if pe is not None:
                pe["kids"].append(rec)
            parent_edge[v] = rec
            order_v.append(v)
            if rec["branch"] is not None:
                r = by[rec["branch"]]
                r["generation"] = rec["gen"]
                r["tree"] = tid if is_tree else 0
                p = pe
                while p is not None and p["branch"] is None:
                    p = p["parent"]
                r["parent_id"] = p["branch"] if p is not None else None
        # Horsfield / Strahler, leaves first (reverse BFS)
        for v in reversed(order_v[1:]):
            rec = parent_edge[v]
            kh = [k["hors"] for k in rec["kids"]]
            ks = [k["strah"] for k in rec["kids"]]
            if not kh:
                rec["hors"], rec["strah"] = 1, 1
            elif rec["virtual"] or len(rec["kids"]) == 1:
                rec["hors"], rec["strah"] = max(kh), max(ks)
            else:
                rec["hors"] = max(kh) + 1
                m = max(ks)
                rec["strah"] = m + 1 if ks.count(m) >= 2 else m
            if rec["branch"] is not None:
                by[rec["branch"]]["horsfield"] = rec["hors"]
                by[rec["branch"]]["strahler"] = rec["strah"]

    tot = sum(r["length_um"] for r in rows) + 1e-9
    main = [r for r in rows if r["tree"] == 1]
    summ = dict(lung=name, branches=len(rows), pieces_before_join=n_pieces_before, links_made=len(links),
                median_link_um=round(float(np.median(links)), 1) if links else 0,
                trees=tid, loop_branches=sum(r["in_loop"] for r in rows),
                main_tree_centreline_pct=round(100 * sum(r["length_um"] for r in main) / tot, 1),
                max_generation=max((r["generation"] or 0) for r in main) if main else 0,
                root_diameter_um=round(max((r["diameter_um"] for r in main if r["generation"] == 1), default=0)))
    return rows, br_lab, summ


def write_rows(path, name, rows, iso):
    cols = ["lung", "branch_id", "tree", "parent_id", "generation", "horsfield", "strahler",
            "length_um", "diameter_um", "terminal", "in_loop", "n_vox", "centre_z_um", "centre_y_um", "centre_x_um"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            z, y, x = r["centre_zyx_um"]
            w.writerow([name, r["branch_id"], r["tree"], r["parent_id"], r["generation"], r["horsfield"],
                        r["strahler"], round(r["length_um"], 1), round(r["diameter_um"], 1), r["terminal"],
                        r["in_loop"], r["n_vox"], round(z, 1), round(y, 1), round(x, 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True, help="folder with <lung>_mask.npz from airway_seg.py")
    ap.add_argument("--out", required=True, help="output folder")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.masks, "*_mask.npz")))
    if not files:
        raise SystemExit(f"no *_mask.npz in {a.masks}")
    os.makedirs(a.out, exist_ok=True)
    summaries = []
    for fp in files:
        name = os.path.basename(fp).replace("_mask.npz", "")
        log(name)
        d = np.load(fp)
        iso = float(d["iso"])
        rows, br_lab, summ = build(name, d["mask"].astype(bool), iso)
        write_rows(os.path.join(a.out, f"{name}_branches.csv"), name, rows, iso)
        np.savez_compressed(os.path.join(a.out, f"{name}_branch_labels.npz"), labels=br_lab, iso=iso)
        summaries.append(summ)
        log("  " + ", ".join(f"{k} {v}" for k, v in summ.items() if k != "lung"))
    with open(os.path.join(a.out, "tree_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)


if __name__ == "__main__":
    main()
