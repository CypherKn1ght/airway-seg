# Airway segmentation for cleared whole mouse lungs

Segments the airway tree in light-sheet images of optically cleared whole mouse lungs, without
machine learning and without an airway label. Airways are found in tissue autofluorescence from
local image shape (multiscale Hessian analysis), combined into an airway probability, thresholded,
repaired across short gaps and converted into a branch tree with per-branch measurements.

The full method, with equations and parameters, is described in the accompanying methods text.

## What it does

1. **Load** one channel of an Imaris (`.ims`) file at pyramid level 2 and resample to cubic voxels
   (about 17 to 20 µm for a whole lung).
2. **Lung mask** by Otsu thresholding of the smoothed image.
3. **Airway probability P** per voxel from four cues: dark tube shape (Frangi-type lumen detector),
   bright sheet shape (airway wall), smooth darkness, and distance from walls (enclosure).
4. **Binary mask** by hysteresis thresholding of P (default 0.74 / 0.68, the same for every lung).
5. **Gap repair**: pieces of the mask are reconnected along the cheapest path through weak airway
   signal (P ≥ 0.55, gaps up to 120 µm).
6. **Cleanup**: small pieces and compact non-tubular pieces removed.
7. **Airway tree**: skeleton, branches split at junctions, spurs pruned, loops broken by a maximum
   spanning tree, branch diameter, length, Horsfield and Strahler order.

## Install

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Run

Two steps, one script each:

```
python airway_seg.py  --lungs <folder with .ims files> --out <output folder>
python airway_tree.py --masks <same output folder>      --out <tree output folder>
```

`airway_seg.py` options: `--ims` (one file instead of a folder), `--channel` (default 0),
`--t-low`, `--t-high`, `--no-bridge`, `--match` (only files whose name contains this text),
`--save-prob` (also save the probability map). All settings are listed at the top of each script.

A whole lung took about 6 minutes on our desktop workstation, most of it in the Hessian filters.

## Outputs

| File | From | Contents |
|---|---|---|
| `<lung>_mask.npz` | `airway_seg.py` | airway mask `mask`, isotropic voxel size `iso` (µm) |
| `<lung>_prob.npz` | `airway_seg.py --save-prob` | airway probability `airway_prob`, lung mask `prob_mask`, `iso` |
| `<lung>_branches.csv` | `airway_tree.py` | one row per branch: tree, parent, generation, Horsfield and Strahler order, length and diameter (µm), terminal, in_loop, centre position |
| `<lung>_branch_labels.npz` | `airway_tree.py` | branch-id volume `labels` on the mask grid |
| `tree_summary.csv` | `airway_tree.py` | per-lung counts (branches, pieces, links, loops, main-tree share) |

The `generation` column is reported but is not reliable at this resolution, because the largest
central airways are not segmented as one connected piece. Use diameter or Horsfield order as the
airway-size axis.

## Citation

If you use this code, please cite: *[paper reference to be added]*.

## License

MIT, see `LICENSE`.
