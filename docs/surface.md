# Surface stage: design notes

How `src/gpu/surface.ts`, `src/gpu/surface.wgsl.ts`, `src/gpu/surface-table.ts`, `src/surface-check.ts`
and `scripts/check-surface-table.ts` work, and where each idea comes from.

The field is `D(x) = Σ o_i · exp(-½ m_i(x)²)`, with `m_i` the Mahalanobis distance to splat `i`
(sigmas raised to at least `minSigma`, cut at `support` sigmas, splats wider than `maxExtent`
ignored). Solid means `D ≥ iso`. The output is closed, manifold and wound counter-clockwise seen
from outside, with normals pointing from solid to empty.

## Design decisions

| # | Decision | Source |
| --- | --- | --- |
| 1 | Field `D(x) = Σ o·exp(-½m²)` with sigmas clamped to `minSigma`, cut at `m ≤ support` and `extent ≤ maxExtent`. | Opacity-weighted Gaussian density as in Kerbl et al. 2023; SuGaR §3 uses the same density level sets. |
| 2 | Density is evaluated by **scatter**: one workgroup per splat. Each thread takes one (y, z) row of the splat's bounding box and solves the quadratic `m²(x) ≤ support²` exactly for the x span, so no samples are wasted outside the ellipsoid. | Own reasoning. |
| 3 | Bounding box half-extent along world axis j is `support·sqrt(Σ_a (axis_a[j]·σ'_a)²)`, the exact axis-aligned extent of the ellipsoid. | Own reasoning (standard linear algebra). |
| 4 | Accumulation in 16.16 fixed point with `atomicAdd` on `u32`, because WGSL has no float atomics. Resolution 1.5e-5 is well below `iso`. Overflow would need a summed opacity above 65 536 at one point, which does not happen in practice. Integer addition is order-independent, so the field is deterministic. | Own reasoning; WGSL spec (atomic built-ins are `i32`/`u32` only). |
| 5 | Grid points on the outer faces of the grid are always classified empty, so the surface closes even if stray splats reach the boundary. | Own reasoning, to guarantee a closed mesh. |
| 6 | Solid/empty and the exterior mask are stored at **1 bit per point** in 8³ bricks, with each 8-point x row in one byte. | Own reasoning. |
| 7 | Cavity sealing is a **6-connected flood fill** of the empty exterior from the grid boundary. One workgroup per brick iterates to a fixed point in workgroup memory, using row bit-shifts for the x direction and neighbouring rows for y/z. Results go to global memory with `atomicOr`; dispatches repeat until a change counter reads 0. Because the exterior set only grows, races between bricks are harmless. Everything not reached is solid. | Textbook flood fill / connected-component labelling; brick-local iteration is own reasoning. |
| 8 | Without cavity sealing the exterior mask is simply "all empty points", so one code path extracts both cases. | Own reasoning. |
| 9 | **Marching cubes** over grid cells, with vertices on sign-change grid edges. **Interpolant: tricubic Catmull-Rom** through the samples. Restricted to a grid line it is the 1-D Catmull-Rom cubic through the four samples on that line, so each vertex is the iso-crossing of that cubic between the edge's endpoints, found by 20 bisection steps (error < 1e-6 voxel). This puts every vertex on the level set of a continuous (C¹) interpolant of the sampled field. Where the samples disagree with the classification (sealed cavity, forced-empty grid face) the vertex goes at the edge midpoint. | Lorensen & Cline 1987 for marching cubes; Catmull-Rom (cardinal) spline is textbook interpolation; using it for the edge crossing is own reasoning, see #18. |
| 10 | The case table is **generated in code**, not copied. Per face: walk the 4 corners counter-clockwise from outside and pair each solid→empty crossing with the next empty→solid crossing. This orients segments with solid on the left, and on ambiguous faces it always joins the solid corners. Segments link into cycles on the cell surface, and each cycle is fan-triangulated, reversed so triangles are CCW seen from the empty side. The face rule depends only on that face's corners, so neighbouring cells agree and the mesh is watertight. | Own reasoning, following the face-ambiguity discussion in Nielson 2004 (Dual Marching Cubes). |
| 11 | The fan apex is chosen so that no fan diagonal joins two vertices on the same cell face. Such a diagonal is the only kind of edge two cells could both create, so this guarantees every edge has exactly two triangles. `scripts/check-surface-table.ts` checks closedness, manifoldness, orientation and sign of volume on random fields. | Own reasoning. |
| 12 | Edge crossing parameter clamped to [0.001, 0.999], so vertices never coincide with grid points and form zero-area triangles. The offset is ≤ 0.001 voxel, inside the 0.1-voxel accuracy target. | Own reasoning. |
| 13 | Normals = −∇D, from central differences at the two edge endpoints, linearly interpolated with the same t and normalised. This gives a smooth, non-voxel-aligned normal field. If the gradient vanishes, fall back to the edge direction from solid to empty. | Standard practice in isosurface rendering (gradient normals; Lorensen & Cline 1987 §4). |
| 14 | Unique shared vertices with **stream compaction**: each grid point owns its +x/+y/+z edges. A two-level exclusive prefix sum over per-point (vertex, triangle) counts gives output offsets: per-block reduce, one-workgroup scan of block sums, then recompute and local scan in the emit passes. | Harris, Sengupta, Owens, GPU Gems 3 ch. 39. |
| 15 | Each cell's corner configuration is computed once into a byte buffer, which is read by the count and emit passes. | Own reasoning. |
| 16 | Output buffers are allocated after reading back the exact totals. Every buffer is checked against `gpu.maxBinding` with an error naming **Resolution**. Every buffer is tracked and destroyed in a `finally` block. | Own reasoning. |
| 17 | `s2m.checkSurface()` sorts undirected-edge keys (with a direction flag) to count boundary, non-manifold and same-direction ("flipped") edges. Degenerate = triangle area ≤ 1e-6 voxel². | Own reasoning. |
| 18 | **Why not linear interpolation.** The first version interpolated D linearly along edges, so vertices sat exactly on the trilinear interpolant. With snapping and smoothing off, lit grey renders of the idol's flat base showed clear horizontal bands a few voxels apart. Their spacing shrank at resolution 448, so they come from the grid, not the data. D is a steep, non-linear sum of Gaussians across the surface, so the linear crossing error depends on the surface's sub-voxel offset. On a gently tilted plane that offset drifts slowly, and the error shows as periodic terraces. The Catmull-Rom crossing (#9) removed the banding in the Dense mesh view and left only a very faint residual on the raw, unsmoothed stage output. | Own reasoning and measurement. |

## Measured (Apple Silicon, Chrome, WebGPU)

Run in the console with `s2m.loadSample(…)`, `s2m.run()`, `s2m.compare('dense' | 'final')` and `s2m.checkSurface()`.

| Metric | Target backpack | backpack | Target idol | idol |
| --- | --- | --- | --- | --- |
| `dense.psnr` | ≥ 27.7 | **27.95** | ≥ 24.6 | **24.70** |
| `final.psnr` | ≥ 27.3 | **27.44** | ≥ 24.6 | **24.78** |
| `final.silhouetteIoU` | ≥ 0.972 | **0.9756** | ≥ 0.925 | **0.9307** |
| `st.components` | 1 | **1** | 1 | **1** |
| `st.rawTriangles` | 0.4M–1.5M | **905,328** | 0.25M–0.8M | **487,344** |
| `st.ms_surface` | ≤ 3000 | **100–306** | ≤ 1500 | **43–93** |
| `checkSurface()` boundary / non-manifold / flipped | 0 / 0 / 0 | **0 / 0 / 0** | 0 / 0 / 0 | **0 / 0 / 0** |
| `checkSurface().degenerate` | ≤ 0.1 % | **94 (0.010 %)** | ≤ 0.1 % | **68 (0.014 %)** |
| `fillCavities: false` rawTriangles | valid mesh | **1,774,232 (×1.96)**, 0/0/0 | valid mesh | **827,280 (×1.70)**, 0/0/0 |

`compare()` must be run with **Lit off** (the app default). With Lit on, it compares a shaded
mesh against unshaded splats and PSNR drops by about 8 dB, which is a measurement artefact, not a regression.

## References

- Kerbl, Kopanas, Leimkühler, Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023.
- Guédon, Lepetit, "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction", CVPR 2024.
- Lorensen, Cline, "Marching Cubes", SIGGRAPH 1987; Nielson, "Dual Marching Cubes", IEEE Vis 2004.
- Harris, Sengupta, Owens, "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3 ch. 39.
