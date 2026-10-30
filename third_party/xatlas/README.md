# xatlas (vendored)

Upstream: https://github.com/jpcy/xatlas, commit `f700c7790aaa030e794b52ba7791a05c085faf0c`
(2022-07-26), files `source/xatlas/xatlas.{cpp,h}` and `LICENSE` (MIT). `xatlas.h` is unchanged.

`xatlas.cpp` carries the speed patches in `splat2mesh.patch`, each marked `splat2mesh` in the source.
None of them changes the result: on the idol mesh (30k triangles) the output vertices, UVs and
indices hash identically to upstream (native build, `-DXA_MULTITHREADED=0`), while chart computation
drops from 3.18 s to 0.92 s.

| Patch | Upstream cost | Change |
| --- | --- | --- |
| `ClusteredCharts::mergeCharts` | After each merge the loop restarts; for every chart it zeroed three chart-count-sized arrays and scanned every other chart. Failed merges (which re-parameterise and run a boundary-intersection test) were retried on every restart. | Visit only neighbouring charts (the only ones that can pass the "must share a boundary" test), in the same descending order; reset only the touched entries; skip a merge that already failed until either chart has changed. Deleted charts are removed with one remap pass instead of one face pass per chart. 1.67 s → 0.08 s. |
| `ClusteredCharts::createChart` | Seed search scanned every face for each new chart. | Faces pre-sorted in the same pick order (largest planar region, then lowest index), read from a cursor reset with the charts. Also fixes upstream picking face 0 (possibly already in a chart) when only zero-area faces remain. |
| `UniformGrid2::createGrid` | Boundary grid sized up to 512 × 512 cells and rebuilt for every validity test (each face added to a chart), so time went into clearing empty cells. | Cell count capped at `XA_GRID_CELLS_PER_EDGE` (2) cells per edge. The grid only narrows candidate pairs; the intersection result is unchanged. |
| `ClusteredCharts::growCharts` | Each added face scanned all charts for the best candidate. | Charts in a heap keyed by (best cost, chart index), the same pick order; stale keys are re-checked when popped. |

The WebAssembly build is about 1.8× slower than native. `src/xatlas/xatlas.ts` additionally splits
meshes into spatial chunks (charting still grows faster than linearly), which brings the idol to
~0.8 s charting + ~0.5 s packing in the browser.

To update upstream: copy the new files over, `patch -p1 < splat2mesh.patch` (resolve by hand if
needed), then `npm run build:xatlas`.
