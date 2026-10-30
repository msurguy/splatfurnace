# Splat Furnace

Gaussian-splat `.ply` → low-poly, textured glTF, entirely in the browser on WebGPU.

**Try it: https://msurguy.github.io/splatfurnace/** (needs a WebGPU browser: recent Chrome, Edge or Safari).
Every push to `main` redeploys it (`.github/workflows/pages.yml`).

```bash
npm install
npm run dev      # http://localhost:5199 — splat PLYs in ~/Downloads appear as one-click samples
```

The UI (`index.html`, `src/ui.ts`, `src/style.css`) is Tailwind v4 + vanilla TS: Draft / Balanced / Detailed presets, technical
settings under *Advanced*, `⌘↵` to convert. `window.furnace` exposes the scripting hooks used below.

## CLI

The same pipeline runs in Node on [Dawn](https://github.com/dawn-gpu/node-webgpu) (the `webgpu`
package: prebuilt for macOS, Linux and Windows, x64 and arm64), with sharp (libwebp) in place of the browser's WebP encoder.

```bash
npm install -g splatfurnace   # Node 22.4+; or run once with `npx splatfurnace scene.ply`
splatfurnace scene.ply                                  # writes scene.glb next to it
splatfurnace scans/ -o out/ --preset detailed --unlit   # every .ply in a directory
splatfurnace "scans/**/*.ply" -o out/ --json -q > results.ndjson
```

Every *Advanced* setting has a flag (`--resolution`, `--triangles`, `--texture-size`, `--no-snap`, ...;
`splatfurnace --help` lists them), on top of `--preset draft|balanced|detailed`. Export flags: `--unlit`,
`--no-compress`, `--full-normal-map`, `--quality`, `--dense` (also writes `<name>-dense.glb`), `--flip`.
Existing outputs are skipped unless `--force`. Files convert one after another on one GPU device;
a failed file is reported and the batch continues (exit code 1). `--json` prints one line per file
(`input`, `output`, `bytes`, `settings`, `stats`, `error`).

CLI output matches the web app: backpack at defaults takes 5.4 s and is 1.69 MB lit / 0.90 MB unlit,
and the CLI's unlit GLB scores the same 27.33 dB against the splats in `furnace.compare()` as the web export.

From a checkout: `npm run build:cli` builds `dist-cli/index.js`, and `npm link` puts it on your PATH.
`npm publish` builds it first (`prepublishOnly`); the package holds only `dist-cli/`.

Code layout: the pipeline takes its platform services as arguments: `initGpu(gpu)`, `convert(..., { unwrap })`
(`xatlas/browser.ts` Web Worker vs `xatlas/node.ts` in-thread), `writeGlb({ encode })`. `src/cli/` holds
the Node side, typechecked by `tsconfig.cli.json` so browser code can't use Node APIs.

## Pipeline

| Stage | Where | What it does |
| --- | --- | --- |
| Load | `ply.ts` | Decodes 3DGS activations: sigmoid opacity, exp scale, normalised quaternion, `0.5 + C0·f_dc` colour. |
| Grid | `grid.ts` | Bounds from the 0.2–99.8 percentile of visible centres, so floaters don't eat resolution. |
| Surface | `gpu/surface.ts` | Splats → closed, manifold, outward-facing triangle mesh: GPU density scatter, exterior flood fill to seal cavities, marching cubes with Catmull-Rom edge crossings. See `docs/surface.md`. |
| Cleanup | `mesh.ts` | Drop small islands, normal-gated Taubin smoothing (no shrinkage, keeps creases). |
| Snap | `field.wgsl.ts` | Ray along −normal, composite splats front-to-back, move each vertex to the 50%-coverage depth. Displacements are smoothed before applying. |
| Colour | `field.wgsl.ts` | Same ray: alpha-composited splat colour, as a splat renderer would show the surface head-on. |
| Decimate | `bake.ts` `simplify` | meshoptimizer `simplifyWithUpdate` (moves vertices, updates normals/colours). |
| Atlas | `unwrap.ts`, `xatlas/` | xatlas (vendored in `third_party/xatlas` with speed fixes, built to wasm, run in a worker) segments the mesh into charts in spatial chunks of ~4k triangles and packs them to fill the texture, so vertices are shared; MikkTSpace-compatible tangents from meshoptimizer. Fallback (and with *Chart UV layout* off): `bake.ts` `buildAtlas`, one square cell per pair of similar-sized triangles. |
| Bake | `bake.ts` `bakeTextures` | Ray-sample the splats at every texel; density-gradient normals → tangent-space normal map; dilate gutters. |
| Export | `export.ts` | GLB via glTF-Transform: WebP textures, welded + quantized geometry, optional meshopt compression, `KHR_materials_unlit` without normals/tangents. See below. |

## Export

| Setting | Effect | Needs |
| --- | --- | --- |
| Always | WebP colour (q 0.9) and normal map (q 0.9), metadata stripped; geometry welded and quantized (positions 14-bit, normals/tangents 8-bit, UVs 16-bit); unlit exports drop normals and tangents | `EXT_texture_webp`, `KHR_mesh_quantization`: Blender 4.2+, three.js, Babylon.js |
| *Compress geometry* (default on) | `EXT_meshopt_compression` | Blender **5.2+**; three.js `GLTFLoader.setMeshoptDecoder(MeshoptDecoder)`; Babylon.js built in |
| *Half-resolution normal map* (default on) | lit exports store the normal map at half the texture size (box-filtered, renormalised): about half the lit file | — |
| *Chart UV layout* (default on) | xatlas charts instead of per-triangle cells: about 3× fewer vertices | adds about 1.1–1.4 s per conversion at 30k triangles |

Final GLB at defaults (30k triangles, 2048² textures):

| File | Before | Per-triangle cells | Charts | Charts + half-res normal map (default) |
| --- | --- | --- | --- | --- |
| idol, unlit | 5.88 MB | 1.73 MB | **0.73 MB** | — |
| idol, lit (normal map) | 18.05 MB | 4.67 MB | 3.10 MB | **1.46 MB** |
| backpack, unlit | — | 1.75 MB | **0.90 MB** | — |
| backpack, lit | — | 4.83 MB | 3.54 MB | **1.68 MB** |

For lit files `checkExport({ unlit: false })` reports `litPsnr`: lit renders of the file against
lit renders of the in-app mesh. Idol: 43.4 dB with the full-size normal map, 39.5 dB at half
size (about 35 dB with no normal map at all).

`furnace.checkExport()` exports, reloads the GLB with three's `GLTFLoader`, and runs `compare()` on it.
Round-trip PSNR stays within 0.05 dB of the in-memory mesh (options: `{ unlit, compress, halfNormalMap, quality }`). `scripts/validate-glb.mjs` (Khronos
validator) and `scripts/blender-import-check.py` (headless Blender import) check exported files.

## Measured (Apple GPU, defaults: 320 res, 30k tris, 2048² texture)

`furnace.compare()` renders the splats (via Spark) and the mesh from 8 orbit cameras and reports PSNR over covered pixels.

| File | Time | Final PSNR | Dense PSNR | Without snapping |
| --- | --- | --- | --- | --- |
| backpack (262k splats) | ~5.3 s | 27.5 dB | 27.9 dB | 25.3 dB |
| idol (262k splats) | ~4.5 s | 24.9 dB | 24.8 dB | 20.3 dB |

Surface stage design notes and checks: `docs/surface.md`.

## xatlas

`third_party/xatlas` is upstream xatlas (MIT, commit `f700c77`, July 2022) with the patches in
`third_party/xatlas/README.md`; they make charting about 3.5× faster with bit-identical output.
`third_party/xatlas-wasm/binding.cpp` is a small C interface (32-bit indices, charts computed
once and packed repeatedly). `npm run build:xatlas` rebuilds `src/xatlas/xatlas-wasm.{mjs,wasm}`
with Emscripten (`brew install emscripten`); the built files are checked in.

## License

MIT, see `LICENSE`. Vendored xatlas (`third_party/xatlas`) is MIT, (c) Jonathan Young.
