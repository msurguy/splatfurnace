// Splat Furnace CLI: the browser pipeline (src/pipeline.ts, src/export.ts) on Dawn's WebGPU in Node.
// Progress goes to stderr; --json prints one result line per file to stdout.
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { parseSplatPly } from '../ply';
import { convert, DEFAULTS, type Settings } from '../pipeline';
import { PRESETS, type Preset } from '../presets';
import { toExportInput, writeGlb, type ExportOptions } from '../export';
import { runInThread } from '../xatlas/node';
import { initNodeGpu } from './gpu';
import { encodeWebpNode } from './webp';
import type { Gpu } from '../gpu/gpu';

declare const __VERSION__: string;

const HELP = `Splat Furnace ${__VERSION__}: Gaussian-splat .ply to low-poly, textured glTF (WebGPU via Dawn)

Usage: splatfurnace <input...> [options]

Inputs are .ply files, directories (their .ply files) or quoted globs ("scans/**/*.ply").

Output
  -o, --out <path>          output .glb (one input) or directory (several); default: next to each input
  -f, --force               overwrite existing outputs (otherwise they are skipped)
      --dense               also write <name>-dense.glb (the full-resolution, vertex-coloured mesh)
      --json                one JSON line per file on stdout: { input, output, bytes, stats, error? }
  -q, --quiet               no progress on stderr

Quality
  -p, --preset <name>       draft | balanced | detailed (default: balanced)
      --resolution <n>      voxel grid resolution along the longest axis
      --triangles <n>       target triangle count
      --texture-size <n>    texture size in texels (power of two)
      --iso <x>             density level of the surface
      --min-sigma <x>       smallest splat radius, in voxels
      --opacity-threshold <x>  ignore splats below this opacity (default 0.004)
      --island-fraction <x> drop pieces smaller than this fraction of the largest (0..1)
      --smoothing <n>       Taubin smoothing iterations
      --max-snap <x>        furthest a vertex may snap, in voxels
      --no-fill-cavities    keep internal cavities
      --no-snap             skip snapping the surface to the splats
      --no-normal-map       skip baking a normal map
      --no-chart-atlas      one atlas cell per triangle pair instead of xatlas charts
      --flip                rotate 180 degrees about X (Y-down / OpenCV captures)

Export
      --unlit               KHR_materials_unlit; no normals, tangents or normal map
      --no-compress         skip EXT_meshopt_compression (opens in Blender 4.2+)
      --full-normal-map     keep the normal map at full texture size
      --quality <x>         WebP quality for both textures, 0..1 (default 0.9)

  -h, --help                show this help
  -v, --version             print the version
`;

const options = {
  out: { type: 'string', short: 'o' },
  force: { type: 'boolean', short: 'f' },
  dense: { type: 'boolean' },
  json: { type: 'boolean' },
  quiet: { type: 'boolean', short: 'q' },
  preset: { type: 'string', short: 'p' },
  resolution: { type: 'string' },
  triangles: { type: 'string' },
  'texture-size': { type: 'string' },
  iso: { type: 'string' },
  'min-sigma': { type: 'string' },
  'opacity-threshold': { type: 'string' },
  'island-fraction': { type: 'string' },
  smoothing: { type: 'string' },
  'max-snap': { type: 'string' },
  'fill-cavities': { type: 'boolean' },
  snap: { type: 'boolean' },
  'normal-map': { type: 'boolean' },
  'chart-atlas': { type: 'boolean' },
  flip: { type: 'boolean' },
  unlit: { type: 'boolean' },
  compress: { type: 'boolean' },
  'full-normal-map': { type: 'boolean' },
  quality: { type: 'string' },
  help: { type: 'boolean', short: 'h' },
  version: { type: 'boolean', short: 'v' },
} as const;

type Values = ReturnType<typeof parseArgs<{ options: typeof options; allowPositionals: true; allowNegative: true }>>['values'];

class UsageError extends Error {}

// Numeric flags: setting, whether it must be an integer, and its allowed range.
const NUMBERS: Record<string, [keyof Settings, boolean, number, number]> = {
  resolution: ['resolution', true, 16, 2048],
  triangles: ['targetTriangles', true, 100, 10_000_000],
  'texture-size': ['textureSize', true, 64, 16384],
  iso: ['iso', false, 0, 1],
  'min-sigma': ['minSigma', false, 0, 16],
  'opacity-threshold': ['opacityThreshold', false, 0, 1],
  'island-fraction': ['islandFraction', false, 0, 1],
  smoothing: ['smoothing', true, 0, 100],
  'max-snap': ['maxSnap', false, 0, 64],
};
const TOGGLES: Record<string, keyof Settings> = {
  'fill-cavities': 'fillCavities', snap: 'snap', 'normal-map': 'normalMap', 'chart-atlas': 'chartAtlas',
};

function number(flag: string, raw: string, integer: boolean, min: number, max: number) {
  const n = Number(raw);
  if (!Number.isFinite(n) || (integer && !Number.isInteger(n)) || n < min || n > max) {
    throw new UsageError(`--${flag} expects ${integer ? 'an integer' : 'a number'} from ${min} to ${max}, got "${raw}"`);
  }
  return n;
}

function readSettings(v: Values): Settings {
  const preset = v.preset ?? 'balanced';
  if (!(preset in PRESETS)) throw new UsageError(`--preset must be one of ${Object.keys(PRESETS).join(', ')}, got "${preset}"`);
  const s = { ...PRESETS[preset as Preset] } as Record<keyof Settings, number | boolean>;
  for (const [flag, [key, integer, min, max]] of Object.entries(NUMBERS)) {
    const raw = v[flag as keyof Values];
    if (typeof raw === 'string') s[key] = number(flag, raw, integer, min, max);
  }
  for (const [flag, key] of Object.entries(TOGGLES)) {
    const on = v[flag as keyof Values];
    if (typeof on === 'boolean') s[key] = on;
  }
  const settings = s as unknown as Settings;
  if (settings.textureSize & (settings.textureSize - 1)) throw new UsageError(`--texture-size must be a power of two, got ${settings.textureSize}`);
  return settings;
}

function readExport(v: Values): ExportOptions {
  const q = v.quality === undefined ? undefined : number('quality', v.quality, false, 0, 1);
  return {
    unlit: Boolean(v.unlit),
    compress: v.compress ?? true,
    halfNormalMap: !v['full-normal-map'],
    quality: q === undefined ? undefined : { color: q, normal: q },
  };
}

const GLOB = /[*?[\]{}]/;

async function expandInputs(args: string[]): Promise<string[]> {
  const out: string[] = [];
  for (const arg of args) {
    const stat = await fs.stat(arg).catch(() => null);
    if (stat?.isDirectory()) {
      const names = (await fs.readdir(arg)).filter((f) => f.toLowerCase().endsWith('.ply')).sort();
      out.push(...names.map((f) => path.join(arg, f)));
    } else if (stat) {
      out.push(arg);
    } else if (GLOB.test(arg)) {
      const matches: string[] = [];
      for await (const f of fs.glob(arg)) if (f.toLowerCase().endsWith('.ply')) matches.push(f);
      if (!matches.length) throw new UsageError(`No .ply files match ${arg}`);
      out.push(...matches.sort());
    } else {
      throw new UsageError(`No such file or directory: ${arg}`);
    }
  }
  return [...new Set(out.map((f) => path.resolve(f)))];
}

/** Output path for `input`: --out as a file (single input) or directory, else next to the input. */
async function outputFor(input: string, out: string | undefined, single: boolean): Promise<string> {
  const name = `${path.basename(input).replace(/\.ply$/i, '')}.glb`;
  if (!out) return path.join(path.dirname(input), name);
  const isDir = await fs.stat(out).then((s) => s.isDirectory(), () => false);
  if (single && !isDir && out.toLowerCase().endsWith('.glb')) return path.resolve(out);
  return path.resolve(out, name);
}

const exists = (f: string) => fs.access(f).then(() => true, () => false);
const shown = (f: string) => { const r = path.relative(process.cwd(), f); return r.startsWith('..') ? f : r; };

async function main(argv: string[]): Promise<number> {
  const { values: v, positionals } = parseArgs({ args: argv, options, allowPositionals: true, allowNegative: true });
  if (v.help) { process.stdout.write(HELP); return 0; }
  if (v.version) { process.stdout.write(`${__VERSION__}\n`); return 0; }
  if (!positionals.length) throw new UsageError('No input files. Run splatfurnace --help for usage.');

  const settings = readSettings(v);
  const exportOptions = readExport(v);
  const inputs = await expandInputs(positionals);
  if (!inputs.length) throw new UsageError('No .ply files found.');
  const single = inputs.length === 1;
  if (!single && v.out?.toLowerCase().endsWith('.glb')) throw new UsageError('--out must be a directory when converting several files.');

  const progress = (msg: string) => { if (!v.quiet) process.stderr.write(`${msg}\n`); };
  let gpu: Gpu | null = null;
  let failed = 0;

  for (const [i, input] of inputs.entries()) {
    const label = inputs.length > 1 ? `[${i + 1}/${inputs.length}] ` : '';
    const name = path.basename(input);
    const output = await outputFor(input, v.out, single);
    const denseOutput = output.replace(/\.glb$/i, '-dense.glb');
    const report: Record<string, unknown> = { input, output };
    try {
      if (!v.force && (await exists(output))) {
        progress(`${label}${name}: ${shown(output)} exists, skipping (--force to overwrite)`);
        report.skipped = true;
        if (v.json) process.stdout.write(`${JSON.stringify(report)}\n`);
        continue;
      }
      progress(`${label}${name}`);
      const bytes = await fs.readFile(input);
      const splats = parseSplatPly(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
      progress(`  ${splats.count.toLocaleString()} splats`);
      gpu ??= await initNodeGpu();
      const result = await convert(gpu, splats, settings, (m) => progress(`  ${m}`), { unwrap: runInThread });

      const write = async (which: 'final' | 'dense', file: string) => {
        const glb = await writeGlb({ ...toExportInput(result, which, { ...exportOptions, flip: Boolean(v.flip) }), encode: encodeWebpNode });
        await fs.mkdir(path.dirname(file), { recursive: true });
        await fs.writeFile(file, glb);
        return glb.byteLength;
      };
      report.bytes = await write('final', output);
      if (v.dense) {
        report.denseOutput = denseOutput;
        report.denseBytes = await write('dense', denseOutput);
      }
      report.settings = settings;
      report.stats = result.stats;
      const st = result.stats;
      progress(`  Wrote ${shown(output)}: ${Number(st.finalTriangles).toLocaleString()} triangles, ${((report.bytes as number) / 1e6).toFixed(2)} MB in ${(Number(st.ms_total) / 1000).toFixed(1)} s`);
    } catch (err) {
      failed++;
      report.error = err instanceof Error ? err.message : String(err);
      process.stderr.write(`${label}${name}: error: ${report.error}\n`);
    }
    if (v.json) process.stdout.write(`${JSON.stringify(report)}\n`);
  }
  if (failed && inputs.length > 1) process.stderr.write(`${failed} of ${inputs.length} files failed\n`);
  return failed ? 1 : 0;
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code), // Dawn keeps the event loop alive, so exit explicitly
  (err) => {
    const usage = err instanceof UsageError || (err as { code?: string }).code?.startsWith('ERR_PARSE_ARGS');
    process.stderr.write(`splatfurnace: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(usage ? 2 : 1);
  },
);
