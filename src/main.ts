import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'meshoptimizer';
import { parseSplatPly, type Splats } from './ply';
import { initGpu, type Gpu } from './gpu/gpu';
import { convert, DEFAULTS, type Result, type Settings } from './pipeline';
import { Viewer, type ViewMode } from './viewer';
import type { ExportOptions } from './export';
import { lastRawSurface } from './gpu/surface';
import { checkSurface } from './surface-check';

import { bindValueInputs, icons, initTooltips, matchPreset, PRESETS, renderFields, renderStages, stageFromLog, syncFields, toast, type Preset } from './ui';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;
renderFields($('#essentials'), $('#advancedFields'));
icons();
initTooltips();
const form = $<HTMLFormElement>('#settings');
const logEl = $<HTMLPreElement>('#log');
const statusEl = $('#status');
const convertBtn = $<HTMLButtonElement>('#convert');
const dlFinal = $<HTMLButtonElement>('#dlFinal');
const dlMenu = $<HTMLButtonElement>('#dlMenu');
const dlDense = $<HTMLButtonElement>('#dlDense');
const viewer = new Viewer($('#viewport'));
document.body.classList.add('no-result');

let gpu: Gpu | null = null;
let splats: Splats | null = null;
let sourceName = '';
let result: Result | null = null;
let resultSettings: Settings | null = null;
let busy = false;

function log(msg: string, reset = false) {
  if (reset) logEl.textContent = '';
  logEl.textContent += `${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
  const line = msg.trim().split('\n')[0];
  if (line) statusEl.textContent = line;
  const stage = stageFromLog(msg);
  if (busy && stage >= 0) renderStages($('#stages'), $('#stageName'), stage, !readSettings().snap);
}

// ---------------------------------------------------------------- settings form
function readSettings(): Settings {
  const s = { ...DEFAULTS } as Record<string, unknown>;
  for (const el of Array.from(form.elements) as HTMLInputElement[]) {
    if (!el.name || (el.type === 'radio' && !el.checked)) continue;
    s[el.name] = el.type === 'checkbox' ? el.checked : Number(el.value);
  }
  return s as unknown as Settings;
}

function writeSettings(s: Settings) {
  for (const el of Array.from(form.elements) as HTMLInputElement[]) {
    if (!el.name || !(el.name in s)) continue;
    const v = (s as unknown as Record<string, number | boolean>)[el.name];
    if (el.type === 'checkbox') el.checked = Boolean(v);
    else if (el.type === 'radio') el.checked = el.value === String(v);
    else el.value = String(v);
  }
  updateOutputs();
}

function updateOutputs() {
  syncFields(form);
  const s = readSettings();
  const preset = matchPreset(s);
  for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#presets button'))) b.classList.toggle('active', b.dataset.preset === preset);
  const dirty = result !== null && resultSettings !== null && JSON.stringify(s) !== JSON.stringify(resultSettings);
  $('#dirty').classList.toggle('hidden', !dirty || busy);
}

form.addEventListener('input', updateOutputs);
form.addEventListener('change', updateOutputs);
form.addEventListener('submit', (e) => {
  e.preventDefault();
  void run().catch(() => {});
});
bindValueInputs(form, updateOutputs);
for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#presets button[data-preset]:not([disabled])'))) {
  b.onclick = () => writeSettings(PRESETS[b.dataset.preset as Preset]);
}
$('#reset').onclick = () => writeSettings(DEFAULTS);
// Info icons sit inside switch labels; a click on them should only show the tip.
form.addEventListener('click', (e) => { if ((e.target as Element).closest('.info')) e.preventDefault(); });
writeSettings(DEFAULTS);

// ---------------------------------------------------------------- loading
async function load(bytes: ArrayBuffer, name: string) {
  if (busy) return;
  busy = true;
  try {
    log(`Reading ${name}`, true);
    splats = parseSplatPly(bytes);
    sourceName = name.replace(/\.ply$/i, '');
    setResult(null);
    $('#sourceName').textContent = name;
    $('#sourceName').title = name;
    $('#sourceMeta').textContent = `${splats.count.toLocaleString()} splats · ${(bytes.byteLength / 1e6).toFixed(1)} MB`;
    document.body.classList.remove('no-file');
    $('#pick').textContent = 'Replace';
    await viewer.showSplats(bytes, name);
    frameSplats();
    setMode('splats');
    convertBtn.disabled = false;
    log(`Loaded ${splats.count.toLocaleString()} splats. Press Convert.`);
  } catch (err) {
    log(`Error: ${(err as Error).message}`);
    toast(`Couldn't read ${name}: ${(err as Error).message}`, 'error');
  } finally {
    busy = false;
  }
}

function frameSplats() {
  if (!splats) return;
  const { center, radius } = splatBounds(splats);
  viewer.frame(center, radius);
}

function splatBounds(s: Splats) {
  const axis = (a: number) => {
    const v = new Float32Array(s.count);
    for (let i = 0; i < s.count; i++) v[i] = s.centers[i * 3 + a];
    v.sort();
    return [v[Math.floor(s.count * 0.01)], v[Math.floor(s.count * 0.99)]];
  };
  const [x, y, z] = [axis(0), axis(1), axis(2)];
  const center = new THREE.Vector3((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2);
  const radius = 0.5 * Math.hypot(x[1] - x[0], y[1] - y[0], z[1] - z[0]);
  return { center, radius };
}

const fileInput = $<HTMLInputElement>('#file');
fileInput.addEventListener('change', async () => {
  const f = fileInput.files?.[0];
  fileInput.value = '';
  if (f) await load(await f.arrayBuffer(), f.name);
});
$('#drop').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });

// Drop anywhere: dragenter/leave fire per child element, so count them.
let dragDepth = 0;
const hasFiles = (e: DragEvent) => e.dataTransfer?.types.includes('Files');
document.addEventListener('dragenter', (e) => { if (hasFiles(e)) { dragDepth++; document.body.classList.add('dragging'); } });
document.addEventListener('dragleave', () => { if (--dragDepth <= 0) { dragDepth = 0; document.body.classList.remove('dragging'); } });
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const f = e.dataTransfer?.files[0];
  if (!f) return;
  if (!/\.ply$/i.test(f.name)) return toast('Drop a Gaussian splat .ply file', 'error');
  await load(await f.arrayBuffer(), f.name);
});

async function loadSample(name: string) {
  const res = await fetch(`/__samples/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Could not fetch ${name}`);
  await load(await res.arrayBuffer(), name);
}

if (import.meta.env.DEV) {
  fetch('/__samples').then((r) => r.json()).then((names: string[]) => {
    const box = $('#samples');
    for (const n of names) {
      const b = document.createElement('button');
      b.textContent = n.replace(/\.ply$/i, '');
      b.title = `~/Downloads/${n}`;
      b.onclick = () => void loadSample(n);
      box.appendChild(b);
    }
  }).catch(() => {});
}

// ---------------------------------------------------------------- convert
async function run(overrides: Partial<Settings> = {}) {
  if (!splats || busy) return null;
  busy = true;
  const settings = { ...readSettings(), ...overrides };
  convertBtn.classList.add('hidden');
  $('#progress').classList.remove('hidden');
  $('#dirty').classList.add('hidden');
  renderStages($('#stages'), $('#stageName'), -1, !settings.snap);
  const t0 = performance.now();
  const timer = setInterval(() => { $('#elapsed').textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`; }, 100);
  try {
    gpu ??= await initGpu();
    log(`Converting ${sourceName}`, true);
    result = await convert(gpu, splats, settings, log);
    viewer.setResult(result.dense, result.final, result.bake.color, result.bake.normal, settings.textureSize);
    setResult(result, settings);
    setMode('final');
    $('#exportStep').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    const st = result.stats;
    log(`\nDone in ${(Number(st.ms_total) / 1000).toFixed(1)} s`);
    log(Object.entries(st).map(([k, v]) => `  ${k}: ${typeof v === 'number' ? v.toLocaleString() : v}`).join('\n'));
    statusEl.textContent = `Done in ${(Number(st.ms_total) / 1000).toFixed(1)} s · ${Number(st.finalTriangles).toLocaleString()} triangles`;
    return st;
  } catch (err) {
    log(`Error: ${(err as Error).message}`);
    toast(/webgpu|adapter/i.test((err as Error).message) ? 'WebGPU is not available. Try a recent Chrome, Edge or Safari.' : `Conversion failed: ${(err as Error).message}`, 'error');
    console.error(err);
    throw err;
  } finally {
    clearInterval(timer);
    busy = false;
    convertBtn.classList.remove('hidden');
    $('#progress').classList.add('hidden');
    updateOutputs();
  }
}

function setResult(r: Result | null, settings: Settings | null = null) {
  result = r;
  resultSettings = settings;
  document.body.classList.toggle('no-result', !r);
  dlFinal.disabled = dlMenu.disabled = dlDense.disabled = !r;
  for (const m of ['dense', 'final']) $<HTMLButtonElement>(`#views [data-mode="${m}"]`).disabled = !r;
  $('#moreStats').classList.toggle('hidden', !r);
  const st = r?.stats ?? {};
  const tile = (value: string, label: string) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`;
  const k = (n: unknown) => (Number(n) >= 1000 ? `${+(Number(n) / 1000).toFixed(1)}k` : String(n ?? '–'));
  $('#stats').innerHTML = r && settings
    ? tile(k(st.finalTriangles), 'Triangles') + tile(`${settings.textureSize / 1024}K`, 'Texture') + tile(`${(Number(st.ms_total) / 1000).toFixed(1)}s`, 'Time')
    : tile('–', 'Triangles') + tile('–', 'Texture') + tile('–', 'Time');
  $('#allStats').innerHTML = Object.entries(st).map(([key, v]) => `<dt>${key}</dt><dd class="text-right text-ink">${typeof v === 'number' ? v.toLocaleString() : v}</dd>`).join('');
}
setResult(null);

document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !convertBtn.disabled) { e.preventDefault(); form.requestSubmit(); }
  if (e.key === 'Escape') closeExportMenu();
});

// ---------------------------------------------------------------- view + export
function setMode(mode: ViewMode) {
  viewer.setMode(mode);
  for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#views button'))) {
    b.classList.toggle('active', b.dataset.mode === mode);
  }
}
for (const b of Array.from(document.querySelectorAll<HTMLButtonElement>('#views button'))) {
  b.onclick = () => setMode(b.dataset.mode as ViewMode);
}
$<HTMLInputElement>('#wire').onchange = (e) => viewer.setWireframe((e.target as HTMLInputElement).checked);
$<HTMLInputElement>('#lit').onchange = (e) => viewer.setLit((e.target as HTMLInputElement).checked);
$<HTMLInputElement>('#flip').onchange = (e) => viewer.setFlip((e.target as HTMLInputElement).checked);
$('#frameBtn').onclick = frameSplats;

const logToggle = $('#logToggle');
logToggle.onclick = () => {
  const open = logEl.classList.toggle('hidden') === false;
  logToggle.setAttribute('aria-expanded', String(open));
  if (open) logEl.scrollTop = logEl.scrollHeight;
};

const exportOptions = $('#exportOptions');
function closeExportMenu() {
  exportOptions.classList.add('hidden');
  dlMenu.setAttribute('aria-expanded', 'false');
}
dlMenu.onclick = (e) => {
  e.stopPropagation();
  const open = exportOptions.classList.toggle('hidden') === false;
  dlMenu.setAttribute('aria-expanded', String(open));
};
document.addEventListener('click', (e) => { if (!exportOptions.contains(e.target as Node)) closeExportMenu(); });

async function download(which: 'final' | 'dense') {
  const blob = await viewer.exportGlb(which, {
    unlit: $<HTMLInputElement>('#unlit').checked,
    compress: $<HTMLInputElement>('#compress').checked,
    halfNormalMap: $<HTMLInputElement>('#halfNormal').checked,
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${sourceName}${which === 'dense' ? '-dense' : ''}.glb`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  toast(`Saved ${a.download} · ${(blob.size / 1e6).toFixed(2)} MB`);
}
const guarded = (which: 'final' | 'dense') => () => void download(which).catch((err) => toast(`Export failed: ${(err as Error).message}`, 'error'));
dlFinal.onclick = guarded('final');
dlDense.onclick = guarded('dense');

/**
 * Objective check: render splats and a mesh from the same cameras and compare.
 * PSNR is over pixels covered by either render; IoU compares silhouettes.
 */
async function compare(mode: 'final' | 'dense' = 'final', views = 8) {
  const bg = [0x15, 0x17, 0x1a];
  const isBg = (d: Uint8ClampedArray, i: number) => Math.abs(d[i] - bg[0]) + Math.abs(d[i + 1] - bg[1]) + Math.abs(d[i + 2] - bg[2]) < 6;
  const saved = viewer.camera.position.clone();
  let se = 0, n = 0, inter = 0, union = 0;
  const bias = [0, 0, 0];
  let both = 0;
  const perView: number[] = [];
  for (let v = 0; v < views; v++) {
    const az = (v / views) * Math.PI * 2;
    const el = v % 2 ? 0.45 : -0.15;
    const a = await viewer.snapshot('splats', az, el);
    const b = await viewer.snapshot(mode, az, el);
    let vse = 0, vn = 0;
    for (let i = 0; i < a.data.length; i += 4) {
      const ia = !isBg(a.data, i), ib = !isBg(b.data, i);
      if (ia && ib) {
        inter++;
        both++;
        for (let k = 0; k < 3; k++) bias[k] += b.data[i + k] - a.data[i + k];
      }
      if (ia || ib) {
        union++;
        for (let k = 0; k < 3; k++) { const d = a.data[i + k] - b.data[i + k]; vse += d * d; }
        vn += 3;
      }
    }
    se += vse; n += vn;
    perView.push(+(10 * Math.log10((255 * 255) / (vse / Math.max(1, vn)))).toFixed(2));
  }
  viewer.camera.position.copy(saved);
  viewer.camera.lookAt(viewer.controls.target);
  setMode(mode);
  return { psnr: +(10 * Math.log10((255 * 255) / (se / Math.max(1, n)))).toFixed(2), silhouetteIoU: +(inter / Math.max(1, union)).toFixed(4), meshMinusSplatRGB: bias.map((x) => +(x / Math.max(1, both)).toFixed(2)), perView };
}

/**
 * Export round trip: write the GLB, load it back the way a three.js app would, and run
 * compare() on it in place of the in-memory mesh. Use unlit exports for PSNR (compare is unlit).
 * Lit exports also get `litPsnr`: lit renders of the file against lit renders of the in-app mesh
 * (full-resolution normal map), which measures what texture compression and resizing cost.
 * Textures get the viewer's anisotropic filtering so the score reflects only what the file contains.
 */
async function checkExport(which: 'final' | 'dense' = 'final', options: Partial<ExportOptions> = {}) {
  const t0 = performance.now();
  const blob = await viewer.exportGlb(which, { unlit: true, compress: true, ...options });
  const ms_export = Math.round(performance.now() - t0);
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.parseAsync(await blob.arrayBuffer(), '');
  // glTF cannot carry anisotropic filtering; match the viewer's so only file content differs.
  gltf.scene.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
    for (const t of [m?.map, m?.normalMap]) if (t) t.anisotropy = 8;
  });
  const lit = options.unlit === false;
  const views = async (n = 8) => {
    const out: ImageData[] = [];
    for (let i = 0; i < n; i++) out.push(await viewer.snapshot('final', (i / n) * Math.PI * 2, 0.3));
    return out;
  };
  const wasLit = viewer.lit;
  let reference: ImageData[] = [];
  if (lit) { viewer.setLit(true); reference = await views(); }
  const restore = viewer.swapFinal(gltf.scene);
  try {
    if (!lit) return { bytes: blob.size, MB: +(blob.size / 1e6).toFixed(2), ms_export, ...(await compare('final')) };
    // compare() scores unlit renders against the splats, so lit files are scored against the in-app lit mesh.
    const got = await views();
    let se = 0, n = 0;
    got.forEach((img, v) => {
      for (let i = 0; i < img.data.length; i += 4) for (let c = 0; c < 3; c++) { se += (img.data[i + c] - reference[v].data[i + c]) ** 2; n++; }
    });
    return { bytes: blob.size, MB: +(blob.size / 1e6).toFixed(2), ms_export, litPsnr: +(10 * Math.log10((255 * 255) / (se / n))).toFixed(2) };
  } finally {
    restore();
    if (lit) viewer.setLit(wasLit);
  }
}

// Scripting hook for automated checks from DevTools.
Object.assign(window, {
  s2m: {
    loadSample,
    run,
    compare,
    setMode,
    viewer,
    get result() { return result; },
    checkSurface: () => {
      const raw = lastRawSurface();
      if (!raw) throw new Error('No surface yet; run a conversion first.');
      return checkSurface(raw);
    },
    exportGlb: (which: 'final' | 'dense', options: Partial<ExportOptions> = {}) => viewer.exportGlb(which, { unlit: true, compress: true, ...options }),
    checkExport,
  },
});
