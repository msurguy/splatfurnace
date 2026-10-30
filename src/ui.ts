import { createIcons, Box, Check, ChevronDown, CircleAlert, Download, FileBox, FlipVertical2, Focus, Grid3x3, Info, RotateCcw, Sparkles, Sun, Terminal, Upload } from 'lucide';
import { DEFAULTS, type Settings } from './pipeline';

export const ICONS = { Box, Check, ChevronDown, CircleAlert, Download, FileBox, FlipVertical2, Focus, Grid3x3, Info, RotateCcw, Sparkles, Sun, Terminal, Upload };
export const icons = (root: Element | Document = document) => createIcons({ icons: ICONS, root, attrs: { 'aria-hidden': 'true' } });

// ---------------------------------------------------------------- settings fields
type Key = keyof Settings;
interface Range { kind: 'range'; name: Key; label: string; tip: string; min: number; max: number; step: number; fmt?: (v: number) => string; parse?: (s: string) => number }
interface Toggle { kind: 'switch'; name: Key; label: string; tip: string }
interface Choice { kind: 'choice'; name: Key; label: string; tip: string; options: [number, string][] }
type Field = (Range | Toggle | Choice) & { advanced?: boolean; dependsOn?: Key };

const pct = { fmt: (v: number) => `${+(v * 100).toFixed(1)}%`, parse: (s: string) => parseFloat(s) / 100 };

export const FIELDS: Field[] = [
  { kind: 'range', name: 'resolution', label: 'Detail', tip: 'How finely the surface is sampled. Higher captures thinner features but takes longer.|Voxel grid resolution', min: 128, max: 512, step: 16 },
  { kind: 'range', name: 'targetTriangles', label: 'Triangles', tip: 'Polygon budget of the final mesh.|Decimation target', min: 1000, max: 200000, step: 1000, fmt: (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v)), parse: (s) => (/k$/i.test(s.trim()) ? parseFloat(s) * 1000 : parseFloat(s)) },
  { kind: 'choice', name: 'textureSize', label: 'Texture', tip: 'Size of the baked colour (and normal) texture.|Texture size, px', options: [[1024, '1K'], [2048, '2K'], [4096, '4K']] },
  { kind: 'switch', name: 'normalMap', label: 'Normal map', tip: 'Bakes fine surface detail into a texture for lit scenes.|Tangent-space normal map' },

  { advanced: true, kind: 'range', name: 'iso', label: 'Density threshold', tip: 'Where the surface sits in the splat cloud. Lower inflates, higher erodes.|iso', min: 0.05, max: 1, step: 0.01 },
  { advanced: true, kind: 'range', name: 'minSigma', label: 'Min splat width', tip: 'Widens tiny splats so they still contribute to the surface.|minSigma, voxels', min: 0.3, max: 2, step: 0.05 },
  { advanced: true, kind: 'range', name: 'opacityThreshold', label: 'Opacity cutoff', tip: 'Ignore splats fainter than this. Removes haze and floaters.|opacityThreshold', min: 0, max: 0.5, step: 0.01 },
  { advanced: true, kind: 'switch', name: 'fillCavities', label: 'Seal interior cavities', tip: 'Fills hidden pockets inside the object so the mesh is one clean shell.|fillCavities' },
  { advanced: true, kind: 'range', name: 'islandFraction', label: 'Remove small pieces', tip: 'Drops loose fragments smaller than this share of the largest piece.|islandFraction', min: 0, max: 0.1, step: 0.005, ...pct },
  { advanced: true, kind: 'range', name: 'smoothing', label: 'Smoothing', tip: 'Passes of crease-preserving smoothing before snapping.|Taubin iterations', min: 0, max: 10, step: 1 },
  { advanced: true, kind: 'switch', name: 'snap', label: 'Snap to visible surface', tip: 'Moves each vertex to where the splats turn opaque. Sharper, more faithful shape.|snap' },
  { advanced: true, kind: 'range', name: 'maxSnap', label: 'Max snap distance', tip: 'How far a vertex may move when snapping.|maxSnap, voxels', min: 0.5, max: 6, step: 0.5, dependsOn: 'snap' },
  { advanced: true, kind: 'switch', name: 'chartAtlas', label: 'Chart UV layout', tip: 'Groups triangles into UV charts: about 3× fewer vertices, ~1 s slower.|xatlas' },
];

const tipAttr = (tip: string) => tip.replace(/"/g, '&quot;');
const info = (tip: string) => `<i data-lucide="info" class="info size-3" data-tip="${tipAttr(tip)}" tabindex="0"></i>`;

function fieldHtml(f: Field): string {
  const dep = f.dependsOn ? ` data-depends="${f.dependsOn}"` : '';
  if (f.kind === 'switch') {
    return `<label class="field"${dep}><span class="field-head">${f.label}${info(f.tip)}<input name="${f.name}" type="checkbox" class="switch ml-auto" /></span></label>`;
  }
  if (f.kind === 'choice') {
    const opts = f.options.map(([v, l]) => `<label><input class="sr-only" type="radio" name="${f.name}" value="${v}" />${l}</label>`).join('');
    return `<div class="field"${dep}><span class="field-head">${f.label}${info(f.tip)}<span class="seg ml-auto w-36">${opts}</span></span></div>`;
  }
  return `<div class="field"${dep}>
    <span class="field-head"><label for="f-${f.name}">${f.label}</label>${info(f.tip)}<input class="val" data-for="${f.name}" aria-label="${f.label} value" inputmode="decimal" /></span>
    <input id="f-${f.name}" class="range" name="${f.name}" type="range" min="${f.min}" max="${f.max}" step="${f.step}" />
  </div>`;
}

export function renderFields(essentials: HTMLElement, advanced: HTMLElement) {
  essentials.innerHTML = FIELDS.filter((f) => !f.advanced).map(fieldHtml).join('');
  advanced.innerHTML = FIELDS.filter((f) => f.advanced).map(fieldHtml).join('');
}

/** Refresh slider fills, value readouts and dependent-field states from the form's inputs. */
export function syncFields(form: HTMLFormElement) {
  for (const f of FIELDS) {
    if (f.kind === 'range') {
      const r = form.querySelector<HTMLInputElement>(`input[name="${f.name}"]`)!;
      const v = Number(r.value);
      r.style.setProperty('--fill', `${((v - f.min) / (f.max - f.min)) * 100}%`);
      const out = form.querySelector<HTMLInputElement>(`.val[data-for="${f.name}"]`)!;
      if (document.activeElement !== out) out.value = f.fmt ? f.fmt(v) : String(v);
    }
  }
  for (const row of Array.from(form.querySelectorAll<HTMLElement>('[data-depends]'))) {
    const on = form.querySelector<HTMLInputElement>(`input[name="${row.dataset.depends}"]`)!.checked;
    row.classList.toggle('off', !on);
    for (const el of Array.from(row.querySelectorAll('input'))) el.disabled = !on;
  }
}

/** Typed values in the readouts: parse, clamp, snap to step, then push into the slider. */
export function bindValueInputs(form: HTMLFormElement, onChange: () => void) {
  for (const f of FIELDS) {
    if (f.kind !== 'range') continue;
    const out = form.querySelector<HTMLInputElement>(`.val[data-for="${f.name}"]`)!;
    const r = form.querySelector<HTMLInputElement>(`input[name="${f.name}"]`)!;
    const commit = () => {
      const v = (f.parse ?? parseFloat)(out.value);
      if (Number.isFinite(v)) r.value = String(Math.min(f.max, Math.max(f.min, Math.round((v - f.min) / f.step) * f.step + f.min)));
      out.blur();
      onChange();
    };
    out.addEventListener('change', commit);
    out.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); commit(); }
      if (e.key === 'Escape') { out.value = ''; out.blur(); onChange(); }
    });
    out.addEventListener('focus', () => out.select());
  }
}

// ---------------------------------------------------------------- presets
export type Preset = 'draft' | 'balanced' | 'detailed';
export const PRESETS: Record<Preset, Settings> = {
  draft: { ...DEFAULTS, resolution: 224, targetTriangles: 15000, textureSize: 1024, normalMap: false },
  balanced: DEFAULTS,
  detailed: { ...DEFAULTS, resolution: 448, targetTriangles: 80000, textureSize: 4096 },
};

export function matchPreset(s: Settings): Preset | 'custom' {
  for (const [name, p] of Object.entries(PRESETS) as [Preset, Settings][]) {
    if ((Object.keys(p) as Key[]).every((k) => Math.abs(Number(p[k]) - Number(s[k])) < 1e-9)) return name;
  }
  return 'custom';
}

// ---------------------------------------------------------------- progress
export const STAGES = ['Surface', 'Cleanup', 'Snap', 'Colour', 'Simplify', 'Atlas', 'Bake'] as const;
const STAGE_FROM_LOG: [RegExp, number][] = [
  [/^Converting/, 0], [/^Removing floating/, 1], [/^Smoothing/, 1], [/^Snapping/, 2],
  [/^Ray-sampling/, 3], [/^Simplifying/, 4], [/^Building texture atlas/, 5], [/^Baking/, 6],
];
export const stageFromLog = (msg: string) => STAGE_FROM_LOG.find(([re]) => re.test(msg))?.[1] ?? -1;

const STAGE_LABEL = ['Extracting surface', 'Cleaning up', 'Snapping to splats', 'Sampling colour', 'Simplifying', 'Laying out UVs', 'Baking textures'];

export function renderStages(list: HTMLElement, label: HTMLElement, current: number, skipSnap: boolean) {
  list.innerHTML = STAGES.map((name, i) => {
    const cls = skipSnap && i === 2 ? 'skip' : i < current ? 'done' : i === current ? 'now' : '';
    return `<li class="${cls}" title="${name}"></li>`;
  }).join('');
  label.textContent = current < 0 ? 'Starting' : STAGE_LABEL[current];
}

// ---------------------------------------------------------------- toasts + tooltips
export function toast(msg: string, kind: 'info' | 'error' = 'info', ms = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  el.innerHTML = `<i data-lucide="${kind === 'error' ? 'circle-alert' : 'check'}" class="size-4 shrink-0 ${kind === 'error' ? '' : 'text-accent'}"></i><span></span>`;
  el.querySelector('span')!.textContent = msg;
  document.getElementById('toasts')!.appendChild(el);
  icons(el);
  setTimeout(() => el.remove(), kind === 'error' ? ms * 2 : ms);
}

/** One fixed-position tooltip for every [data-tip]; text after "|" is shown as a technical footnote. */
export function initTooltips() {
  const tip = document.getElementById('tip')!;
  let target: HTMLElement | null = null;
  const show = (el: HTMLElement) => {
    target = el;
    const [text, tech] = el.dataset.tip!.split('|');
    tip.textContent = text;
    if (tech) { const em = document.createElement('em'); em.textContent = tech; tip.appendChild(em); }
    const r = el.getBoundingClientRect();
    tip.classList.add('show');
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const above = r.top - h - 8 > 4;
    tip.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left + r.width / 2 - w / 2))}px`;
    tip.style.top = `${above ? r.top - h - 8 : r.bottom + 8}px`;
  };
  const hide = () => { target = null; tip.classList.remove('show'); };
  const find = (e: Event) => (e.target as Element).closest?.<HTMLElement>('[data-tip]');
  document.addEventListener('pointerover', (e) => { const el = find(e); if (el && el !== target) show(el); else if (!el && target) hide(); });
  document.addEventListener('focusin', (e) => { const el = find(e); if (el) show(el); });
  document.addEventListener('focusout', hide);
  document.addEventListener('scroll', hide, true);
}
