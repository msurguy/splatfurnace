import { DEFAULTS, type Settings } from './pipeline';

export type Preset = 'draft' | 'balanced' | 'detailed';
export const PRESETS: Record<Preset, Settings> = {
  draft: { ...DEFAULTS, resolution: 224, targetTriangles: 15000, textureSize: 1024, normalMap: false },
  balanced: DEFAULTS,
  detailed: { ...DEFAULTS, resolution: 448, targetTriangles: 80000, textureSize: 4096 },
};

export function matchPreset(s: Settings): Preset | 'custom' {
  for (const [name, p] of Object.entries(PRESETS) as [Preset, Settings][]) {
    if ((Object.keys(p) as (keyof Settings)[]).every((k) => Math.abs(Number(p[k]) - Number(s[k])) < 1e-9)) return name;
  }
  return 'custom';
}
