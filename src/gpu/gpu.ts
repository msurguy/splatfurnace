import { FIELD_WGSL } from './field.wgsl';
import type { Buckets } from '../grid';

export const S = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };

export interface Gpu {
  device: GPUDevice;
  maxBinding: number;
}

export async function initGpu(): Promise<Gpu> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter.');
  const maxBinding = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: maxBinding,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    },
  });
  device.lost.then((info) => console.error('WebGPU device lost:', info.message));
  return { device, maxBinding };
}

export function buffer(device: GPUDevice, size: number, usage: number, data?: ArrayBufferView) {
  const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
  if (data) device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
  return b;
}

export async function readBack(device: GPUDevice, src: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage: S.MAP_READ | S.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, staging.size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0, size);
  staging.unmap();
  staging.destroy();
  return out;
}

export async function checkShader(module: GPUShaderModule, label: string) {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`${label} shader: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
}

export function splatDispatch(count: number) {
  const groups = Math.ceil(count / 64);
  const x = Math.min(groups, 65535);
  return [x, Math.ceil(groups / x)] as const;
}

/** Persistent splat-ray sampler; queries are (point, normal, range) in grid units. */
export interface FieldSampler {
  sample(points: Float32Array, normals: Float32Array, ranges: Float32Array | number): Promise<FieldResult>;
  dispose(): void;
}

export interface FieldResult {
  colors: Float32Array; // rgb, -1 where nothing was hit
  coverage: Float32Array;
  normals: Float32Array;
  offsets: Float32Array; // signed distance along the input normal to 50% coverage; NaN if none
}

export async function createFieldSampler(gpu: Gpu, splats: Float32Array, buckets: Buckets, minAlpha = 1 / 255): Promise<FieldSampler> {
  const { device } = gpu;
  const module = device.createShaderModule({ code: FIELD_WGSL, label: 'field' });
  await checkShader(module, 'Field');
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'sampleSplats' } });
  const bSplats = buffer(device, splats.byteLength, S.STORAGE | S.COPY_DST, splats);
  const bStarts = buffer(device, buckets.starts.byteLength, S.STORAGE | S.COPY_DST, buckets.starts);
  const bRefs = buffer(device, buckets.refs.byteLength, S.STORAGE | S.COPY_DST, buckets.refs);
  const CHUNK = 1 << 20;

  return {
    async sample(points, normals, ranges) {
      const n = points.length / 3;
      const out: FieldResult = {
        colors: new Float32Array(n * 3), coverage: new Float32Array(n),
        normals: new Float32Array(n * 3), offsets: new Float32Array(n),
      };
      for (let start = 0; start < n; start += CHUNK) {
        const m = Math.min(CHUNK, n - start);
        const q = new Float32Array(m * 8);
        for (let i = 0; i < m; i++) {
          const j = start + i;
          q[i * 8] = points[j * 3]; q[i * 8 + 1] = points[j * 3 + 1]; q[i * 8 + 2] = points[j * 3 + 2];
          q[i * 8 + 3] = typeof ranges === 'number' ? ranges : ranges[j];
          q[i * 8 + 4] = normals[j * 3]; q[i * 8 + 5] = normals[j * 3 + 1]; q[i * 8 + 6] = normals[j * 3 + 2];
        }
        const params = new ArrayBuffer(32);
        new Uint32Array(params, 0, 4).set([...buckets.dims, m]);
        new Float32Array(params, 16, 4).set([buckets.size, minAlpha, 0, 0]);
        const uParams = buffer(device, 32, S.UNIFORM | S.COPY_DST, new Uint8Array(params));
        const bQueries = buffer(device, q.byteLength, S.STORAGE | S.COPY_DST, q);
        const bResults = buffer(device, q.byteLength, S.STORAGE | S.COPY_SRC);
        const group = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [uParams, bSplats, bStarts, bRefs, bQueries, bResults].map((b, binding) => ({ binding, resource: { buffer: b } })),
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        const groups = Math.ceil(m / 64);
        pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
        pass.end();
        device.queue.submit([enc.finish()]);
        const r = new Float32Array(await readBack(device, bResults, q.byteLength));
        for (const b of [uParams, bQueries, bResults]) b.destroy();
        for (let i = 0; i < m; i++) {
          const j = start + i;
          out.colors[j * 3] = r[i * 8]; out.colors[j * 3 + 1] = r[i * 8 + 1]; out.colors[j * 3 + 2] = r[i * 8 + 2];
          out.coverage[j] = r[i * 8 + 3];
          out.normals[j * 3] = r[i * 8 + 4]; out.normals[j * 3 + 1] = r[i * 8 + 5]; out.normals[j * 3 + 2] = r[i * 8 + 6];
          out.offsets[j] = r[i * 8 + 7] > 1e8 ? NaN : r[i * 8 + 7];
        }
      }
      return out;
    },
    dispose() {
      for (const b of [bSplats, bStarts, bRefs]) b.destroy();
    },
  };
}
