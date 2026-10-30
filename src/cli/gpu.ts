// WebGPU in Node through Dawn (the `webgpu` package). The pipeline code is the same as in the
// browser; only the GPU object comes from here instead of navigator.gpu.
import { create, globals } from 'webgpu';
import { initGpu, type Gpu } from '../gpu/gpu';

// Dawn tears down every adapter and device when the GPU object is collected, so it lives here.
let instance: GPU | null = null;

export async function initNodeGpu(): Promise<Gpu> {
  Object.assign(globalThis, globals); // GPUBufferUsage, GPUShaderStage, ... as in a browser
  instance ??= create([]);
  return initGpu(instance);
}
