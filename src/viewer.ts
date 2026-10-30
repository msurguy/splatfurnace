import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import type { Mesh } from './mesh';
import type { Atlased } from './bake';
import { srgbToLinear } from './export';

export type ViewMode = 'splats' | 'dense' | 'final';

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(40, 1, 0.001, 1000);
  readonly controls: OrbitControls;
  private root = new THREE.Group();
  private splat: SplatMesh | null = null;
  private dense: THREE.Mesh | null = null;
  private final: THREE.Mesh | null = null;
  private wire: THREE.LineSegments | null = null;
  private colorTex: THREE.DataTexture | null = null;
  private normalTex: THREE.DataTexture | null = null;
  mode: ViewMode = 'splats';
  lit = false;
  wireframe = false;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);
    this.scene.background = new THREE.Color(0x15171a);
    this.scene.environment = new THREE.PMREMGenerator(this.renderer).fromScene(new RoomEnvironment(), 0.04).texture;
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(1, 2, 1.5);
    this.scene.add(sun);
    this.scene.add(new SparkRenderer({ renderer: this.renderer }));
    this.scene.add(this.root);
    this.camera.position.set(0, 0, 2);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  /** Splat PLYs are commonly Y-down (OpenCV convention); flipping rotates everything 180 degrees about X. */
  setFlip(flip: boolean) {
    this.root.rotation.x = flip ? Math.PI : 0;
  }

  private resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  async showSplats(bytes: ArrayBuffer, fileName: string) {
    this.clearAll();
    this.splat = new SplatMesh({ fileBytes: bytes.slice(0), fileName });
    this.root.add(this.splat);
    await this.splat.initialized;
    this.setMode('splats');
  }

  frame(center: THREE.Vector3, radius: number) {
    const dir = new THREE.Vector3(0.35, 0.25, 1).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(dir, radius * 2.6);
    this.camera.near = radius / 100;
    this.camera.far = radius * 100;
    this.camera.updateProjectionMatrix();
  }

  setResult(dense: Mesh, final: Atlased, color: Uint8ClampedArray, normal: Uint8ClampedArray | null, size: number) {
    this.disposeMeshes();
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.BufferAttribute(dense.positions, 3));
    dg.setAttribute('normal', new THREE.BufferAttribute(dense.normals, 3));
    if (dense.colors) {
      const lin = new Float32Array(dense.colors.length);
      for (let i = 0; i < lin.length; i++) lin[i] = srgbToLinear(dense.colors[i]);
      dg.setAttribute('color', new THREE.BufferAttribute(lin, 3));
    }
    dg.setIndex(new THREE.BufferAttribute(dense.indices, 1));
    this.dense = new THREE.Mesh(dg, new THREE.MeshBasicMaterial({ vertexColors: true }));
    this.dense.name = 'dense';

    this.colorTex = dataTexture(color, size, THREE.SRGBColorSpace);
    this.colorTex.userData.mimeType = 'image/jpeg'; // GLTFExporter: far smaller than PNG for photo-like colour
    this.normalTex = normal ? dataTexture(normal, size, THREE.NoColorSpace) : null;
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(final.positions, 3));
    fg.setAttribute('normal', new THREE.BufferAttribute(final.normals, 3));
    fg.setAttribute('uv', new THREE.BufferAttribute(final.uvs, 2));
    fg.setAttribute('tangent', new THREE.BufferAttribute(final.tangents, 4));
    this.final = new THREE.Mesh(fg, new THREE.MeshBasicMaterial());
    this.final.name = 'splatfurnace';
    this.wire = new THREE.LineSegments(new THREE.WireframeGeometry(fg), new THREE.LineBasicMaterial({ color: 0x9be15d, transparent: true, opacity: 0.35 }));
    this.root.add(this.dense, this.final, this.wire);
    this.applyMaterials();
    this.setMode('final');
    fg.computeBoundingSphere();
    this.frame(fg.boundingSphere!.center.clone().applyMatrix4(this.root.matrixWorld), fg.boundingSphere!.radius);
  }

  private makeMaterial(unlitForExport = false): THREE.Material {
    if (!this.lit || unlitForExport) return new THREE.MeshBasicMaterial({ map: this.colorTex });
    return new THREE.MeshStandardMaterial({ map: this.colorTex, normalMap: this.normalTex, roughness: 0.9, metalness: 0 });
  }

  private applyMaterials() {
    if (this.final) {
      (this.final.material as THREE.Material).dispose();
      this.final.material = this.makeMaterial();
    }
    if (this.dense) {
      (this.dense.material as THREE.Material).dispose();
      this.dense.material = this.lit
        ? new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0 })
        : new THREE.MeshBasicMaterial({ vertexColors: true });
    }
  }

  setLit(lit: boolean) { this.lit = lit; this.applyMaterials(); }
  setWireframe(on: boolean) { this.wireframe = on; this.setMode(this.mode); }

  setMode(mode: ViewMode) {
    this.mode = mode;
    if (this.splat) this.splat.visible = mode === 'splats';
    if (this.dense) this.dense.visible = mode === 'dense';
    if (this.final) this.final.visible = mode === 'final';
    if (this.wire) this.wire.visible = mode === 'final' && this.wireframe;
  }

  /** Render `mode` from an orbit angle around the current target and return its pixels. */
  async snapshot(mode: ViewMode, azimuth: number, elevation: number, width = 480): Promise<ImageData> {
    this.controls.enableDamping = false;
    const target = this.controls.target;
    const dist = this.camera.position.distanceTo(target);
    this.camera.position.set(
      target.x + dist * Math.cos(elevation) * Math.sin(azimuth),
      target.y + dist * Math.sin(elevation),
      target.z + dist * Math.cos(elevation) * Math.cos(azimuth),
    );
    this.camera.lookAt(target);
    this.setMode(mode);
    // Spark re-sorts splats asynchronously; give it a few frames to settle.
    for (let i = 0; i < 6; i++) await new Promise((r) => requestAnimationFrame(r));
    const src = this.renderer.domElement;
    const height = Math.round((width * src.height) / src.width);
    const c = new OffscreenCanvas(width, height);
    const ctx = c.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(src, 0, 0, width, height);
    this.controls.enableDamping = true;
    return ctx.getImageData(0, 0, width, height);
  }

  /**
   * Temporarily stand `object` in for the final mesh, e.g. an exported GLB loaded back for a
   * round-trip check. It goes on the scene, not the root: exports already carry the root transform.
   * Returns an undo.
   */
  swapFinal(object: THREE.Object3D): () => void {
    const original = this.final;
    if (original) original.visible = false;
    this.scene.add(object);
    this.final = object as THREE.Mesh; // snapshot('final') toggles this.final
    return () => {
      this.scene.remove(object);
      this.final = original;
      this.setMode(this.mode);
    };
  }

  private disposeMeshes() {
    for (const m of [this.dense, this.final, this.wire]) {
      if (!m) continue;
      this.root.remove(m);
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    }
    this.colorTex?.dispose();
    this.normalTex?.dispose();
    this.dense = this.final = this.wire = null;
  }

  private clearAll() {
    this.disposeMeshes();
    if (this.splat) {
      this.root.remove(this.splat);
      this.splat.dispose();
      this.splat = null;
    }
  }
}

function dataTexture(data: Uint8ClampedArray, size: number, colorSpace: THREE.ColorSpace) {
  const t = new THREE.DataTexture(new Uint8Array(data.buffer), size, size, THREE.RGBAFormat);
  t.colorSpace = colorSpace;
  t.flipY = false;
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.anisotropy = 8;
  t.needsUpdate = true;
  return t;
}
