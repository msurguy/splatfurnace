// Builds the Node CLI (src/cli/index.ts) into dist-cli/index.js; dependencies stay in node_modules.
import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';

const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8')) as { version: string };

// src/xatlas/node.ts loads the wasm from next to the bundle.
function xatlasWasm(): Plugin {
  return {
    name: 'xatlas-wasm',
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'xatlas-wasm.wasm', source: fs.readFileSync('src/xatlas/xatlas-wasm.wasm') });
    },
  };
}

export default defineConfig({
  plugins: [xatlasWasm()],
  define: { __VERSION__: JSON.stringify(pkg.version) },
  build: {
    ssr: 'src/cli/index.ts',
    outDir: 'dist-cli',
    target: 'node22',
    emptyOutDir: true,
    minify: false,
    rollupOptions: { output: { entryFileNames: 'index.js', banner: '#!/usr/bin/env node' } },
  },
});
