#!/bin/sh
# Build the xatlas WebAssembly module used by src/unwrap.ts.
# Needs Emscripten (brew install emscripten). Output is committed, so this only needs
# rerunning after changing third_party/xatlas or third_party/xatlas-wasm/binding.cpp.
set -e
cd "$(dirname "$0")/.."
mkdir -p src/xatlas
em++ third_party/xatlas-wasm/binding.cpp third_party/xatlas/xatlas.cpp \
  -O3 -flto -msimd128 -DNDEBUG -DXA_MULTITHREADED=0 -fno-exceptions -fno-rtti \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createXAtlas -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sFILESYSTEM=0 -sINITIAL_MEMORY=64MB \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_xa_create,_xa_destroy,_xa_add_mesh,_xa_add_uv_mesh,_xa_compute_charts,_xa_pack_charts,_xa_atlas_info,_xa_mesh_info,_xa_mesh_copy \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8 \
  -o src/xatlas/xatlas-wasm.mjs
ls -l src/xatlas
