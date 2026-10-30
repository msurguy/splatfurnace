// Minimal C interface to xatlas for the WebAssembly build (scripts/build-xatlas.sh).
// One atlas at a time per module instance. 32-bit indices throughout. Charts are computed
// once and can be packed repeatedly, so a pack that overshoots the target resolution can be
// redone at a corrected scale without re-charting.
#include <stdint.h>
#include "../xatlas/xatlas.h"

static xatlas::Atlas *atlas = nullptr;

extern "C" {

void xa_create() {
	if (atlas) xatlas::Destroy(atlas);
	atlas = xatlas::Create();
}

void xa_destroy() {
	if (atlas) xatlas::Destroy(atlas);
	atlas = nullptr;
}

// positions/normals: xyz floats; normals may be null. Returns xatlas::AddMeshError.
int xa_add_mesh(const float *positions, const float *normals, uint32_t vertexCount, const uint32_t *indices, uint32_t indexCount) {
	xatlas::MeshDecl d;
	d.vertexPositionData = positions;
	d.vertexPositionStride = 12;
	if (normals) {
		d.vertexNormalData = normals;
		d.vertexNormalStride = 12;
	}
	d.vertexCount = vertexCount;
	d.indexData = indices;
	d.indexCount = indexCount;
	d.indexFormat = xatlas::IndexFormat::UInt32;
	return (int)xatlas::AddMesh(atlas, d);
}

// Mesh whose charts are already given by UV islands (uv: xy floats); packing only.
int xa_add_uv_mesh(const float *uvs, uint32_t vertexCount, const uint32_t *indices, uint32_t indexCount) {
	xatlas::UvMeshDecl d;
	d.vertexUvData = uvs;
	d.vertexStride = 8;
	d.vertexCount = vertexCount;
	d.indexData = indices;
	d.indexCount = indexCount;
	d.indexFormat = xatlas::IndexFormat::UInt32;
	return (int)xatlas::AddUvMesh(atlas, d);
}

// o: normalDeviationWeight, roundnessWeight, straightnessWeight, normalSeamWeight,
//    textureSeamWeight, maxCost, maxIterations, maxChartArea, maxBoundaryLength
void xa_compute_charts(const float *o) {
	xatlas::ChartOptions c;
	c.normalDeviationWeight = o[0];
	c.roundnessWeight = o[1];
	c.straightnessWeight = o[2];
	c.normalSeamWeight = o[3];
	c.textureSeamWeight = o[4];
	c.maxCost = o[5];
	c.maxIterations = (uint32_t)o[6];
	c.maxChartArea = o[7];
	c.maxBoundaryLength = o[8];
	xatlas::ComputeCharts(atlas, c);
}

// o: resolution, padding, texelsPerUnit, bilinear, blockAlign, bruteForce, maxChartSize,
//    rotateChartsToAxis, rotateCharts
void xa_pack_charts(const float *o) {
	xatlas::PackOptions p;
	p.resolution = (uint32_t)o[0];
	p.padding = (uint32_t)o[1];
	p.texelsPerUnit = o[2];
	p.bilinear = o[3] != 0;
	p.blockAlign = o[4] != 0;
	p.bruteForce = o[5] != 0;
	p.maxChartSize = (uint32_t)o[6];
	p.rotateChartsToAxis = o[7] != 0;
	p.rotateCharts = o[8] != 0;
	xatlas::PackCharts(atlas, p);
}

// width, height, atlasCount, chartCount, meshCount, texelsPerUnit (as float bits)
void xa_atlas_info(uint32_t *out) {
	out[0] = atlas->width;
	out[1] = atlas->height;
	out[2] = atlas->atlasCount;
	out[3] = atlas->chartCount;
	out[4] = atlas->meshCount;
	*(float *)&out[5] = atlas->texelsPerUnit;
}

// vertexCount, indexCount, chartCount
void xa_mesh_info(uint32_t mesh, uint32_t *out) {
	const xatlas::Mesh &m = atlas->meshes[mesh];
	out[0] = m.vertexCount;
	out[1] = m.indexCount;
	out[2] = m.chartCount;
}

// xref: vertexCount u32, uv: vertexCount * 2 floats in texels, atlasIndex: vertexCount i32
// (may be null), indices: indexCount u32.
void xa_mesh_copy(uint32_t mesh, uint32_t *xref, float *uv, int32_t *atlasIndex, uint32_t *indices) {
	const xatlas::Mesh &m = atlas->meshes[mesh];
	for (uint32_t i = 0; i < m.vertexCount; i++) {
		const xatlas::Vertex &v = m.vertexArray[i];
		xref[i] = v.xref;
		uv[i * 2] = v.uv[0];
		uv[i * 2 + 1] = v.uv[1];
		if (atlasIndex) atlasIndex[i] = v.atlasIndex;
	}
	for (uint32_t i = 0; i < m.indexCount; i++) indices[i] = m.indexArray[i];
}

}
