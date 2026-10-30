# Import exported GLBs in Blender headlessly and report what came through.
#   /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
#     --python scripts/blender-import-check.py -- [--render] file.glb ...
# --render also writes <file>-blender.png (Workbench, texture colours) next to each GLB.
# Meshopt-compressed files need Blender 5.2+.
import bpy, sys, os, mathutils

args = sys.argv[sys.argv.index('--') + 1:]
render = '--render' in args
for f in [a for a in args if a != '--render']:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=os.path.abspath(f))
    except Exception as e:
        print(f'RESULT {f}: IMPORT FAILED {e}')
        continue
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    v = sum(len(o.data.vertices) for o in meshes)
    p = sum(len(o.data.polygons) for o in meshes)
    imgs = [f'{i.name}:{i.size[0]}x{i.size[1]}:{i.colorspace_settings.name}' for i in bpy.data.images]
    uv = all(o.data.uv_layers for o in meshes)
    dims = tuple(round(d, 3) for d in meshes[0].dimensions) if meshes else None
    print(f'RESULT {f}: objects {len(meshes)} verts {v} faces {p} uv {uv} images {imgs} dims {dims}')
    if not render or not meshes:
        continue
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.color_type = 'TEXTURE'
    sc.render.resolution_x = sc.render.resolution_y = 640
    sc.render.filepath = os.path.abspath(f[:-4] + '-blender.png')
    o = meshes[0]
    bb = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
    ctr = sum(bb, mathutils.Vector()) / 8
    r = max((c - ctr).length for c in bb)
    cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location = ctr + mathutils.Vector((0.35, -1.0, 0.25)).normalized() * r * 2.8
    cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.ops.render.render(write_still=True)
    print(f'RENDER {sc.render.filepath}')
