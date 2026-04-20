"""QA mod-9 torus — Blender bpy render script.

Run:
    blender -b -P tools/qa_viz/blender/qa_torus.py -- --out /tmp/qa_torus.png

Matches the Three.js scene in tools/qa_viz/threejs/qa_torus.html:
  theta = 2 pi (b-1)/9, phi = 2 pi (e-1)/9
  Singularity (9,9) | Satellite b=e, b!=9 | Cosmos otherwise.
"""
QA_COMPLIANCE = {
    "observer": "render_projection",
    "state_alphabet": "integer (b,e) in {1..9}^2; continuous floats used ONLY in the observer layer (torus embedding for rendering); no QA dynamics run in this file",
}

import argparse
import math
import sys

try:
    import bpy
except ImportError:
    sys.exit("bpy not found — run under Blender: blender -b -P <this file>")


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/qa_torus.png")
    p.add_argument("--R", type=float, default=2.4)
    p.add_argument("--r", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--res", type=int, nargs=2, default=[1600, 1000])
    p.add_argument("--engine", choices=["cycles", "eevee"], default="eevee",
                   help="cycles = slow physically-accurate; eevee = fast realtime (default)")
    return p.parse_args(argv)


def classify(b, e, m=9):
    """Canonical per qa_orbit_rules.py: Satellite = (m/3)|b ∧ (m/3)|e ∧ not (m,m)."""
    sat_div = m // 3
    if b == m and e == m:
        return "singularity"
    if b % sat_div == 0 and e % sat_div == 0:
        return "satellite"
    return "cosmos"


def toroidal(b, e, R, r):
    theta = 2 * math.pi * (b - 1) / 9
    phi = 2 * math.pi * (e - 1) / 9
    x = (R + r * math.cos(phi)) * math.cos(theta)
    y = (R + r * math.cos(phi)) * math.sin(theta)
    z = r * math.sin(phi)
    return x, y, z


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.materials, bpy.data.meshes, bpy.data.lights, bpy.data.cameras):
        for item in list(coll):
            coll.remove(item)


def make_emissive(name, rgb, strength=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    emit.inputs["Strength"].default_value = strength
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def make_glass(name, rgb, alpha=0.35):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.25
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    mat.blend_method = "BLEND"
    return mat


def setup_bloom(scene):
    """Compositor Glare node — Eevee Next lost the old bloom toggle."""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    glare.glare_type = "BLOOM" if hasattr(glare, "glare_type") and "BLOOM" in {i.identifier for i in glare.bl_rna.properties["glare_type"].enum_items} else "FOG_GLOW"
    glare.quality = "HIGH"
    if hasattr(glare, "size"):
        glare.size = 6
    if hasattr(glare, "threshold"):
        glare.threshold = 2.5
    if hasattr(glare, "mix"):
        glare.mix = 0.6  # +1 = image only, -1 = glare only; 0.6 keeps image dominant
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], comp.inputs["Image"])


def main():
    args = parse_args()
    clear_scene()

    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = max(16, args.samples)
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("World") if not scene.world else scene.world
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.028, 0.034, 0.055, 1.0)
        bg.inputs["Strength"].default_value = 1.0

    setup_bloom(scene)

    bpy.ops.mesh.primitive_torus_add(
        major_radius=args.R, minor_radius=args.r, major_segments=128, minor_segments=64
    )
    torus = bpy.context.active_object
    torus.data.materials.append(make_glass("torus_glass", (0.22, 0.28, 0.38), alpha=0.72))

    mats = {
        "cosmos": make_emissive("mat_cosmos", (0.32, 0.62, 1.0), 3.0),
        "satellite": make_emissive("mat_satellite", (1.0, 0.32, 0.32), 4.0),
        "singularity": make_emissive("mat_singularity", (1.0, 0.78, 0.22), 7.0),
    }
    sizes = {"cosmos": 0.10, "satellite": 0.15, "singularity": 0.22}

    for b in range(1, 10):
        for e in range(1, 10):
            cls = classify(b, e)
            x, y, z = toroidal(b, e, args.R, args.r)
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=sizes[cls], location=(x, y, z), segments=20, ring_count=12
            )
            obj = bpy.context.active_object
            obj.data.materials.append(mats[cls])
            obj.name = f"pt_{b}_{e}_{cls}"

    # Camera + TRACK_TO empty: guarantee full torus is framed regardless of R/r.
    target = bpy.data.objects.new("cam_target", None)
    target.location = (0, 0, 0)
    scene.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 42   # wider FOV (was 55)
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    # Pull back: torus outer radius = R + r = 3.4; 12 units distance gives plenty of margin.
    cam.location = (8.5, -8.5, 7.2)
    scene.camera = cam
    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # key / fill / rim lights to show torus 3D form
    for loc, energy, kind in [
        ((7, -5, 9), 900, "POINT"),      # key
        ((-5, -3, 5), 450, "POINT"),     # fill
        ((0, 6, 3), 600, "POINT"),       # rim (behind torus)
        ((0, 0, -4), 250, "POINT"),      # under-lighter for bottom nodes
    ]:
        light_data = bpy.data.lights.new("L", type=kind)
        light_data.energy = energy
        light = bpy.data.objects.new("L", light_data)
        light.location = loc
        scene.collection.objects.link(light)

    scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)
    print(f"[qa_torus] wrote {args.out}")


if __name__ == "__main__":
    main()
