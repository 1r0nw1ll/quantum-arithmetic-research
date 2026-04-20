"""Figure 3 — Primitive Pythagorean triples with hypotenuse c <= 100 in
Euclid generator space (m, n, c). Highlights the 3 all-Fibonacci triples
(3:4:5, 5:12:13, 39:80:89) vs the 13 non-Fibonacci ones.

Supports §6.1 of the Fibonacci resonance paper.

Run:
    blender -b -P tools/qa_viz/blender/qa_pythagorean_fib.py -- \
        --out papers/in-progress/fibonacci-resonance/figures/figure3_pythagorean_fib.png
"""
QA_COMPLIANCE = {
    "observer": "render_projection",
    "state_alphabet": "integer Euclid generators (m, n) with m > n > 0, gcd=1, m-n odd, c = m^2+n^2; Fibonacci membership via integer test on (m-n, n, m, m+n); continuous floats used ONLY in observer layer (scene geometry for rendering)",
}

import argparse
import math
import sys
from math import gcd

try:
    import bpy
except ImportError:
    sys.exit("bpy not found — run under Blender: blender -b -P <this file>")


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/qa_pythagorean_fib.png")
    p.add_argument("--c-max", type=int, default=100)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--res", type=int, nargs=2, default=[1800, 1200])
    p.add_argument("--engine", choices=["cycles", "eevee"], default="eevee")
    return p.parse_args(argv)


def fibs_up_to(n):
    out, a, b = set(), 1, 1
    while a <= n:
        out.add(a)
        a, b = b, a + b
    return out


def enumerate_primitive_triples(c_max):
    """Euclid: (m, n) with m > n > 0, gcd=1, m-n odd. c = m^2 + n^2."""
    triples = []
    m = 2
    while m * m <= c_max:
        for n in range(1, m):
            if (m - n) % 2 == 0:
                continue
            if gcd(m, n) != 1:
                continue
            a, b, c = m * m - n * n, 2 * m * n, m * m + n * n
            if c > c_max:
                continue
            if a > b:
                a, b = b, a
            triples.append({"m": m, "n": n, "a": a, "b": b, "c": c,
                            "qn": (m - n, n, m, m + n)})
        m += 1
    return triples


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.materials, bpy.data.meshes, bpy.data.lights,
                 bpy.data.cameras, bpy.data.curves):
        for item in list(coll):
            coll.remove(item)


def make_emissive(name, rgb, strength):
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


def make_principled(name, rgb, roughness=0.4, metallic=0.2, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    if alpha < 1.0 and "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
        mat.blend_method = "BLEND"
    return mat


def setup_bloom(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    try:
        items = {i.identifier for i in glare.bl_rna.properties["glare_type"].enum_items}
        glare.glare_type = "BLOOM" if "BLOOM" in items else "FOG_GLOW"
    except Exception:
        glare.glare_type = "FOG_GLOW"
    glare.quality = "HIGH"
    if hasattr(glare, "size"):
        glare.size = 6
    if hasattr(glare, "threshold"):
        glare.threshold = 2.5
    if hasattr(glare, "mix"):
        glare.mix = 0.6
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], comp.inputs["Image"])


def add_axis(length, radius, color, label, label_pos, rot=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                        location=(0, 0, length / 2))
    axis = bpy.context.active_object
    if rot is not None:
        axis.rotation_euler = rot
        # shift so cylinder base sits at origin after rotation
        # simpler: rebuild location
        bpy.ops.object.delete()
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                            location=(0, 0, 0))
        axis = bpy.context.active_object
        axis.rotation_euler = rot
        # translate along axis direction to put base at origin
        half = length / 2
        rx, ry, rz = rot
        if abs(rx) > 0.01:  # x-axis (rotated about y)
            pass
        axis = bpy.context.active_object
    axis.data.materials.append(make_principled(f"ax_{label}", color,
                                                roughness=0.5, metallic=0.3))
    return axis


def add_text(body, location, size=0.6, rotation=(math.radians(90), 0, 0),
             rgb=(1, 1, 1), extrude=0.02, track_target=None):
    bpy.ops.object.text_add(location=location, rotation=rotation)
    t = bpy.context.active_object
    t.data.body = body
    t.data.size = size
    t.data.extrude = extrude
    t.data.align_x = "CENTER"
    t.data.materials.append(make_emissive(f"txt_{body}", rgb, 2.0))
    if track_target is not None:
        # clear rotation first so TRACK_TO fully controls orientation
        t.rotation_euler = (0, 0, 0)
        c = t.constraints.new(type="TRACK_TO")
        c.target = track_target
        c.track_axis = "TRACK_Z"
        c.up_axis = "UP_Y"
    return t


def main():
    args = parse_args()
    triples = enumerate_primitive_triples(args.c_max)
    fib = fibs_up_to(args.c_max * 2)
    for t in triples:
        t["all_fib"] = all(x in fib for x in t["qn"])

    n_fib = sum(1 for t in triples if t["all_fib"])
    print(f"[qa_pyth_fib] enumerated {len(triples)} primitive triples c<={args.c_max}; "
          f"{n_fib} all-Fibonacci")

    clear_scene()

    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = max(16, args.samples)
    scene.render.resolution_x, scene.render.resolution_y = args.res

    scene.world = bpy.data.worlds.new("World") if not scene.world else scene.world
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.028, 0.034, 0.055, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    setup_bloom(scene)

    # Coordinate scaling — c range 5..100, m/n range 1..10. Scale c to match.
    X = lambda m: float(m)
    Y = lambda n: float(n)
    Z = lambda c: float(c) / 8.0  # c=100 → z=12.5

    # Materials
    mat_fib = make_emissive("mat_fib", (1.0, 0.78, 0.18), 5.0)
    mat_reg = make_emissive("mat_reg", (0.55, 0.72, 1.0), 2.2)

    R_FIB, R_REG = 0.45, 0.22

    # Ground grid (m, n) plane at z = 0
    grid = bpy.data.meshes.new("grid")
    verts, edges = [], []
    for i in range(11):
        verts.append((i, 0, 0)); verts.append((i, 10, 0))
        edges.append((len(verts) - 2, len(verts) - 1))
        verts.append((0, i, 0)); verts.append((10, i, 0))
        edges.append((len(verts) - 2, len(verts) - 1))
    grid.from_pydata(verts, edges, [])
    grid_obj = bpy.data.objects.new("grid", grid)
    scene.collection.objects.link(grid_obj)
    grid_mat = make_principled("grid_mat", (0.35, 0.42, 0.55), roughness=0.9, metallic=0.0, alpha=0.35)
    grid_obj.data.materials.append(grid_mat)

    # Points
    fib_points = []
    fib_labels_pending = []  # defer text creation until camera target exists
    for t in triples:
        m, n, c = t["m"], t["n"], t["c"]
        loc = (X(m), Y(n), Z(c))
        radius = R_FIB if t["all_fib"] else R_REG
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc,
                                              segments=24, ring_count=16)
        sphere = bpy.context.active_object
        sphere.data.materials.append(mat_fib if t["all_fib"] else mat_reg)
        sphere.name = f"trip_{t['a']}_{t['b']}_{t['c']}"

        # vertical stem from grid to point (helps read height)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.03, depth=Z(c),
            location=(X(m), Y(n), Z(c) / 2))
        stem = bpy.context.active_object
        stem.data.materials.append(grid_mat)

        if t["all_fib"]:
            fib_points.append(loc)
            label = f"{t['a']}:{t['b']}:{t['c']}"
            fib_labels_pending.append((label, (X(m), Y(n), Z(c) + 0.9)))

    # Fibonacci ladder: line connecting the 3 all-Fib points
    if len(fib_points) >= 2:
        ladder_mesh = bpy.data.meshes.new("ladder")
        edges = [(i, i + 1) for i in range(len(fib_points) - 1)]
        ladder_mesh.from_pydata(fib_points, edges, [])
        ladder = bpy.data.objects.new("ladder", ladder_mesh)
        scene.collection.objects.link(ladder)
        # Give it thickness via skin modifier
        ladder.modifiers.new("Skin", type="SKIN")
        for v in ladder.data.skin_vertices[0].data:
            v.radius = (0.06, 0.06)
        ladder_mat = make_emissive("ladder_mat", (1.0, 0.78, 0.18), 1.5)
        ladder.data.materials.append(ladder_mat)

    # Camera + TRACK_TO empty so framing is guaranteed
    target = bpy.data.objects.new("cam_target", None)
    target.location = (5.0, 5.0, Z(args.c_max) * 0.45)
    scene.collection.objects.link(target)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 40
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (22, -20, 16)
    scene.camera = cam
    track = cam.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Billboard labels — text TRACK_TOs the camera
    for label, loc in fib_labels_pending:
        add_text(label, location=loc, size=0.85, rgb=(1.0, 0.92, 0.55),
                 track_target=cam)
    add_text("m", location=(10.8, -0.8, 0), size=1.0, rgb=(0.8, 0.9, 1.0),
             track_target=cam)
    add_text("n", location=(-0.8, 10.8, 0), size=1.0, rgb=(0.8, 0.9, 1.0),
             track_target=cam)
    add_text("c/8", location=(-0.8, -0.8, Z(args.c_max) + 1.0), size=1.0,
             rgb=(0.8, 0.9, 1.0), track_target=cam)

    # Lights
    for loc, energy in [((18, -15, 22), 2000),
                        ((-10, -8, 14), 900),
                        ((0, 18, 12), 700),
                        ((5, 5, -8), 400)]:
        light_data = bpy.data.lights.new("L", type="POINT")
        light_data.energy = energy
        light = bpy.data.objects.new("L", light_data)
        light.location = loc
        scene.collection.objects.link(light)

    scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)
    print(f"[qa_pyth_fib] wrote {args.out}")


if __name__ == "__main__":
    main()
