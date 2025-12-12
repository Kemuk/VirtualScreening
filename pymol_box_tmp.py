from pymol import cmd, stored
import sys
rec, lig, pad, min_size, max_size = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cmd.reinitialize()
cmd.load(rec, "prot")
if lig:
    cmd.load(lig, "lig")
    sel = "lig and not hydro"
    stored.m = []
    cmd.iterate(sel, "stored.m.append(1)")
    if stored.m:
        cx, cy, cz = cmd.centerofmass(sel)
        xs, ys, zs = [], [], []
        for a in cmd.get_model(sel).atom:
            xs.append(a.coord[0]); ys.append(a.coord[1]); zs.append(a.coord[2])
        clamp = lambda v: max(min_size, min(max_size, v))
        sx = clamp((max(xs)-min(xs)) + 2*pad)
        sy = clamp((max(ys)-min(ys)) + 2*pad)
        sz = clamp((max(zs)-min(zs)) + 2*pad)
        print(f"{cx:.3f},{cy:.3f},{cz:.3f},{sx:.3f},{sy:.3f},{sz:.3f}")
        sys.exit(0)
cx, cy, cz = cmd.centerofmass("prot and not hydro")
print(f"{cx:.3f},{cy:.3f},{cz:.3f},22.000,22.000,22.000")
