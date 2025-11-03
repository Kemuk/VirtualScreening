from pymol import cmd
import sys

rec, lig, pad, min_size, max_size = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
cmd.reinitialize()
cmd.load(rec, "prot")
cmd.load(lig, "lig")
sel = "lig and not hydro"
cx, cy, cz = cmd.centerofmass(sel)
m = cmd.get_model(sel)
xs = [a.coord[0] for a in m.atom]; ys = [a.coord[1] for a in m.atom]; zs = [a.coord[2] for a in m.atom]
if not xs:
    cx, cy, cz = cmd.centerofmass("prot and not hydro"); sx = sy = sz = 22.0
else:
    clamp = lambda v: max(min_size, min(max_size, v))
    sx = clamp((max(xs) - min(xs)) + 2.0*pad)
    sy = clamp((max(ys) - min(ys)) + 2.0*pad)
    sz = clamp((max(zs) - min(zs)) + 2.0*pad)
print(f"{cx:.3f},{cy:.3f},{cz:.3f},{sx:.3f},{sy:.3f},{sz:.3f}")
