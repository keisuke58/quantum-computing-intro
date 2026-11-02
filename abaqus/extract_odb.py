# -*- coding: utf-8 -*-
# Extract results from an Abaqus odb to CSV (run with the Abaqus python):
#   abaqus python abaqus/extract_odb.py tsv_axisym_clamped.odb
# Writes <name>_stress.csv (element centroid coordinates, stress components, von Mises)
#    and <name>_disp.csv   (node coordinates and displacement).
# axisym: x = r, y = z, S11 = rr, S22 = zz, S33 = theta-theta, S12 = rz.  3d: x, y, z and S11..S23.
import sys
from odbAccess import openOdb
from abaqusConstants import CENTROID

path = sys.argv[-1]
base = path[:-4] if path.endswith(".odb") else path
odb = openOdb(path, readOnly=True)
inst = list(odb.rootAssembly.instances.values())[0]
xyz = dict((n.label, n.coordinates) for n in inst.nodes)
conn = dict((e.label, e.connectivity) for e in inst.elements)
frame = odb.steps["cool"].frames[-1]

U = frame.fieldOutputs["U"]
dim = len(U.values[0].data)      # 2 for axisym (node coordinates still have 3 entries), 3 for 3d
S = frame.fieldOutputs["S"].getSubset(position=CENTROID)
with open(base + "_stress.csv", "w") as f:
    ncomp = len(S.values[0].data)
    head = ["elem"] + ["x", "y", "z"][:dim] + ["S11", "S22", "S33", "S12", "S13", "S23"][:ncomp] + ["mises"]
    f.write(",".join(head) + "\n")
    for v in S.values:
        nodes = conn[v.elementLabel]
        c = [sum(xyz[n][k] for n in nodes) / len(nodes) for k in range(dim)]
        f.write(",".join([str(v.elementLabel)] + ["%.6e" % x for x in c] +
                         ["%.6e" % x for x in v.data] + ["%.6e" % v.mises]) + "\n")

with open(base + "_disp.csv", "w") as f:
    f.write(",".join(["node"] + ["x", "y", "z"][:dim] + ["U1", "U2", "U3"][:dim]) + "\n")
    for v in U.values:
        f.write(",".join([str(v.nodeLabel)] + ["%.6e" % x for x in xyz[v.nodeLabel][:dim]] +
                         ["%.6e" % x for x in v.data[:dim]]) + "\n")
odb.close()
print("wrote %s_stress.csv, %s_disp.csv" % (base, base))
