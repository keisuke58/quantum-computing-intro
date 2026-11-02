# -*- coding: utf-8 -*-
# Abaqus/CAE script: TSV thermal stress reference (MORE-Stress dimensions).
# Run on fifa (no GUI):
#   abaqus cae noGUI=abaqus/tsv_abaqus.py -- mode=axisym -- bc=clamped -- size=0.05 -- ncpu=4
#   abaqus cae noGUI=abaqus/tsv_abaqus.py -- mode=3d -- bc=free -- size=0.4 -- ncpu=8
#   (put "--" before every argument; this is what worked on fifa)
# Units: um, MPa (force uN). Cu via radius 2.5 um, height 50 um, pitch 15 um, dT = -250 K (275 C -> 25 C).
# mode=axisym : one via in a cylinder cell of the same area (R = 15/sqrt(pi)); same model as 42_tsv_axisym.py
# mode=3d     : quarter of the square unit cell (x, y in [0, 7.5]); same as MORE-Stress unit block
# Only the upper half (z in [0, 25]) is modelled: z = 0 is the mid-plane (symmetry).
# bc=clamped : top face fixed (MORE-Stress scenario 1);  bc=free : top face traction free (wafer surface)
# The liner (0.5 um dielectric) is omitted to match our PINN models.
from abaqus import *
from abaqusConstants import *
from caeModules import *      # needed in noGUI mode, otherwise the model has no fieldOutputRequests
import regionToolset
import mesh
import sys
import math

args = dict(a.split("=") for a in sys.argv if "=" in a and not a.startswith("-"))
MODE = args.get("mode", "axisym")
BC = args.get("bc", "clamped")
SIZE = float(args.get("size", "0.1" if MODE == "axisym" else "0.4"))
NCPU = int(args.get("ncpu", "4"))
LINER = float(args.get("liner", "0"))   # SiO2 liner thickness [um], axisym only (0 = none)

A = 2.5                    # via radius
HH = 25.0                  # half height
P2 = 7.5                   # half pitch (3d)
RC = 15.0 / math.sqrt(math.pi)   # equal-area cylinder cell (axisym)
T0, T1 = 275.0, 25.0
CU = (110000.0, 0.35, 17.0e-6)   # E [MPa], nu, alpha [1/K]  (same as 30_qerc_tsv_single.py)
SI = (130000.0, 0.28, 2.6e-6)
OX = (70000.0, 0.16, 0.5e-6)     # SiO2 liner
TOL = 1e-4

name = "tsv_%s_%s" % (MODE, BC) + ("_liner%g" % LINER if LINER > 0 else "")
m = mdb.Model(name=name)
for key, (E, nu, al) in (("Cu", CU), ("Si", SI), ("SiO2", OX)):
    mat = m.Material(name=key)
    mat.Elastic(table=((E, nu),))
    mat.Expansion(table=((al,),))
    m.HomogeneousSolidSection(name=key, material=key, thickness=None)

asm = m.rootAssembly
if MODE == "axisym":
    def rect_part(pname, r0, r1):
        s = m.ConstrainedSketch(name=pname, sheetSize=100.0)
        s.sketchOptions.setValues(viewStyle=AXISYM)
        axis = s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
        s.FixedConstraint(entity=axis)
        s.rectangle(point1=(r0, 0.0), point2=(r1, HH))
        p = m.Part(name=pname, dimensionality=AXISYMMETRIC, type=DEFORMABLE_BODY)
        p.BaseShell(sketch=s)
        return p
    pcu = rect_part("cu", 0.0, A)
    psi = rect_part("si", A + LINER, RC)
    pox = rect_part("ox", A, A + LINER) if LINER > 0 else None
else:
    s = m.ConstrainedSketch(name="si", sheetSize=100.0)
    s.rectangle(point1=(0.0, 0.0), point2=(P2, P2))
    psi = m.Part(name="si", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    psi.BaseSolidExtrude(sketch=s, depth=HH)
    s = m.ConstrainedSketch(name="cu", sheetSize=100.0)
    s.ArcByCenterEnds(center=(0.0, 0.0), point1=(A, 0.0), point2=(0.0, A), direction=COUNTERCLOCKWISE)
    s.Line(point1=(0.0, 0.0), point2=(A, 0.0))
    s.Line(point1=(0.0, 0.0), point2=(0.0, A))
    pcu = m.Part(name="cu", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    pcu.BaseSolidExtrude(sketch=s, depth=HH)

asm.Instance(name="si", part=psi, dependent=ON)
asm.Instance(name="cu", part=pcu, dependent=ON)
parts = [asm.instances["si"], asm.instances["cu"]]
if MODE == "axisym" and LINER > 0:
    asm.Instance(name="ox", part=pox, dependent=ON)
    parts.append(asm.instances["ox"])
asm.InstanceFromBooleanMerge(name="tsv", instances=tuple(parts),
                             keepIntersections=ON, originalInstances=DELETE, domain=GEOMETRY)
p = m.parts["tsv"]
inst = asm.instances["tsv-1"]

# sections: pick the region by a point inside it
if MODE == "axisym":
    cu_reg = p.faces.findAt(((0.5 * A, 0.5 * HH, 0.0),))
    si_reg = p.faces.findAt(((0.5 * (A + LINER + RC), 0.5 * HH, 0.0),))
    p.SectionAssignment(region=regionToolset.Region(faces=cu_reg), sectionName="Cu")
    p.SectionAssignment(region=regionToolset.Region(faces=si_reg), sectionName="Si")
    if LINER > 0:
        ox_reg = p.faces.findAt(((A + 0.5 * LINER, 0.5 * HH, 0.0),))
        p.SectionAssignment(region=regionToolset.Region(faces=ox_reg), sectionName="SiO2")
    p.setMeshControls(regions=p.faces, elemShape=QUAD, technique=STRUCTURED)
    p.setElementType(regions=(p.faces,), elemTypes=(mesh.ElemType(elemCode=CAX8, elemLibrary=STANDARD),))
else:
    cu_reg = p.cells.findAt(((0.5 * A, 0.5 * A, 0.5 * HH),))
    si_reg = p.cells.findAt(((0.9 * P2, 0.9 * P2, 0.5 * HH),))
    p.SectionAssignment(region=regionToolset.Region(cells=cu_reg), sectionName="Cu")
    p.SectionAssignment(region=regionToolset.Region(cells=si_reg), sectionName="Si")
    p.setMeshControls(regions=p.cells, elemShape=TET, technique=FREE)
    p.setElementType(regions=(p.cells,), elemTypes=(mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD),))
p.seedPart(size=SIZE, deviationFactor=0.1, minSizeFactor=0.1)
p.generateMesh()

m.StaticStep(name="cool", previous="Initial")
if "F-Output-1" in m.fieldOutputRequests.keys():
    m.fieldOutputRequests["F-Output-1"].setValues(variables=("S", "U"))
else:
    m.FieldOutputRequest(name="F-Output-1", createStepName="cool", variables=("S", "U"))

if MODE == "axisym":
    allreg = regionToolset.Region(faces=inst.faces)
else:
    allreg = regionToolset.Region(cells=inst.cells)
m.Temperature(name="T", createStepName="Initial", region=allreg, distributionType=UNIFORM,
              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(T0,))
m.predefinedFields["T"].setValuesInStep(stepName="cool", magnitudes=(T1,))


def side(**bb):
    """faces (3d) or edges (axisym) inside a bounding box"""
    box = dict(xMin=-1e3, xMax=1e3, yMin=-1e3, yMax=1e3, zMin=-1e3, zMax=1e3)
    box.update(bb)
    if MODE == "axisym":
        return regionToolset.Region(edges=inst.edges.getByBoundingBox(**box))
    return regionToolset.Region(faces=inst.faces.getByBoundingBox(**box))


if MODE == "axisym":
    # axis r = 0: u_r = 0 ; outer r = RC: u_r = 0 ; mid-plane z = 0: u_z = 0
    m.DisplacementBC(name="axis", createStepName="Initial", region=side(xMin=-TOL, xMax=TOL), u1=0.0)
    m.DisplacementBC(name="outer", createStepName="Initial", region=side(xMin=RC - TOL, xMax=RC + TOL), u1=0.0)
    m.DisplacementBC(name="mid", createStepName="Initial", region=side(yMin=-TOL, yMax=TOL), u2=0.0)
    if BC == "clamped":
        m.DisplacementBC(name="top", createStepName="Initial", region=side(yMin=HH - TOL, yMax=HH + TOL), u1=0.0, u2=0.0)
else:
    # symmetry planes x = 0, x = P2 (u_x = 0), y = 0, y = P2 (u_y = 0), mid-plane z = 0 (u_z = 0)
    for nm, bb, kw in (("x0", dict(xMin=-TOL, xMax=TOL), dict(u1=0.0)),
                       ("x1", dict(xMin=P2 - TOL, xMax=P2 + TOL), dict(u1=0.0)),
                       ("y0", dict(yMin=-TOL, yMax=TOL), dict(u2=0.0)),
                       ("y1", dict(yMin=P2 - TOL, yMax=P2 + TOL), dict(u2=0.0)),
                       ("mid", dict(zMin=-TOL, zMax=TOL), dict(u3=0.0))):
        m.DisplacementBC(name=nm, createStepName="Initial", region=side(**bb), **kw)
    if BC == "clamped":
        m.DisplacementBC(name="top", createStepName="Initial", region=side(zMin=HH - TOL, zMax=HH + TOL),
                         u1=0.0, u2=0.0, u3=0.0)

job = mdb.Job(name=name, model=name, numCpus=NCPU, numDomains=NCPU)
mdb.saveAs(pathName=name + ".cae")
job.submit()
job.waitForCompletion()
print("done: %s.odb  (elements: %d)" % (name, len(p.elements)))
