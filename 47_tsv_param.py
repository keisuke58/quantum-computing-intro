# -*- coding: utf-8 -*-
"""
47 - TSV の熱応力のパラメータスタディ（軸対称 FE，µm と MPa）

42 と同じ軸対称 P1 FE を，寸法を変えられる形で書き直したもの（Abaqus と一致を確認済みの定式化）．
  ビア直径 d，ピッチ p（同じ面積の円柱セル R = p/√π），高さ 50 µm，絶縁膜（ライナー）の厚さ t
  材料: Cu（E 110 GPa，ν 0.35，α 17 ppm/K），Si（E 130 GPa，ν 0.28，α 2.6 ppm/K），
        ライナー SiO2（E 70 GPa，ν 0.16，α 0.5 ppm/K）
  ΔT = −250 K，高さの中央で上下対称，上面は clamped / free
出力: Si 側の界面の von Mises と，von Mises が 300 MPa を下回るまでの界面からの距離（KOZ の目安）を
      中央面付近（z < 0.5 µm）と表面のすぐ下（上面から 0.25〜0.5 µm）で．
"""
import argparse, json, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MATS = {"Cu": (110000.0, 0.35, 17e-6), "Si": (130000.0, 0.28, 2.6e-6), "SiO2": (70000.0, 0.16, 0.5e-6)}
DT, HEIGHT, THRESH = -250.0, 50.0, 300.0


def lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def axis_nodes(segments):
    """[(端点, 刻み)...] から，各境界に節点がそろった 1 次元の格子"""
    xs = [0.0]
    for x1, h in segments:
        n = max(int(np.ceil((x1 - xs[-1]) / h)), 1)
        xs += list(np.linspace(xs[-1], x1, n + 1)[1:])
    return np.array(xs)


def solve(d, p, t, bc, h):
    a = d / 2; R = p / np.sqrt(np.pi); H = HEIGHT / 2
    seg = [(a, h)] + ([(a + t, min(h, t / 4))] if t > 0 else []) + [(R, h)]
    r = axis_nodes(seg); z = axis_nodes([(H, h)])
    nr, nz = len(r), len(z)
    Rg, Zg = np.meshgrid(r, z, indexing="ij"); pts = np.stack([Rg.ravel(), Zg.ravel()], 1)
    I, J = np.meshgrid(np.arange(nr - 1), np.arange(nz - 1), indexing="ij"); I, J = I.ravel(), J.ravel()
    n00, n10, n11, n01 = I * nz + J, (I + 1) * nz + J, (I + 1) * nz + J + 1, I * nz + J + 1
    tri = np.concatenate([np.stack([n00, n10, n11], 1), np.stack([n00, n11, n01], 1)])
    cen = pts[tri].mean(1)
    mat = np.where(cen[:, 0] < a, 0, np.where(cen[:, 0] < a + t, 1, 2))      # 0 Cu, 1 ライナー, 2 Si
    props = [MATS["Cu"], MATS["SiO2"], MATS["Si"]]
    # 要素ごとの B 行列（ベクトル化）
    P = pts[tri]; r1, z1 = P[:, 0, 0], P[:, 0, 1]; r2, z2 = P[:, 1, 0], P[:, 1, 1]; r3, z3 = P[:, 2, 0], P[:, 2, 1]
    det = (r2 - r1) * (z3 - z1) - (r3 - r1) * (z2 - z1)
    b = np.stack([z2 - z3, z3 - z1, z1 - z2], 1) / det[:, None]
    c = np.stack([r3 - r2, r1 - r3, r2 - r1], 1) / det[:, None]
    rc = cen[:, 0]; area = 0.5 * np.abs(det); w = 2 * np.pi * rc * area
    ne = len(tri); B = np.zeros((ne, 4, 6))
    B[:, 0, 0::2] = b; B[:, 1, 0::2] = (1 / (3 * rc))[:, None]; B[:, 2, 1::2] = c; B[:, 3, 0::2] = c; B[:, 3, 1::2] = b
    D = np.zeros((ne, 4, 4)); eth = np.zeros((ne, 4))
    for k, (E, nu, al) in enumerate(props):
        lam, mu = lame(E, nu); m = mat == k
        D[m] = np.array([[lam + 2 * mu, lam, lam, 0], [lam, lam + 2 * mu, lam, 0], [lam, lam, lam + 2 * mu, 0], [0, 0, 0, mu]])
        eth[m] = al * DT * np.array([1.0, 1.0, 1.0, 0.0])
    dofs = np.empty((ne, 6), int); dofs[:, 0::2] = 2 * tri; dofs[:, 1::2] = 2 * tri + 1
    Ke = w[:, None, None] * np.einsum("eki,ekl,elj->eij", B, D, B)
    fe = w[:, None] * np.einsum("eki,ekl,el->ei", B, D, eth)
    ndof = 2 * len(pts)
    K = sp.csr_matrix((Ke.ravel(), (np.repeat(dofs, 6, 1).ravel(), np.tile(dofs, (1, 6)).ravel())), shape=(ndof, ndof))
    F = np.zeros(ndof); np.add.at(F, dofs.ravel(), fe.ravel())
    tol = 1e-9; fixed = np.zeros(ndof, bool)
    fixed[2 * np.where(pts[:, 0] < tol)[0]] = True
    fixed[2 * np.where(np.abs(pts[:, 0] - R) < tol)[0]] = True
    fixed[2 * np.where(pts[:, 1] < tol)[0] + 1] = True
    if bc == "clamped":
        top = np.where(np.abs(pts[:, 1] - H) < tol)[0]; fixed[2 * top] = True; fixed[2 * top + 1] = True
    U = np.zeros(ndof); free = ~fixed
    U[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    eps = np.einsum("eij,ej->ei", B, U[dofs])
    S = np.einsum("eij,ej->ei", D, eps - eth)                                 # [rr, θθ, zz, rz]
    rr, tt, zz, rz = S.T
    vm = np.sqrt(0.5 * ((rr - tt) ** 2 + (tt - zz) ** 2 + (zz - rr) ** 2) + 3 * rz ** 2)
    return {"cen": cen, "mat": mat, "vm": vm, "a": a, "t": t, "R": R, "H": H}


def koz(res, band):
    """Si の要素を r の区間に分けて平均し，界面の値と 300 MPa を下回る距離（界面から）を返す"""
    a0 = res["a"] + res["t"]; m = band & (res["mat"] == 2)
    rr, vv = res["cen"][m, 0], res["vm"][m]
    edges = np.arange(a0, res["R"] + 1e-9, 0.25); c = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array([vv[(rr >= e0) & (rr < e1)].mean() if ((rr >= e0) & (rr < e1)).any() else np.nan
                     for e0, e1 in zip(edges[:-1], edges[1:])])
    below = np.where(prof < THRESH)[0]
    return {"vm_if": float(prof[0]), "koz": float(c[below[0]] - a0) if len(below) else float("nan"),
            "vm_far": float(prof[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--out", default="results/tsv_param.json")
    a = ap.parse_args()
    cases = []
    for p in (10.0, 15.0, 20.0, 30.0):
        cases.append(("pitch", 5.0, p, 0.0))
    for d in (3.0, 10.0):
        cases.append(("diam", d, 3 * d, 0.0))
    cases.append(("liner", 5.0, 15.0, 0.5))
    rows = []
    for kind, d, p, t in cases:
        for bc in ("clamped", "free"):
            t0 = time.time(); res = solve(d, p, t, bc, a.h)
            out = {"kind": kind, "d": d, "p": p, "t": t, "bc": bc}
            for key, band in (("mid", res["cen"][:, 1] < 0.5),
                              ("top", (res["cen"][:, 1] > res["H"] - 0.5) & (res["cen"][:, 1] < res["H"] - 0.25))):
                for k, v in koz(res, band).items():
                    out[f"{key}_{k}"] = v
            rows.append(out)
            print(f"{kind:5s} d={d:4.1f} p={p:4.1f} t={t:3.1f} {bc:7s}  中央面: 界面 {out['mid_vm_if']:5.0f} MPa, "
                  f"KOZ {out['mid_koz']:.2f} µm ／ 表面直下: 界面 {out['top_vm_if']:5.0f} MPa, KOZ {out['top_koz']:.2f} µm "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
