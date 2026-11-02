# -*- coding: utf-8 -*-
"""
44 - TSV の 3D 単位セル: 量子リザバー特徴量 vs 古典特徴量

寸法は MORE-Stress（arXiv:2411.12690）と同じ．長さの単位はビア半径 a = 2.5 µm:
  ピッチ 15 µm → 正方セルの 1/4 は x, y ∈ [0, 3]，高さ 50 µm の上半分 z ∈ [0, 10]（z = 0 が中央面）
  ΔT = −250 K，材料は 30 と同じ Cu / Si（応力の単位 100 GPa，熱膨張率の単位 1e-3）
境界条件（abaqus/tsv_abaqus.py の mode=3d と同じ）:
  x = 0, 3: u = 0, σ_xy = σ_xz = 0 ／ y = 0, 3: v = 0, σ_xy = σ_yz = 0 ／ z = 0: w = 0, σ_xz = σ_yz = 0
  z = 10: "clamped" = u = v = w = 0，"free" = σ_xz = σ_yz = σ_zz = 0
  界面 x² + y² = 1: 変位の連続，表面力の連続（熱膨張差による跳び）

参照解:
  --ref abaqus  : fifa の Abaqus の結果 results/abaqus/tsv_3d_<bc>_{stress,disp}.csv.gz（µm, MPa を換算）
  --ref plane   : 確認用（clamped のみ）．中央面 z = 0 を 34 の平面ひずみ FE（L = 6）と比べる近似
評価: 変位の相対 L2 誤差，von Mises の規格化平均絶対誤差（MORE-Stress と同じ）を中央面付近と表面付近で
"""
import argparse, gzip, importlib.util, json, os, time
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_h = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, os.path.join(_h, fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


T = _load("t30", "30_qerc_tsv_single.py")
DT = T.DT
MAT = {0: (T.LAM1, T.MU1, T.BET1), 1: (T.LAM2, T.MU2, T.BET2)}     # 0: Cu, 1: Si
HP, HH = 3.0, 10.0                     # 半ピッチ，半高さ（a = 1）
SC = np.array([HP, HP, HH])            # 特徴量の入力は (x/3, y/3, z/10) ∈ [0, 1]
LEN_UM, U_UNIT_UM, S_UNIT_MPA = 2.5, 2.5e-3, 100.0   # 換算: x[µm] = 2.5 x，u[µm] = 2.5e-3 u，σ[MPa] = 100 σ


# ==============================================================
# 1. 3 入力の特徴量（30 / 31 の 2 入力版をそのまま 3 入力にしたもの）
# ==============================================================
def make_qerc(N, seed, scale):
    g = np.random.default_rng(1000 + seed)
    W = jnp.asarray(g.normal(size=(3, 2 * N)) * scale); b = jnp.asarray(g.normal(size=2 * N))
    z = (g.normal(size=(2**N,) * 2) + 1j * g.normal(size=(2**N,) * 2)) / np.sqrt(2)
    q, r = np.linalg.qr(z); U = jnp.asarray(q * (np.diag(r) / np.abs(np.diag(r))))   # Haar

    def feat(p):
        ang = jnp.pi * (jnp.tanh(p @ W + b) + 1) / 2
        psi = jnp.ones((1,), complex)
        for l in range(N):
            psi = jnp.kron(psi, jnp.stack([jnp.cos(ang[l] / 2), jnp.exp(1j * ang[N + l]) * jnp.sin(ang[l] / 2)]))
        return jnp.abs(U @ psi) ** 2
    return feat


def make_mlp2(N, seed, scale):
    g = np.random.default_rng(3000 + seed)
    W1 = jnp.asarray(g.normal(size=(3, 2 * N)) * scale); b1 = jnp.asarray(g.normal(size=2 * N))
    W2 = jnp.asarray(g.normal(size=(2 * N, 2**N)) * 2 / np.sqrt(2 * N)); b2 = jnp.asarray(g.normal(size=2**N))
    return lambda p: jnp.tanh(jnp.tanh(p @ W1 + b1) @ W2 + b2)


def make_tanh(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(3, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda p: jnp.tanh(p @ W + b)


def make_fourier(m, seed, scale):
    g = np.random.default_rng(seed)
    W = jnp.asarray(g.normal(size=(3, m)) * scale); b = jnp.asarray(g.random(m) * 2 * np.pi)
    return lambda p: jnp.cos(p @ W + b)


def derivs(feat, chunk=2048):
    """座標 x (n, 3) → (値 (n, m), 1 階 (n, m, 3), 2 階 (n, m, 3, 3))．入力の規格化を含む"""
    f = lambda x: feat(x / jnp.asarray(SC))
    fs = [jax.jit(jax.vmap(f)), jax.jit(jax.vmap(jax.jacfwd(f))), jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(f))))]

    def get(P, order):
        P = np.asarray(P, float)
        return np.concatenate([np.asarray(fs[order](jnp.asarray(P[i:i + chunk]))) for i in range(0, len(P), chunk)])
    return get


# ==============================================================
# 2. 最小二乗
# ==============================================================
def collocation(n_cu, n_si, n_if, n_face, seed):
    g = np.random.default_rng(seed)
    r = np.sqrt(g.random(n_cu)); t = g.random(n_cu) * np.pi / 2
    p_cu = np.stack([r * np.cos(t), r * np.sin(t), HH * g.random(n_cu)], 1)
    q = g.random((3 * n_si, 3)) * SC
    p_si = q[np.hypot(q[:, 0], q[:, 1]) > 1.0][:n_si]
    t = g.random(n_if) * np.pi / 2
    p_if = np.stack([np.cos(t), np.sin(t), HH * g.random(n_if)], 1)
    faces = {}
    for name, (k, val) in {"x0": (0, 0.0), "x1": (0, HP), "y0": (1, 0.0), "y1": (1, HP),
                           "z0": (2, 0.0), "z1": (2, HH)}.items():
        P = g.random((n_face, 3)) * SC; P[:, k] = val; faces[name] = P
    return p_cu, p_si, p_if, faces


def stress_blocks(J, lam, mu):
    """J (n, m, 3) → σ_ij の行（[c_u | c_v | c_w] に対する (n, 3m)），熱の項は含まない"""
    n, m, _ = J.shape
    def blk(comp, d):            # ∂u_comp/∂x_d の行
        B = np.zeros((n, 3 * m)); B[:, comp * m:(comp + 1) * m] = J[:, :, d]; return B
    div = blk(0, 0) + blk(1, 1) + blk(2, 2)
    S = {}
    for i in range(3):
        for j in range(i, 3):
            S[i, j] = mu * (blk(i, j) + blk(j, i)) + (lam * div if i == j else 0)
    return S


def assemble(d, bc, pts):
    p_cu, p_si, p_if, faces = pts
    m = d[0](p_cu[:1], 0).shape[1]
    nb = 3 * m; ntot = 2 * nb
    rows, rhs = [], []

    def add(ph, blk, b, w, blk2=None):
        R = np.zeros((len(b), ntot))
        if blk2 is None:
            R[:, ph * nb:(ph + 1) * nb] = blk
        else:
            R[:, :nb] = blk; R[:, nb:] = blk2
        rows.append(w * R); rhs.append(w * np.asarray(b, float))

    # Navier: μ Δu_i + (λ+μ) ∂_i (∇·u) = 0
    for ph, P in ((0, p_cu), (1, p_si)):
        lam, mu, _ = MAT[ph]; H = d[ph](P, 2); w = 1 / np.sqrt(len(P))
        lap = H[:, :, 0, 0] + H[:, :, 1, 1] + H[:, :, 2, 2]
        for i in range(3):
            R = np.zeros((len(P), nb))
            R[:, i * m:(i + 1) * m] += mu * lap
            for j in range(3):
                R[:, j * m:(j + 1) * m] += (lam + mu) * H[:, :, j, i]
            add(ph, R, np.zeros(len(P)), w)
    # 界面
    w = 1 / np.sqrt(len(p_if)); nrm = np.c_[p_if[:, 0], p_if[:, 1], 0 * p_if[:, 0]]
    F1, F2 = d[0](p_if, 0), d[1](p_if, 0)
    S1, S2 = stress_blocks(d[0](p_if, 1), *MAT[0][:2]), stress_blocks(d[1](p_if, 1), *MAT[1][:2])
    jump = (MAT[0][2] - MAT[1][2]) * DT
    for i in range(3):
        B1 = np.zeros((len(p_if), nb)); B1[:, i * m:(i + 1) * m] = F1
        B2 = np.zeros((len(p_if), nb)); B2[:, i * m:(i + 1) * m] = F2
        add(None, B1, np.zeros(len(p_if)), w, -B2)
        t1 = sum(S1[min(i, j), max(i, j)] * nrm[:, j:j + 1] for j in range(3))
        t2 = sum(S2[min(i, j), max(i, j)] * nrm[:, j:j + 1] for j in range(3))
        add(None, t1, jump * nrm[:, i], w, -t2)
    # 外側の面
    for name, P in faces.items():
        k = {"x": 0, "y": 1, "z": 2}[name[0]]
        for ph in (0, 1):
            mask = (np.hypot(P[:, 0], P[:, 1]) < 1.0) == (ph == 0)
            if not mask.any():
                continue
            Q = P[mask]; w = 1 / np.sqrt(len(P)); lam, mu, bet = MAT[ph]
            F = d[ph](Q, 0); S = stress_blocks(d[ph](Q, 1), lam, mu)
            disp = lambda c: np.concatenate([F if c == cc else np.zeros_like(F) for cc in range(3)], 1)
            if name == "z1" and bc == "clamped":
                for c in range(3):
                    add(ph, disp(c), np.zeros(len(Q)), w)
                continue
            if name == "z1":             # 上面自由
                add(ph, S[2, 2], np.full(len(Q), bet * DT), w)
            else:                        # 対称面: 法線方向の変位 0
                add(ph, disp(k), np.zeros(len(Q)), w)
            for j in range(3):           # せん断の表面力 0
                if j != k:
                    add(ph, S[min(j, k), max(j, k)], np.zeros(len(Q)), w)
    return np.concatenate(rows), np.concatenate(rhs), m


def predict(d, c, m, P):
    nb = 3 * m
    phase = (np.hypot(P[:, 0], P[:, 1]) >= 1.0).astype(int)
    U = np.zeros((len(P), 3)); S = np.zeros((len(P), 6))
    for ph in (0, 1):
        mask = phase == ph
        if not mask.any():
            continue
        Q = P[mask]; lam, mu, bet = MAT[ph]; cp = c[ph * nb:(ph + 1) * nb]
        F = d[ph](Q, 0)
        U[mask] = np.stack([F @ cp[i * m:(i + 1) * m] for i in range(3)], 1)
        Sb = stress_blocks(d[ph](Q, 1), lam, mu)
        for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
            S[mask, k] = Sb[i, j] @ cp - (bet * DT if i == j else 0)
    return U, S


def von_mises(S):
    xx, yy, zz, xy, xz, yz = S.T
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + xz ** 2 + yz ** 2))


# ==============================================================
# 3. 参照解
# ==============================================================
def _csv(path):
    with gzip.open(path, "rt") as f:
        head = f.readline().strip().split(",")
        data = np.loadtxt(f, delimiter=",")
    return {h: data[:, i] for i, h in enumerate(head)}


def ref_abaqus(bc):
    base = os.path.join(_h, "results", "abaqus", f"tsv_3d_{bc}")
    s, u = _csv(base + "_stress.csv.gz"), _csv(base + "_disp.csv.gz")
    cen = np.stack([s["x"], s["y"], s["z"]], 1) / LEN_UM
    S = np.stack([s[k] for k in ("S11", "S22", "S33", "S12", "S13", "S23")], 1) / S_UNIT_MPA
    nodes = np.stack([u["x"], u["y"], u["z"]], 1) / LEN_UM
    U = np.stack([u["U1"], u["U2"], u["U3"]], 1) / U_UNIT_UM
    return {"nodes": nodes, "U": U, "cen": cen, "S": S}


def ref_plane(h=0.01):
    """確認用: 34 の平面ひずみ FE（L = 6）を中央面 z = 0 に置く（clamped で高さが十分なら近い）"""
    P34 = _load("p34", "34_tsv_periodic.py")
    saved = P34.DT
    fe = P34.fe_solve(2 * HP, h)
    assert saved == DT
    nodes = np.c_[fe["pts"], np.zeros(len(fe["pts"]))]
    U = np.c_[fe["u"], np.zeros(len(fe["pts"]))]
    cen = np.c_[fe["cen"], np.zeros(len(fe["cen"]))]
    sxx, syy, sxy = fe["S"].T
    S = np.zeros((len(cen), 6)); S[:, 0], S[:, 1], S[:, 3] = sxx, syy, sxy
    for ph in (0, 1):                    # 平面ひずみの σ_zz = λ(ε_xx + ε_yy) − βΔT
        lam, mu, bet = MAT[ph]; mk = fe["phase"] == ph
        tr = (sxx[mk] + syy[mk] + 2 * bet * DT) / (2 * (lam + mu))
        S[mk, 2] = lam * tr - bet * DT
    return {"nodes": nodes, "U": U, "cen": cen, "S": S}


def evaluate(d, c, m, ref, n_eval=20000):
    g = np.random.default_rng(0)
    iu = g.choice(len(ref["nodes"]), min(n_eval, len(ref["nodes"])), replace=False)
    U, _ = predict(d, c, m, ref["nodes"][iu])
    out = {"u": float(np.linalg.norm(U - ref["U"][iu]) / np.linalg.norm(ref["U"][iu]))}
    z = ref["cen"][:, 2]
    for key, band in (("mid", z < 0.5), ("top", (z > HH - 1.0) & (z < HH - 0.25))):
        idx = np.where(band)[0]
        if len(idx) == 0:
            continue
        idx = g.choice(idx, min(n_eval, len(idx)), replace=False)
        _, S = predict(d, c, m, ref["cen"][idx])
        vm, vr = von_mises(S), von_mises(ref["S"][idx])
        out["vm_" + key] = float(np.mean(np.abs(vm - vr)) / vr.max())
    return out


# ==============================================================
# 4. 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="clamped,free")
    ap.add_argument("--ref", default="abaqus", choices=["abaqus", "plane"])
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", default="QERC-Haar,MLP2-RF,tanh-RF,Fourier")
    ap.add_argument("--n_cu", type=int, default=4000)
    ap.add_argument("--n_si", type=int, default=12000)
    ap.add_argument("--out", default="results/tsv_3d.json")
    a = ap.parse_args()
    m = 2 ** a.N
    grid = {"QERC-Haar": ([1.0, 2.0, 4.0], lambda sd, s: make_qerc(a.N, sd, s)),
            "MLP2-RF": ([1.0, 2.0, 4.0], lambda sd, s: make_mlp2(a.N, sd, s)),
            "tanh-RF": ([2.0, 4.0, 8.0], lambda sd, s: make_tanh(m, sd, s)),
            "Fourier": ([2.0, 4.0, 8.0, 16.0], lambda sd, s: make_fourier(m, sd, s))}
    grid = {k: v for k, v in grid.items() if k in a.models.split(",")}
    rows = []
    for bc in a.bc.split(","):
        if a.ref == "plane" and bc != "clamped":
            continue
        ref = ref_abaqus(bc) if a.ref == "abaqus" else ref_plane()
        print(f"bc={bc} ref={a.ref}: 節点 {len(ref['nodes'])}，要素 {len(ref['cen'])}", flush=True)
        for sd in range(a.seeds):
            pts = collocation(a.n_cu, a.n_si, 1500, 800, 10 + sd)
            for name, (scales, ctor) in grid.items():
                for s in scales:
                    t0 = time.time()
                    d = {0: derivs(ctor(2 * sd, s)), 1: derivs(ctor(2 * sd + 1, s))}
                    A, b, mm = assemble(d, bc, pts)
                    c = T.solve(A, b, 1e-10)
                    res = evaluate(d, c, mm, ref)
                    rows.append({"bc": bc, "ref": a.ref, "model": name, "seed": sd, "scale": s, **res})
                    print(f"bc={bc} seed={sd} {name:10s} ×{s:<4} " +
                          " ".join(f"{k}={v:.2e}" for k, v in res.items()) + f" {time.time()-t0:.0f}s", flush=True)
                    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n=== シードごとに u 最良の倍率（幾何平均）===")
    for bc in dict.fromkeys(r["bc"] for r in rows):
        for name in grid:
            best = []
            for sd in range(a.seeds):
                rr = [r for r in rows if r["bc"] == bc and r["model"] == name and r["seed"] == sd]
                if rr:
                    best.append(min(rr, key=lambda r: r["u"]))
            if best:
                keys = [k for k in best[0] if k in ("u", "vm_mid", "vm_top")]
                print(f"bc={bc:8s} {name:10s} " + "  ".join(
                    f"{k} {np.exp(np.mean(np.log([b[k] for b in best]))):.2e}" for k in keys))


if __name__ == "__main__":
    main()
