# -*- coding: utf-8 -*-
"""
13 - QPINN ベンチ #2: 円孔付き板の応力集中（Kirsch 解）

研究計画のベンチ #3（星形グレイン）の差し替え。応力集中という論点は維持したまま、
解析解のある問題にすることで FEM 参照解の工数と非圧縮性（ν→0.5）の体積ロッキングを回避する。

## 問題

遠方一軸引張 S を受ける円孔付き無限板の、円孔まわりの円環領域 a ≤ r ≤ R を解く。

    平面応力の Navier 方程式:  ∂σ_xx/∂x + ∂σ_xy/∂y = 0
                               ∂σ_xy/∂x + ∂σ_yy/∂y = 0
    内側 r=a : traction-free （σ_rr = σ_rθ = 0）  ← 応力集中が起きる面
    外側 r=R : 変位の Dirichlet 条件（Kirsch 解を与える）

解析解は Muskhelishvili の複素ポテンシャルから得る:

    φ(z) = S/4 (z + 2a²/z),   ψ(z) = -S/2 (z + a²/z - a⁴/z³)
    2μ(u + iv) = κ φ(z) - z conj(φ'(z)) - conj(ψ(z))

孔縁の σ_θθ は θ=90° で 3S（応力集中係数 K_t = 3）、θ=0° で -S。
`verify_exact_solution()` がこれを含む3つの検証を実行する。

## #1（Lamé）との違い

| | #1 厚肉円筒 | #2 円孔付き板（これ） |
|---|---|---|
| 次元 | 1D（軸対称） | 2D |
| 出力 | スカラー u(r) | **ベクトル場 (u, v)** ← QPINN の多出力読み出し設計が要る |
| 解の性質 | 滑らか | **応力集中**（局所的な急勾配） |
| 検証量 | 変位・応力の L2 | L2 に加えて **K_t の再現精度** |

QPINN は多出力を「観測量を分ける」方式で実装している（⟨Z₀⟩→u, ⟨Z₁⟩→v）。
これは計画 §4 が最大のリスクに挙げていた設計そのもの。
"""

import argparse
import math
import time

import numpy as np
import torch
import pennylane as qml

torch.set_default_dtype(torch.float64)

# ==============================================================
# 1. 問題設定（無次元化: a=1, S=1, E=1）
# ==============================================================
A_HOLE = 1.0        # 円孔半径
R_OUT = 4.0         # 計算領域の外半径
S_FAR = 1.0         # 遠方引張応力
E_MOD, NU = 1.0, 0.3

MU_G = E_MOD / (2 * (1 + NU))                 # せん断弾性係数
KAPPA = (3 - NU) / (1 + NU)                   # 平面応力の Kolosov 定数
C_PS = E_MOD / (1 - NU**2)                    # 平面応力の係数
LAM_PS = E_MOD * NU / (1 - NU**2)             # 平面応力の実効 λ


# --- 複素ポテンシャル（Muskhelishvili） ------------------------------
def _phi(z):
    return S_FAR / 4 * (z + 2 * A_HOLE**2 / z)


def _dphi(z):
    return S_FAR / 4 * (1 - 2 * A_HOLE**2 / z**2)


def _ddphi(z):
    return S_FAR / 4 * (4 * A_HOLE**2 / z**3)


def _psi(z):
    return -S_FAR / 2 * (z + A_HOLE**2 / z - A_HOLE**4 / z**3)


def _dpsi(z):
    return -S_FAR / 2 * (1 - A_HOLE**2 / z**2 + 3 * A_HOLE**4 / z**4)


def uv_exact(x, y):
    """解析解の変位 (u, v)。x, y は numpy 配列"""
    z = np.asarray(x) + 1j * np.asarray(y)
    w = (KAPPA * _phi(z) - z * np.conj(_dphi(z)) - np.conj(_psi(z))) / (2 * MU_G)
    return np.real(w), np.imag(w)


def stress_exact(x, y):
    """解析解の応力 (σxx, σyy, σxy)

    σxx+σyy = 4 Re φ' ,  σyy-σxx+2iσxy = 2( z̄ φ'' + ψ' )
    """
    z = np.asarray(x) + 1j * np.asarray(y)
    s1 = 4 * np.real(_dphi(z))
    s2 = 2 * (np.conj(z) * _ddphi(z) + _dpsi(z))
    sxx = (s1 - np.real(s2)) / 2
    syy = (s1 + np.real(s2)) / 2
    sxy = np.imag(s2) / 2
    return sxx, syy, sxy


def sigma_tt_hole_exact(theta):
    """孔縁 r=a の周方向応力。θ=90° で 3S、θ=0° で -S"""
    return S_FAR * (1 - 2 * np.cos(2 * np.asarray(theta)))


# ==============================================================
# 2. 解析解の検証
# ==============================================================
def _stress_polar_kirsch(r, th):
    """独立な参照: 教科書の極座標 Kirsch 公式"""
    a2 = A_HOLE**2
    srr = (S_FAR / 2 * (1 - a2 / r**2)
           + S_FAR / 2 * (1 - 4 * a2 / r**2 + 3 * a2**2 / r**4) * np.cos(2 * th))
    stt = (S_FAR / 2 * (1 + a2 / r**2)
           - S_FAR / 2 * (1 + 3 * a2**2 / r**4) * np.cos(2 * th))
    srt = -S_FAR / 2 * (1 + 2 * a2 / r**2 - 3 * a2**2 / r**4) * np.sin(2 * th)
    return srr, stt, srt


def _cart_to_polar_stress(sxx, syy, sxy, th):
    c, s = np.cos(th), np.sin(th)
    srr = sxx * c**2 + syy * s**2 + 2 * sxy * c * s
    stt = sxx * s**2 + syy * c**2 - 2 * sxy * c * s
    srt = (syy - sxx) * s * c + sxy * (c**2 - s**2)
    return srr, stt, srt


def verify_exact_solution(verbose=True):
    """解析解が正しいことを3つの独立な方法で確認する。

    これが通らないと以降の比較に意味がないので、ベンチの前に必ず実行する。
    """
    ok = True

    # (1) 複素ポテンシャル由来の応力 vs 教科書の極座標 Kirsch 公式
    rng = np.random.default_rng(0)
    r = rng.uniform(A_HOLE, 6.0, 2000)
    th = rng.uniform(0, 2 * np.pi, 2000)
    x, y = r * np.cos(th), r * np.sin(th)
    srr_c, stt_c, srt_c = _cart_to_polar_stress(*stress_exact(x, y), th)
    srr_k, stt_k, srt_k = _stress_polar_kirsch(r, th)
    d1 = max(np.max(np.abs(srr_c - srr_k)), np.max(np.abs(stt_c - stt_k)),
             np.max(np.abs(srt_c - srt_k)))
    ok &= d1 < 1e-12

    # (2) 孔縁の traction-free と K_t = 3
    th2 = np.linspace(0, 2 * np.pi, 2001)
    xh, yh = A_HOLE * np.cos(th2), A_HOLE * np.sin(th2)
    srr_h, stt_h, srt_h = _cart_to_polar_stress(*stress_exact(xh, yh), th2)
    d2 = max(np.max(np.abs(srr_h)), np.max(np.abs(srt_h)))
    kt = np.max(stt_h) / S_FAR
    ok &= d2 < 1e-12 and abs(kt - 3.0) < 1e-4

    # (3) 変位 → ひずみ → Hooke で応力に戻るか（有限差分）
    h = 1e-6
    xs = np.array([2.0, 1.5, 3.0, 1.01])
    ys = np.array([0.3, 1.2, -2.5, 0.0])
    ux_p, _ = uv_exact(xs + h, ys)
    ux_m, _ = uv_exact(xs - h, ys)
    uy_p, vy_p = uv_exact(xs, ys + h)
    uy_m, vy_m = uv_exact(xs, ys - h)
    _, vx_p = uv_exact(xs + h, ys)
    _, vx_m = uv_exact(xs - h, ys)
    exx = (ux_p - ux_m) / (2 * h)
    eyy = (vy_p - vy_m) / (2 * h)
    exy = 0.5 * ((uy_p - uy_m) / (2 * h) + (vx_p - vx_m) / (2 * h))
    sxx_fd = LAM_PS * (exx + eyy) + 2 * MU_G * exx
    syy_fd = LAM_PS * (exx + eyy) + 2 * MU_G * eyy
    sxy_fd = 2 * MU_G * exy
    sxx_e, syy_e, sxy_e = stress_exact(xs, ys)
    d3 = max(np.max(np.abs(sxx_fd - sxx_e)), np.max(np.abs(syy_fd - syy_e)),
             np.max(np.abs(sxy_fd - sxy_e)))
    ok &= d3 < 1e-7

    if verbose:
        print("解析解の検証:")
        print(f"  (1) 複素ポテンシャル vs 極座標 Kirsch 公式 : max|diff| = {d1:.2e}")
        print(f"  (2) 孔縁 traction-free                    : max|σ_rr|,|σ_rθ| = {d2:.2e}")
        print(f"      応力集中係数 K_t                      : {kt:.6f} (= 3)")
        print(f"  (3) 変位→ひずみ→Hooke（有限差分）         : max|diff| = {d3:.2e}")
        print(f"  → {'OK' if ok else '*** 失敗 ***'}")
    return ok


# ==============================================================
# 3. 近似器
# ==============================================================
class MLPPINN(torch.nn.Module):
    """古典 PINN: (x, y) -> (u, v)"""

    def __init__(self, hidden=32, depth=3):
        super().__init__()
        layers, d_in = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d_in, hidden), torch.nn.Tanh()]
            d_in = hidden
        layers += [torch.nn.Linear(d_in, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


class FourierPINN(torch.nn.Module):
    """Fourier feature PINN: 角度エンコード QPINN と同じ関数クラスを持つ古典モデル。

    QPINN の優位を主張するなら必ず比較すべきベースライン（dequantization 対策）。
    """

    def __init__(self, n_freq=6, hidden=32):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("B", torch.randn(2, n_freq, generator=g) * 1.5)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * n_freq + 2, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2))

    def forward(self, xy):
        z = xy @ self.B
        feats = torch.cat([torch.sin(z), torch.cos(z), xy], dim=1)
        return self.net(feats)


# --- QPINN -----------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 3
N_ENCODE = 2

_DEV_CACHE, _QNODE_CACHE = {}, {}


def _get_qnode(n_qubits, n_layers, n_encode):
    """2D 入力 (x, y) を受け、2つの観測量 ⟨Z₀⟩, ⟨Z₁⟩ を返す回路。

    多出力（ベクトル場）の読み出しを「観測量を分ける」方式で実装している。
    計画 §4 が最大のリスクに挙げていた設計。
    """
    key = (n_qubits, n_layers, n_encode)
    if key in _QNODE_CACHE:
        return _QNODE_CACHE[key]
    if n_qubits not in _DEV_CACHE:
        _DEV_CACHE[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    dev = _DEV_CACHE[n_qubits]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(xs, ys, theta):
        for e in range(n_encode):
            # エンコード層: x を RY、y を RZ に入れる（全 qubit 同じ角度 = 低周波）。
            # #1 のアブレーション（12_qpinn_ablation.py）で、qubit ごとに周波数を
            # 上げる ladder 方式は表現力を上げずに最適化だけ壊すと分かっているため。
            for w in range(n_qubits):
                qml.RY(xs, wires=w)
                qml.RZ(ys, wires=w)
            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.RY(theta[e, l, w, 0], wires=w)
                    qml.RZ(theta[e, l, w, 1], wires=w)
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    _QNODE_CACHE[key] = circuit
    return circuit


class QPINN(torch.nn.Module):
    def __init__(self, seed=0, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_encode=N_ENCODE):
        super().__init__()
        self.circuit = _get_qnode(n_qubits, n_layers, n_encode)
        g = torch.Generator().manual_seed(seed)
        self.theta = torch.nn.Parameter(
            torch.rand(n_encode, n_layers, n_qubits, 2, generator=g) * 2 * math.pi)
        # 出力ごとに独立なスケール・シフト
        self.scale = torch.nn.Parameter(torch.ones(2))
        self.shift = torch.nn.Parameter(torch.zeros(2))

    def forward(self, xy):
        # 領域 [-R, R]² を [-π, π] に正規化
        xs = math.pi * xy[:, 0] / R_OUT
        ys = math.pi * xy[:, 1] / R_OUT
        out = self.circuit(xs, ys, self.theta)
        out = torch.stack([out[0].reshape(-1), out[1].reshape(-1)], dim=1)
        return self.scale * out + self.shift


class QPINNHybrid(torch.nn.Module):
    """ハイブリッド QPINN: 古典層で挟む構成。

    OIST の Rakala 実装（github.com/GeetRakala/QPINN、2D Navier-Stokes 円柱後流）を参考にした。
    向こうは素の量子（79 params, loss 1.03e-1）に古典層を足して 311 params で 2.89e-2 と
    3.5 倍改善している。この構成が表現力の頭打ちを破るかを確かめる。

        (x, y) → Dense(h) → tanh → Dense(n_qubits) → 量子回路 → Dense(h) → tanh → Dense(2)

    注意: 古典層を厚くすると「量子回路が何をしているのか」が曖昧になる。
    パラメータ効率を主張するなら、同じ古典層だけを持つモデル（量子回路を恒等写像に
    置き換えたもの）との比較が必須。`QPINNHybrid(n_qubits=0)` がそれを与える。
    """

    def __init__(self, seed=0, n_qubits=4, n_layers=3, n_encode=2, hidden=8):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(2, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, max(n_qubits, 1)))
        if n_qubits > 0:
            self.circuit = _get_qnode_multi(n_qubits, n_layers, n_encode)
            g = torch.Generator().manual_seed(seed)
            self.theta = torch.nn.Parameter(
                torch.rand(n_encode, n_layers, n_qubits, 2, generator=g) * 2 * math.pi)
        else:
            self.circuit = None
        self.post = torch.nn.Sequential(
            torch.nn.Linear(max(n_qubits, 1), hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2))

    def forward(self, xy):
        z = torch.tanh(self.pre(xy)) * math.pi      # 回路の角度に収める
        if self.circuit is None:
            q = z                                    # アブレーション: 量子回路なし
        else:
            out = self.circuit(z, self.theta)
            q = torch.stack([o.reshape(-1) for o in out], dim=1)
        return self.post(q)


def _get_qnode_multi(n_qubits, n_layers, n_encode):
    """各 qubit に別々の入力特徴量を入れ、全 qubit の ⟨Z⟩ を返す回路"""
    key = ("multi", n_qubits, n_layers, n_encode)
    if key in _QNODE_CACHE:
        return _QNODE_CACHE[key]
    if n_qubits not in _DEV_CACHE:
        _DEV_CACHE[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    dev = _DEV_CACHE[n_qubits]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(z, theta):
        for e in range(n_encode):
            for w in range(n_qubits):
                qml.RY(z[:, w], wires=w)
            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.RY(theta[e, l, w, 0], wires=w)
                    qml.RZ(theta[e, l, w, 1], wires=w)
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    _QNODE_CACHE[key] = circuit
    return circuit


# ==============================================================
# 4. PINN の loss
# ==============================================================
def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def stresses_from_uv(model, xy):
    """(u, v) から平面応力の σxx, σyy, σxy を autograd で出す"""
    xy = xy.clone().requires_grad_(True)
    uv = model(xy)
    u, v = uv[:, 0], uv[:, 1]
    du = _grad(u, xy)
    dv = _grad(v, xy)
    exx, eyy = du[:, 0], dv[:, 1]
    exy = 0.5 * (du[:, 1] + dv[:, 0])
    sxx = LAM_PS * (exx + eyy) + 2 * MU_G * exx
    syy = LAM_PS * (exx + eyy) + 2 * MU_G * eyy
    sxy = 2 * MU_G * exy
    return xy, sxx, syy, sxy


def sample_points(n_colloc, n_bc, device=None):
    """円環領域 a ≤ r ≤ R の内点と、内外境界の点"""
    g = torch.Generator().manual_seed(0)
    # 面積に比例させるため r は sqrt 分布
    t = torch.rand(n_colloc, generator=g)
    r = torch.sqrt(A_HOLE**2 + t * (R_OUT**2 - A_HOLE**2))
    th = torch.rand(n_colloc, generator=g) * 2 * math.pi
    xy = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1)

    th_b = torch.linspace(0, 2 * math.pi, n_bc + 1)[:-1]
    xy_in = torch.stack([A_HOLE * torch.cos(th_b), A_HOLE * torch.sin(th_b)], dim=1)
    xy_out = torch.stack([R_OUT * torch.cos(th_b), R_OUT * torch.sin(th_b)], dim=1)
    return xy, xy_in, xy_out, th_b


def pinn_loss(model, xy, xy_in, xy_out, uv_out_ref, th_b, w_bc=10.0, w_dir=10.0):
    # --- PDE 残差（平衡方程式） ---
    pts, sxx, syy, sxy = stresses_from_uv(model, xy)
    r1 = _grad(sxx, pts)[:, 0] + _grad(sxy, pts)[:, 1]
    r2 = _grad(sxy, pts)[:, 0] + _grad(syy, pts)[:, 1]
    l_pde = torch.mean(r1**2 + r2**2)

    # --- 内側 r=a: traction-free（σ_rr = σ_rθ = 0） ---
    pts_i, sxx_i, syy_i, sxy_i = stresses_from_uv(model, xy_in)
    c, s = torch.cos(th_b), torch.sin(th_b)
    srr = sxx_i * c**2 + syy_i * s**2 + 2 * sxy_i * c * s
    srt = (syy_i - sxx_i) * s * c + sxy_i * (c**2 - s**2)
    l_hole = torch.mean(srr**2 + srt**2)

    # --- 外側 r=R: 変位の Dirichlet 条件 ---
    uv_pred = model(xy_out)
    l_dir = torch.mean((uv_pred - uv_out_ref) ** 2)

    total = l_pde + w_bc * l_hole + w_dir * l_dir
    return total, l_pde.item(), l_hole.item(), l_dir.item()


# ==============================================================
# 5. 評価指標
# ==============================================================
def _eval_grid(n_r=40, n_t=80):
    r = torch.linspace(A_HOLE, R_OUT, n_r)
    th = torch.linspace(0, 2 * math.pi, n_t)
    R, T = torch.meshgrid(r, th, indexing="ij")
    return torch.stack([(R * torch.cos(T)).reshape(-1),
                        (R * torch.sin(T)).reshape(-1)], dim=1)


def rel_l2_uv(model):
    xy = _eval_grid()
    with torch.no_grad():
        pred = model(xy).numpy()
    ue, ve = uv_exact(xy[:, 0].numpy(), xy[:, 1].numpy())
    ref = np.stack([ue, ve], axis=1)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_stress(model):
    xy = _eval_grid()
    _, sxx, syy, sxy = stresses_from_uv(model, xy)
    pred = np.stack([sxx.detach().numpy(), syy.detach().numpy(),
                     sxy.detach().numpy()], axis=1)
    ref = np.stack(stress_exact(xy[:, 0].numpy(), xy[:, 1].numpy()), axis=1)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def kt_predicted(model, n_t=721):
    """孔縁の σ_θθ の最大値 = 応力集中係数 K_t。真値は 3.0

    この問題で最も重要な工学的指標。L2 誤差が小さくても K_t を外すことがある。
    """
    th = torch.linspace(0, 2 * math.pi, n_t)
    xy = torch.stack([A_HOLE * torch.cos(th), A_HOLE * torch.sin(th)], dim=1)
    _, sxx, syy, sxy = stresses_from_uv(model, xy)
    c, s = torch.cos(th), torch.sin(th)
    stt = sxx * s**2 + syy * c**2 - 2 * sxy * c * s
    return float(stt.detach().max()) / S_FAR


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ==============================================================
# 6. 学習
# ==============================================================
def train(name, model, steps=3000, lr=0.01, n_colloc=512, n_bc=64, log_every=500,
          lbfgs_steps=0):
    """Adam で学習し、必要なら L-BFGS で仕上げる。

    L-BFGS 仕上げは PINN の定番で、Adam が止まった後に 1〜2 桁効くことがある
    （OIST の Rakala 実装 github.com/GeetRakala/QPINN も Adam → L-BFGS-B）。
    ベンチ #2 では「表現力を直すと最適化が律速になる」と分かったので、ここが効くはず。
    """
    xy, xy_in, xy_out, th_b = sample_points(n_colloc, n_bc)
    ue, ve = uv_exact(xy_out[:, 0].numpy(), xy_out[:, 1].numpy())
    uv_out_ref = torch.tensor(np.stack([ue, ve], axis=1))

    def closure_loss():
        return pinn_loss(model, xy, xy_in, xy_out, uv_out_ref, th_b)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1),
                                                       eta_min=lr * 0.01)
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        loss, l_pde, l_hole, l_dir = closure_loss()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss={loss.item():.3e} "
                  f"(pde={l_pde:.2e}, hole={l_hole:.2e}, dir={l_dir:.2e})  "
                  f"rel_L2(u)={rel_l2_uv(model):.3e}", flush=True)
    t_adam = time.time() - t0
    res_adam = {"u_err": rel_l2_uv(model), "s_err": rel_l2_stress(model),
                "kt": kt_predicted(model)}
    print(f"    [Adam 終了] rel_L2(u)={res_adam['u_err']:.3e}, "
          f"K_t={res_adam['kt']:.3f}, {t_adam:.0f}s", flush=True)

    # --- L-BFGS 仕上げ ---
    t_lbfgs = 0.0
    if lbfgs_steps > 0:
        t1 = time.time()
        lopt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                 history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            lopt.zero_grad()
            loss, *_ = closure_loss()
            loss.backward()
            return loss

        for it in range(lbfgs_steps):
            loss = lopt.step(closure)
            if not torch.isfinite(loss):
                print(f"    [L-BFGS] loss が発散したので {it} 回で打ち切り", flush=True)
                break
            if it % max(lbfgs_steps // 4, 1) == 0 or it == lbfgs_steps - 1:
                print(f"    L-BFGS {it:4d}  loss={loss.item():.3e}  "
                      f"rel_L2(u)={rel_l2_uv(model):.3e}", flush=True)
        t_lbfgs = time.time() - t1

    dt = t_adam + t_lbfgs
    res = {"name": name, "n_param": n_params(model), "u_err": rel_l2_uv(model),
           "s_err": rel_l2_stress(model), "kt": kt_predicted(model), "time": dt,
           "adam_u_err": res_adam["u_err"], "adam_kt": res_adam["kt"],
           "t_adam": t_adam, "t_lbfgs": t_lbfgs}
    tag = f" (Adam {t_adam:.0f}s + L-BFGS {t_lbfgs:.0f}s)" if lbfgs_steps else ""
    print(f"  → {name}: rel_L2(u)={res['u_err']:.3e}, rel_L2(σ)={res['s_err']:.3e}, "
          f"K_t={res['kt']:.3f} (真値 3.000), params={res['n_param']}, {dt:.0f}s{tag}",
          flush=True)
    return res


def supervised_fit(model, steps=2000, lr=0.02, n=1024):
    """表現力の上限: 解析解の変位を直接フィットする（PDE 損失を使わない）。

    #1 のアブレーションで「誤差の大半は表現力ではなく最適化で失われる」と分かったので、
    応力集中でも同じ切り分けをする。ここで既に駄目なら表現力の限界が原因。
    """
    g = torch.Generator().manual_seed(1)
    t = torch.rand(n, generator=g)
    r = torch.sqrt(A_HOLE**2 + t * (R_OUT**2 - A_HOLE**2))
    th = torch.rand(n, generator=g) * 2 * math.pi
    xy = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1)
    ue, ve = uv_exact(xy[:, 0].numpy(), xy[:, 1].numpy())
    target = torch.tensor(np.stack([ue, ve], axis=1))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((model(xy) - target) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return rel_l2_uv(model), kt_predicted(model)


# ==============================================================
# 7. 実行
# ==============================================================
def expressivity_sweep(configs=((4, 3, 2), (6, 3, 2), (8, 3, 2), (4, 3, 3), (6, 4, 3)),
                       sup_steps=2000, seed=0):
    """QPINN の表現力だけを測るスイープ（PDE 損失を使わないので安い）。

    #1（1D・滑らか）では表現力は十分で最適化が律速だった。2D・応力集中では
    まず「そもそも表現できるのか」を確かめる必要がある。表現力が足りないなら
    最適化をいくら触っても無駄で、それ自体が QPINN の限界という主張になる。
    """
    print("=" * 72)
    print("  QPINN 表現力スイープ（教師ありフィット、PDE 損失なし）")
    print("=" * 72)
    rows = []
    for n_q, n_l, n_e in configs:
        torch.manual_seed(seed)
        m = QPINN(seed=seed, n_qubits=n_q, n_layers=n_l, n_encode=n_e)
        t0 = time.time()
        u_err, kt = supervised_fit(m, steps=sup_steps)
        rows.append({"cfg": (n_q, n_l, n_e), "n_param": n_params(m),
                     "u_err": u_err, "kt": kt, "time": time.time() - t0})
        print(f"  q={n_q} l={n_l} e={n_e}: params={rows[-1]['n_param']:<3} "
              f"rel_L2(u)={u_err:.3e}  K_t={kt:.3f}  ({rows[-1]['time']:.0f}s)", flush=True)

    print(f"\n参考: 古典 MLP(32x3) の表現力は約 1.5e-02、K_t の真値は 3.000")
    return rows


def main(steps=20000, q_steps=1500, n_colloc=512, seed=0, skip_qpinn=False,
         with_supervised=True, lbfgs_steps=0):
    print("=" * 72)
    print("  QPINN ベンチ #2: 円孔付き板の応力集中（Kirsch 解）")
    print("=" * 72)
    print(f"円孔半径 a={A_HOLE}, 外半径 R={R_OUT}, 遠方引張 S={S_FAR}, E={E_MOD}, ν={NU}")
    if not verify_exact_solution():
        raise SystemExit("解析解の検証に失敗した")

    torch.manual_seed(seed)
    results = []

    specs = [("古典PINN", lambda: MLPPINN(), steps, 0.01),
             ("Fourier PINN", lambda: FourierPINN(), steps, 0.01)]
    if not skip_qpinn:
        specs.append(("QPINN", lambda: QPINN(seed=seed), q_steps, 0.02))

    for name, ctor, st, lr in specs:
        print("\n" + "=" * 72)
        print(f"  {name}")
        print("=" * 72)
        torch.manual_seed(seed)
        r = train(name, ctor(), steps=st, lr=lr, n_colloc=n_colloc,
                  lbfgs_steps=lbfgs_steps)
        if with_supervised:
            torch.manual_seed(seed)
            sup_u, sup_kt = supervised_fit(ctor())
            r["sup_u"], r["sup_kt"] = sup_u, sup_kt
            print(f"     表現力の上限（教師ありフィット）: rel_L2(u)={sup_u:.3e}, "
                  f"K_t={sup_kt:.3f}")
        results.append(r)

    print("\n" + "=" * 72)
    print("  比較表")
    print("=" * 72)
    head = f"{'モデル':<14}{'params':>8}{'rel_L2(u)':>12}{'rel_L2(σ)':>12}{'K_t':>8}{'秒':>7}"
    if with_supervised:
        head += f"{'表現力':>12}"
    print(head)
    for r in results:
        line = (f"{r['name']:<14}{r['n_param']:>8}{r['u_err']:>12.3e}"
                f"{r['s_err']:>12.3e}{r['kt']:>8.3f}{r['time']:>7.0f}")
        if with_supervised:
            line += f"{r['sup_u']:>12.3e}"
        print(line)
    print(f"\nK_t の真値は 3.000。L2 誤差が小さくても K_t を外すことがあるので併記する。")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000,
                    help="古典モデルの学習ステップ数。3000 では未収束だったので増やしてある")
    ap.add_argument("--q-steps", type=int, default=1500, help="QPINN の学習ステップ数")
    ap.add_argument("--colloc", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-qpinn", action="store_true")
    ap.add_argument("--no-supervised", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="解析解の検証だけ実行する")
    ap.add_argument("--lbfgs", type=int, default=0, dest="lbfgs_steps",
                    help="Adam の後に回す L-BFGS の反復数（1反復あたり最大20回の内部反復）")
    ap.add_argument("--expressivity-sweep", action="store_true",
                    help="QPINN の表現力スイープだけ実行する（PDE 学習なし）")
    args = ap.parse_args()

    if args.verify_only:
        verify_exact_solution()
    elif args.expressivity_sweep:
        verify_exact_solution()
        expressivity_sweep(seed=args.seed)
    else:
        main(steps=args.steps, q_steps=args.q_steps, n_colloc=args.colloc,
             seed=args.seed, skip_qpinn=args.skip_qpinn,
             with_supervised=not args.no_supervised, lbfgs_steps=args.lbfgs_steps)
