# -*- coding: utf-8 -*-
"""
14 - QPINN × Deep Energy Method: 熱弾性連成（2D 円環 / 3D 中空球）

ベンチ #1（Lamé, 1D）・#2（Kirsch, 2D 応力集中）で分かったこと:

- 1D・滑らか     → 最適化が律速（表現力は十分）
- 2D・応力集中   → 表現力が律速。古典層のサンドイッチで破れるが、今度は最適化が律速
- L-BFGS は古典 PINN を 3.1 倍改善するが QPINN は 1.1 倍しか動かない
  → QPINN の最適化の壁は2次法で直る悪条件ではない。損失の形そのものを変える必要がある

そこで **Deep Energy Method (DEM)** を試す。PDE 残差ではなくポテンシャルエネルギー汎関数を
最小化する定式化で、QPINN に対して2つ効くはずの性質がある:

1. **u の1階微分しか現れない。** 残差形は2階微分が要る。実機では parameter-shift 則で
   高階微分のコストが爆発するので、これは本質的な違い。
2. **traction-free 境界が自然境界条件になる。** #2 では σ_rr = σ_rθ = 0 を罰則項で
   入れる必要があったが、DEM では汎関数を最小化するだけで満たされる。損失項が減るので
   損失バランスの問題も減る。

## 問題（一方向連成: 熱 → 変形）

段階1: 定常熱伝導 ∇²T = 0 を解いて T を得る
段階2: T を固定して熱弾性 σ = C(ε(u) − αT·I), div σ = 0 を解く

どちらも解析解のある形状を使う:

| | 2D | 3D |
|---|---|---|
| 形状 | 円環板 a≤r≤b（平面応力） | 中空球 a≤r≤b |
| 温度 | T = c₀ + c₁ ln r | T = d₀ + d₁/r |
| 応力 | 厚肉円筒熱応力の閉形式解 | 中空球熱応力の閉形式解 |
| 境界 | 内外面とも traction-free | 同左 |

`verify_exact_solution()` が熱伝導方程式・平衡方程式・traction-free・Hooke 則の
4つを独立に検証する。

## エネルギー汎関数

熱（ポアソン）:  Π_T[T] = ∫ ½k|∇T|² dV  （Dirichlet 条件は罰則）
熱弾性:          Π[u]   = ∫ W(ε(u) − αT·I) dV  （traction-free は自然境界条件）

2D 平面応力: W = ½[σ_xx e_xx + σ_yy e_yy + 2σ_xy e_xy],  σ = C_ps e
3D:          W = ½λ(tr e)² + μ e:e

剛体モード（並進・回転）はエネルギーがゼロなので罰則で固定する。
"""

import argparse
import math
import time

import numpy as np
import torch
import pennylane as qml

torch.set_default_dtype(torch.float64)

# ==============================================================
# 1. 問題設定（無次元化）
# ==============================================================
A_IN, B_OUT = 1.0, 2.0      # 内半径・外半径
T_A, T_B = 1.0, 0.0         # 内面・外面温度
E_MOD, NU, ALPHA = 1.0, 0.3, 1.0
K_COND = 1.0                # 熱伝導率

LAM_3D = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU_G = E_MOD / (2 * (1 + NU))
C_PS = E_MOD / (1 - NU**2)   # 平面応力


# --- 温度の解析解 ---------------------------------------------------
def T_exact(r, dim):
    r = np.asarray(r, dtype=float)
    if dim == 2:
        return T_A + (T_B - T_A) * np.log(r / A_IN) / np.log(B_OUT / A_IN)
    d1 = (T_A - T_B) / (1 / A_IN - 1 / B_OUT)
    d0 = T_A - d1 / A_IN
    return d0 + d1 / r


def _moment(r, dim):
    """2D: ∫_a^r T ρ dρ   3D: ∫_a^r T ρ² dρ  （解析積分）"""
    r = np.asarray(r, dtype=float)
    if dim == 2:
        c1 = (T_B - T_A) / np.log(B_OUT / A_IN)
        c0 = T_A - c1 * np.log(A_IN)          # T = c0 + c1 ln ρ

        def F(x):
            return c0 * x**2 / 2 + c1 * (x**2 / 2 * np.log(x) - x**2 / 4)
        return F(r) - F(A_IN)
    d1 = (T_A - T_B) / (1 / A_IN - 1 / B_OUT)
    d0 = T_A - d1 / A_IN
    return d0 * (r**3 - A_IN**3) / 3 + d1 * (r**2 - A_IN**2) / 2


# --- 熱応力・変位の解析解 -------------------------------------------
def stress_exact_polar(r, dim):
    """(σ_r, σ_θ) を返す。内外面とも traction-free"""
    r = np.asarray(r, dtype=float)
    Ib = _moment(B_OUT, dim)
    if dim == 2:
        k = ALPHA * E_MOD
        sr = k * ((r**2 - A_IN**2) / (B_OUT**2 - A_IN**2) * Ib - _moment(r, 2)) / r**2
        st = k * ((r**2 + A_IN**2) / (B_OUT**2 - A_IN**2) * Ib
                  + _moment(r, 2) - T_exact(r, 2) * r**2) / r**2
        return sr, st
    k = ALPHA * E_MOD / (1 - NU)
    sr = 2 * k * ((r**3 - A_IN**3) / (B_OUT**3 - A_IN**3) * Ib - _moment(r, 3)) / r**3
    st = k * ((2 * r**3 + A_IN**3) / (B_OUT**3 - A_IN**3) * Ib
              + _moment(r, 3) - T_exact(r, 3) * r**3) / r**3
    return sr, st


def u_r_exact(r, dim):
    """半径方向変位 u_r = r ε_θ"""
    sr, st = stress_exact_polar(r, dim)
    if dim == 2:
        return r * ((st - NU * sr) / E_MOD + ALPHA * T_exact(r, 2))
    return r * ((st * (1 - NU) - NU * sr) / E_MOD + ALPHA * T_exact(r, 3))


def uv_exact(x, dim):
    """直交座標の変位ベクトル。x: (N, dim)"""
    r = np.linalg.norm(x, axis=1)
    return (u_r_exact(r, dim) / r)[:, None] * x


# ==============================================================
# 2. 解析解の検証
# ==============================================================
def verify_exact_solution(dim, verbose=True):
    """熱伝導・平衡・traction-free・Hooke 則の4つを独立に検証する"""
    h = 1e-4          # 2階微分を差分で取るので h は大きめ（丸め誤差 ~ eps/h²）
    rs = np.linspace(A_IN + 0.05, B_OUT - 0.05, 9)

    def d1(f, r, hh=1e-6):
        return (f(r + hh) - f(r - hh)) / (2 * hh)

    # (1) 熱伝導 (r^{dim-1} T')' = 0
    p = dim - 1
    res_T = np.array([d1(lambda x: x**p * d1(lambda y: T_exact(y, dim), x, h), r, h)
                      for r in rs])
    e1 = float(np.max(np.abs(res_T)))

    # (2) 平衡 σ_r' + (dim-1)(σ_r − σ_θ)/r = 0
    sr_f = lambda x: stress_exact_polar(x, dim)[0]
    res_eq = np.array([d1(sr_f, r) + (dim - 1) * (stress_exact_polar(r, dim)[0]
                                                  - stress_exact_polar(r, dim)[1]) / r
                       for r in rs])
    e2 = float(np.max(np.abs(res_eq)))

    # (3) traction-free
    e3 = max(abs(float(stress_exact_polar(A_IN, dim)[0])),
             abs(float(stress_exact_polar(B_OUT, dim)[0])))

    # (4) 変位 → ひずみ → Hooke が応力に戻るか
    errs = []
    for r in [1.3, 1.7]:
        er = d1(lambda x: u_r_exact(x, dim), r)
        et = float(u_r_exact(r, dim)) / r
        eth = ALPHA * float(T_exact(r, dim))
        if dim == 2:
            sr_h = C_PS * ((er - eth) + NU * (et - eth))
            st_h = C_PS * ((et - eth) + NU * (er - eth))
        else:
            tr = (er - eth) + 2 * (et - eth)
            sr_h = LAM_3D * tr + 2 * MU_G * (er - eth)
            st_h = LAM_3D * tr + 2 * MU_G * (et - eth)
        sr_e, st_e = stress_exact_polar(r, dim)
        errs += [abs(sr_h - float(sr_e)), abs(st_h - float(st_e))]
    e4 = max(errs)

    ok = e1 < 1e-6 and e2 < 1e-8 and e3 < 1e-12 and e4 < 1e-8
    if verbose:
        name = "2D 円環板（平面応力）" if dim == 2 else "3D 中空球"
        print(f"解析解の検証 [{name}]:")
        print(f"  (1) 熱伝導 (r^{dim-1} T')' = 0        : max|res| = {e1:.2e}")
        print(f"  (2) 平衡 σr' + {dim-1}(σr−σθ)/r = 0   : max|res| = {e2:.2e}")
        print(f"  (3) traction-free σr(a), σr(b)       : max|σr|  = {e3:.2e}")
        print(f"  (4) 変位→ひずみ→Hooke                 : max|diff| = {e4:.2e}")
        print(f"  → {'OK' if ok else '*** 失敗 ***'}")
    return ok


# ==============================================================
# 3. 近似器
# ==============================================================
class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden=32, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), torch.nn.Tanh()]
            d = hidden
        layers += [torch.nn.Linear(d, d_out)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


_DEV_CACHE, _QNODE_CACHE = {}, {}


def _get_qnode(n_qubits, n_layers, n_encode, entangle=True):
    """entangle=False で CNOT を全部外す（積状態のみ＝古典で効率よく計算できる対照）"""
    key = (n_qubits, n_layers, n_encode, entangle)
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
                if entangle:
                    for w in range(n_qubits - 1):
                        qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    _QNODE_CACHE[key] = circuit
    return circuit


class QPINNHybrid(torch.nn.Module):
    """古典層で挟むハイブリッド QPINN。

    ベンチ #2 で、素の角度エンコード構成は 2D ベクトル場を表現できず（3.28e-01）、
    この構成で 13 倍改善した（2.59e-02）ことが分かっている。
    `n_qubits=0` で量子回路を外した対照モデルになる（#2 ではこれが 3.06e-01 で、
    古典層だけでは駄目＝量子回路が実際に寄与していることの証拠だった）。
    """

    def __init__(self, d_in, d_out, seed=0, n_qubits=4, n_layers=3, n_encode=2, hidden=8,
                 entangle=True, identity_width=None):
        """identity_width を与えると量子回路を恒等写像に置き換えた対照になる
        （前後の古典層は同じ幅のまま）。n_qubits=0 の旧対照は幅1に潰れるので、
        公平な対照としてはこちらを使う。"""
        super().__init__()
        if identity_width is not None:
            n_qubits = 0
        self.n_qubits = n_qubits
        w = identity_width if identity_width is not None else max(n_qubits, 1)
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, w))
        if n_qubits > 0:
            self.circuit = _get_qnode(n_qubits, n_layers, n_encode, entangle=entangle)
            g = torch.Generator().manual_seed(seed)
            self.theta = torch.nn.Parameter(
                torch.rand(n_encode, n_layers, n_qubits, 2, generator=g) * 2 * math.pi)
        else:
            self.circuit = None
        self.post = torch.nn.Sequential(
            torch.nn.Linear(w, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, d_out))

    def forward(self, x):
        z = torch.tanh(self.pre(x)) * math.pi
        if self.circuit is None:
            q = z
        else:
            out = self.circuit(z, self.theta)
            q = torch.stack([o.reshape(-1) for o in out], dim=1)
        return self.post(q)


# ==============================================================
# 4. 点のサンプリング
# ==============================================================
def sample_domain(n, dim, seed=0):
    """円環 / 球殻の内部を体積に比例して一様サンプリング"""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g)
    if dim == 2:
        r = torch.sqrt(A_IN**2 + t * (B_OUT**2 - A_IN**2))
    else:
        r = (A_IN**3 + t * (B_OUT**3 - A_IN**3)) ** (1 / 3)
    v = torch.randn(n, dim, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    return r[:, None] * v


def sample_sphere(n, radius, dim, seed=0):
    """半径 radius の円周 / 球面上の点"""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    return radius * v / v.norm(dim=1, keepdim=True)


def domain_volume(dim):
    if dim == 2:
        return math.pi * (B_OUT**2 - A_IN**2)
    return 4 / 3 * math.pi * (B_OUT**3 - A_IN**3)


# ==============================================================
# 5. 段階1: 温度場
# ==============================================================
def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def thermal_loss_dem(model, x, x_in, x_out, w_bc=1000.0):
    """DEM: Π_T = ∫ ½k|∇T|² dV（1階微分のみ）+ Dirichlet 罰則"""
    x = x.clone().requires_grad_(True)
    T = model(x).reshape(-1)
    gT = _grad(T, x)
    energy = 0.5 * K_COND * (gT**2).sum(dim=1).mean() * domain_volume(x.shape[1])
    l_bc = ((model(x_in).reshape(-1) - T_A) ** 2).mean() + \
           ((model(x_out).reshape(-1) - T_B) ** 2).mean()
    return energy + w_bc * l_bc, energy.item(), l_bc.item()


def thermal_loss_residual(model, x, x_in, x_out, w_bc=1000.0):
    """残差形: ∇²T = 0（2階微分が要る）+ Dirichlet 罰則"""
    x = x.clone().requires_grad_(True)
    T = model(x).reshape(-1)
    gT = _grad(T, x)
    lap = sum(_grad(gT[:, i], x)[:, i] for i in range(x.shape[1]))
    l_pde = (lap**2).mean()
    l_bc = ((model(x_in).reshape(-1) - T_A) ** 2).mean() + \
           ((model(x_out).reshape(-1) - T_B) ** 2).mean()
    return l_pde + w_bc * l_bc, l_pde.item(), l_bc.item()


# ==============================================================
# 6. 段階2: 熱弾性
# ==============================================================
def strain(model, x):
    """ひずみテンソル ε(u) を返す。x は requires_grad 済みであること"""
    u = model(x)
    dim = x.shape[1]
    grads = [_grad(u[:, i], x) for i in range(dim)]          # grads[i][:, j] = ∂u_i/∂x_j
    G = torch.stack(grads, dim=1)                             # (N, dim, dim)
    return 0.5 * (G + G.transpose(1, 2))


def energy_density(eps_e, dim):
    """W = ½ C e : e （熱ひずみを引いた弾性ひずみ e を受ける）"""
    if dim == 2:
        exx, eyy = eps_e[:, 0, 0], eps_e[:, 1, 1]
        exy = eps_e[:, 0, 1]
        sxx = C_PS * (exx + NU * eyy)
        syy = C_PS * (eyy + NU * exx)
        sxy = E_MOD / (1 + NU) * exy
        return 0.5 * (sxx * exx + syy * eyy + 2 * sxy * exy)
    tr = eps_e[:, 0, 0] + eps_e[:, 1, 1] + eps_e[:, 2, 2]
    return 0.5 * LAM_3D * tr**2 + MU_G * (eps_e**2).sum(dim=(1, 2))


def _rigid_body_penalty(model, x):
    """剛体モード（並進・回転）はエネルギーがゼロなので固定する"""
    u = model(x)
    dim = x.shape[1]
    pen = (u.mean(dim=0) ** 2).sum()
    if dim == 2:
        rot = (x[:, 0] * u[:, 1] - x[:, 1] * u[:, 0]).mean()
        pen = pen + rot**2
    else:
        rot = torch.cross(x, u, dim=1).mean(dim=0)
        pen = pen + (rot**2).sum()
    return pen


def elastic_loss_dem(model, x, T_field, w_rb=100.0):
    """DEM: Π[u] = ∫ W(ε(u) − αT·I) dV

    traction-free は自然境界条件なので罰則項が要らない。
    #2（残差形）では σ_rr = σ_rθ = 0 を明示的に入れる必要があった。
    """
    dim = x.shape[1]
    x = x.clone().requires_grad_(True)
    eps = strain(model, x)
    eye = torch.eye(dim, dtype=x.dtype).expand(x.shape[0], dim, dim)
    eps_e = eps - ALPHA * T_field.reshape(-1, 1, 1) * eye
    energy = energy_density(eps_e, dim).mean() * domain_volume(dim)
    pen = _rigid_body_penalty(model, x)
    return energy + w_rb * pen, energy.item(), pen.item()


def stresses(model, x, T_field):
    """σ = C(ε(u) − αT·I)。x は requires_grad 済みであること"""
    dim = x.shape[1]
    eps = strain(model, x)
    eye = torch.eye(dim, dtype=x.dtype).expand(x.shape[0], dim, dim)
    e = eps - ALPHA * T_field.reshape(-1, 1, 1) * eye
    if dim == 2:
        exx, eyy, exy = e[:, 0, 0], e[:, 1, 1], e[:, 0, 1]
        sxx = C_PS * (exx + NU * eyy)
        syy = C_PS * (eyy + NU * exx)
        sxy = E_MOD / (1 + NU) * exy
        S = torch.stack([torch.stack([sxx, sxy], 1), torch.stack([sxy, syy], 1)], 1)
    else:
        tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
        S = LAM_3D * tr.reshape(-1, 1, 1) * eye + 2 * MU_G * e
    return S


def elastic_loss_residual(model, x, T_fn, x_in, x_out, w_bc=100.0, w_rb=100.0):
    """残差形: div σ = 0 + traction-free を明示的に罰則で入れる（#2 と同じ形）

    T_fn は点を受けて温度を返す「微分可能な」関数でなければならない。
    温度を定数テンソルで渡すと div σ の中の熱勾配項 −(3λ+2μ)α∇T が落ちて、
    別の場に収束してしまう（実際にこのバグを踏んだ）。
    DEM は σ を微分しないのでこの問題が起きない。
    """
    dim = x.shape[1]
    x = x.clone().requires_grad_(True)
    S = stresses(model, x, T_fn(x))
    div = torch.stack([sum(_grad(S[:, i, j], x)[:, j] for j in range(dim))
                       for i in range(dim)], dim=1)
    l_pde = (div**2).sum(dim=1).mean()

    l_bc = 0.0
    for xb in [x_in, x_out]:
        xb = xb.clone().requires_grad_(True)
        Sb = stresses(model, xb, T_fn(xb))
        n = xb / xb.norm(dim=1, keepdim=True)
        t = torch.einsum("nij,nj->ni", Sb, n)       # トラクション
        l_bc = l_bc + (t**2).sum(dim=1).mean()

    pen = _rigid_body_penalty(model, x)
    total = l_pde + w_bc * l_bc + w_rb * pen
    return total, l_pde.item(), l_bc.item(), pen.item()


# ==============================================================
# 7. 評価
# ==============================================================
def _eval_points(dim, n=2000):
    return sample_domain(n, dim, seed=999)


def rel_l2_T(model, dim):
    x = _eval_points(dim)
    with torch.no_grad():
        pred = model(x).reshape(-1).numpy()
    ref = T_exact(np.linalg.norm(x.numpy(), axis=1), dim)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_u(model, dim):
    x = _eval_points(dim)
    with torch.no_grad():
        pred = model(x).numpy()
    ref = uv_exact(x.numpy(), dim)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_sigma(model, dim, T_field_fn):
    """σ_r, σ_θ の相対 L2（解析解と比較）"""
    x = _eval_points(dim).requires_grad_(True)
    S = stresses(model, x, T_field_fn(x))
    n = x / x.norm(dim=1, keepdim=True)
    sr = torch.einsum("ni,nij,nj->n", n, S, n)
    tr = S[:, 0, 0] + S[:, 1, 1] + (S[:, 2, 2] if dim == 3 else 0)
    st = (tr - sr) / (dim - 1)          # 接線方向の平均
    r = x.norm(dim=1).detach().numpy()
    sr_e, st_e = stress_exact_polar(r, dim)
    pred = np.stack([sr.detach().numpy(), st.detach().numpy()], 1)
    ref = np.stack([sr_e, st_e], 1)
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ==============================================================
# 8. 学習
# ==============================================================
def _run(loss_fn, model, steps, lr, lbfgs_steps, metric, label, log_every,
         resample=None, resample_every=100):
    """Adam → L-BFGS。resample があれば collocation 点を定期的に取り直す。

    固定点のまま学習すると、DEM は数値積分そのものを過学習して
    「厳密解のエネルギーより低い値」に到達してしまう（点と点の間で振動する）。
    実測で確認済み: 厳密解 Π=0.357 に対し固定点では loss 0.328 まで下がり、
    真の誤差は逆に悪化した。再サンプリングはこれを防ぐために必須。
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1),
                                                       eta_min=lr * 0.01)
    t0 = time.time()
    for step in range(steps):
        if resample is not None and step > 0 and step % resample_every == 0:
            resample()
        opt.zero_grad()
        loss = loss_fn()[0]
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"      {label} {step:5d}  loss={loss.item():+.4e}  "
                  f"err={metric():.3e}", flush=True)
    t_adam = time.time() - t0
    err_adam = metric()

    t_lb = 0.0
    if lbfgs_steps > 0:
        t1 = time.time()
        lopt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                 history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lopt.zero_grad()
            l = loss_fn()[0]
            l.backward()
            return l

        for it in range(lbfgs_steps):
            # L-BFGS は内部の線形探索中に目的関数が変わると壊れるので、
            # 外側の反復の境目でだけ点を取り直す
            if resample is not None and it > 0:
                resample()
            l = lopt.step(closure)
            if not torch.isfinite(l):
                print(f"      [L-BFGS] 発散したので {it} 回で打ち切り", flush=True)
                break
        t_lb = time.time() - t1
        print(f"      {label} L-BFGS 後  err={metric():.3e}", flush=True)
    return {"err_adam": err_adam, "err": metric(), "t_adam": t_adam, "t_lbfgs": t_lb}


def solve_thermal(model, dim, form="dem", steps=3000, lr=0.01, lbfgs_steps=30,
                  n_colloc=1024, n_bc=256, log_every=1000, resample_every=100):
    pts = {"k": 0}

    def resample():
        pts["k"] += 1
        k = pts["k"]
        pts["x"] = sample_domain(n_colloc, dim, seed=1000 + k)
        pts["in"] = sample_sphere(n_bc, A_IN, dim, seed=2000 + k)
        pts["out"] = sample_sphere(n_bc, B_OUT, dim, seed=3000 + k)

    resample()
    fn = thermal_loss_dem if form == "dem" else thermal_loss_residual
    return _run(lambda: fn(model, pts["x"], pts["in"], pts["out"]), model, steps, lr,
                lbfgs_steps, lambda: rel_l2_T(model, dim), f"熱({form})", log_every,
                resample=resample, resample_every=resample_every)


def solve_elastic(model, dim, T_model, form="dem", steps=3000, lr=0.01, lbfgs_steps=30,
                  n_colloc=1024, n_bc=256, log_every=1000, use_exact_T=False,
                  resample_every=100):
    def T_fn(pts):
        """微分可能に温度を返す（残差形では ∇T が要る）"""
        if use_exact_T:
            r = pts.norm(dim=1)
            if dim == 2:
                return T_A + (T_B - T_A) * torch.log(r / A_IN) / math.log(B_OUT / A_IN)
            d1 = (T_A - T_B) / (1 / A_IN - 1 / B_OUT)
            return (T_A - d1 / A_IN) + d1 / r
        return T_model(pts).reshape(-1)

    pts = {"k": 0}

    def resample():
        pts["k"] += 1
        k = pts["k"]
        pts["x"] = sample_domain(n_colloc, dim, seed=1000 + k)
        pts["in"] = sample_sphere(n_bc, A_IN, dim, seed=2000 + k)
        pts["out"] = sample_sphere(n_bc, B_OUT, dim, seed=3000 + k)

    resample()

    if form == "dem":
        def loss_fn():
            x = pts["x"]
            with torch.no_grad():
                T_x = T_fn(x)
            return elastic_loss_dem(model, x, T_x)
    else:
        def loss_fn():
            return elastic_loss_residual(model, pts["x"], T_fn, pts["in"], pts["out"])

    res = _run(loss_fn, model, steps, lr, lbfgs_steps,
               lambda: rel_l2_u(model, dim), f"弾性({form})", log_every,
               resample=resample, resample_every=resample_every)
    res["s_err"] = rel_l2_sigma(model, dim, lambda p: T_fn(p).detach())
    return res


# ==============================================================
# 9. 実行
# ==============================================================
def make_model(kind, d_in, d_out, seed=0, n_qubits=4):
    if kind == "mlp":
        return MLP(d_in, d_out)
    if kind == "hybrid":
        return QPINNHybrid(d_in, d_out, seed=seed, n_qubits=n_qubits)
    if kind == "classical-only":
        return QPINNHybrid(d_in, d_out, seed=seed, n_qubits=0)
    raise ValueError(kind)


def main(dim=2, models=("mlp", "hybrid"), forms=("dem", "residual"),
         steps=3000, q_steps=1500, lbfgs_steps=30, n_colloc=1024,
         n_qubits=4, seed=0, use_exact_T=False):
    print("=" * 76)
    print(f"  QPINN × Deep Energy Method: 熱弾性連成（{dim}D）")
    print("=" * 76)
    if not verify_exact_solution(dim):
        raise SystemExit("解析解の検証に失敗した")

    rows = []
    for kind in models:
        for form in forms:
            print(f"\n{'=' * 76}\n  {kind} / {form}\n{'=' * 76}")
            st = steps if kind == "mlp" else q_steps
            torch.manual_seed(seed)

            # 段階1: 温度場
            Tm = make_model(kind, dim, 1, seed=seed, n_qubits=n_qubits)
            rt = solve_thermal(Tm, dim, form=form, steps=st, lbfgs_steps=lbfgs_steps,
                               n_colloc=n_colloc)

            # 段階2: 熱弾性（段階1 の T を使う。一方向連成）
            torch.manual_seed(seed)
            Um = make_model(kind, dim, dim, seed=seed, n_qubits=n_qubits)
            re = solve_elastic(Um, dim, Tm, form=form, steps=st,
                               lbfgs_steps=lbfgs_steps, n_colloc=n_colloc,
                               use_exact_T=use_exact_T)

            rows.append({"model": kind, "form": form,
                         "n_param_T": n_params(Tm), "n_param_u": n_params(Um),
                         "T_err": rt["err"], "u_err": re["err"], "s_err": re["s_err"],
                         "time": rt["t_adam"] + rt["t_lbfgs"] + re["t_adam"] + re["t_lbfgs"]})
            print(f"  → {kind}/{form}: T={rt['err']:.3e}, u={re['err']:.3e}, "
                  f"σ={re['s_err']:.3e}, {rows[-1]['time']:.0f}s", flush=True)

    print("\n" + "=" * 76)
    print(f"  比較表（{dim}D）")
    print("=" * 76)
    print(f"{'モデル':<16}{'定式化':<10}{'params(u)':>10}{'rel_L2(T)':>12}"
          f"{'rel_L2(u)':>12}{'rel_L2(σ)':>12}{'秒':>7}")
    for r in rows:
        print(f"{r['model']:<16}{r['form']:<10}{r['n_param_u']:>10}{r['T_err']:>12.3e}"
              f"{r['u_err']:>12.3e}{r['s_err']:>12.3e}{r['time']:>7.0f}")
    print("\nDEM は u の1階微分のみ・traction-free が自然境界条件。")
    print("残差形は2階微分が要り、traction-free を罰則で入れる必要がある。")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--models", default="mlp,hybrid",
                    help="mlp / hybrid / classical-only をカンマ区切りで")
    ap.add_argument("--forms", default="dem,residual", help="dem / residual")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--q-steps", type=int, default=1500)
    ap.add_argument("--lbfgs", type=int, default=30, dest="lbfgs_steps")
    ap.add_argument("--colloc", type=int, default=1024)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exact-T", action="store_true",
                    help="段階2 で解析解の温度を使う（熱の誤差を切り離す）")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        verify_exact_solution(2)
        print()
        verify_exact_solution(3)
    else:
        main(dim=args.dim, models=tuple(args.models.split(",")),
             forms=tuple(args.forms.split(",")), steps=args.steps,
             q_steps=args.q_steps, lbfgs_steps=args.lbfgs_steps,
             n_colloc=args.colloc, n_qubits=args.qubits, seed=args.seed,
             use_exact_T=args.exact_T)
