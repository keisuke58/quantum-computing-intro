# -*- coding: utf-8 -*-
"""
11 - QPINN ベンチ #1: 厚肉円筒の内圧問題（Lamé 解）

研究計画 docs/qpinn_rocket_research_plan.md のベンチマーク #1 の最小実装。
実装は PyTorch + PennyLane（torch インターフェース、diff_method="backprop"）。

問題（軸対称・平面ひずみ、線形弾性 Navier 方程式の 1D 形）:
    u'' + u'/r - u/r² = 0       (a ≤ r ≤ b)
    σ_r(a) = -p_i,  σ_r(b) = 0

解析解（Lamé）:
    σ_r = A - B/r²,  σ_θ = A + B/r²
    A = p_i a²/(b²-a²),  B = p_i a² b²/(b²-a²)
    u(r) = (1+ν)/E * [ (1-2ν) A r + B/r ]

比較する3つの近似器（loss の形は全て同じ。違うのは u(r) を出す部分だけ）:
  1. 古典 PINN            : 小さな MLP(tanh)
  2. Fourier feature PINN : sin/cos 特徴量 + 線形層（査読で必ず聞かれるベースライン）
  3. QPINN                : データ再アップロード型の変分量子回路、u(r) = w·⟨Z₀⟩ + c

微分は全て torch.autograd（シミュレータだから可能）。実機では parameter-shift が必要で
2階微分のコストが跳ね上がる — docs の「限界」節を参照。
"""

import time
import math
import numpy as np
import torch
import pennylane as qml

torch.set_default_dtype(torch.float64)

# ==============================================================
# 1. 問題設定（無次元化: a=1, b=2, p_i=1, E=1）
# ==============================================================
A_IN, B_OUT = 1.0, 2.0       # 内半径・外半径
P_IN = 1.0                    # 内圧
E_MOD, NU = 1.0, 0.3          # ヤング率・ポアソン比

_A = P_IN * A_IN**2 / (B_OUT**2 - A_IN**2)
_B = P_IN * A_IN**2 * B_OUT**2 / (B_OUT**2 - A_IN**2)
C_STIFF = E_MOD / ((1.0 + NU) * (1.0 - 2.0 * NU))   # 平面ひずみの係数


def u_exact(r):
    return (1.0 + NU) / E_MOD * ((1.0 - 2.0 * NU) * _A * r + _B / r)


def sigma_r_exact(r):
    return _A - _B / r**2


def sigma_t_exact(r):
    return _A + _B / r**2


def sigma_r_from_u(u, du, r):
    """平面ひずみ: σ_r = C[(1-ν) u' + ν u/r]"""
    return C_STIFF * ((1.0 - NU) * du + NU * u / r)


def _print_header():
    print("=" * 62)
    print("  QPINN ベンチ #1: 厚肉円筒（Lamé）")
    print("=" * 62)
    print(f"内半径 a={A_IN}, 外半径 b={B_OUT}, 内圧 p_i={P_IN}, E={E_MOD}, ν={NU}")
    print(f"解析解: σ_r = {_A:.4f} - {_B:.4f}/r²,  σ_θ = {_A:.4f} + {_B:.4f}/r²")
    print(f"検証  : σ_r(a)={sigma_r_exact(A_IN):+.4f} (=-p_i), "
          f"σ_r(b)={sigma_r_exact(B_OUT):+.4f} (=0)")


# ==============================================================
# 2. 近似器 1: 古典 PINN（MLP）
# ==============================================================
class MLPPINN(torch.nn.Module):
    def __init__(self, hidden=16, depth=2):
        super().__init__()
        layers, d_in = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(d_in, hidden), torch.nn.Tanh()]
            d_in = hidden
        layers += [torch.nn.Linear(d_in, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, r):
        return self.net(r.reshape(-1, 1)).reshape(-1)


# ==============================================================
# 3. 近似器 2: Fourier feature PINN
# ==============================================================
class FourierPINN(torch.nn.Module):
    """u(r) = Σ_k [a_k sin(ω_k (r-a)) + b_k cos(ω_k (r-a))] + c·r + d

    QPINN の角度エンコードが実質フーリエ級数である以上、これが「同じ関数クラスを持つ
    古典モデル」。公平比較のため必ず入れる。
    """

    def __init__(self, n_freq=8):
        super().__init__()
        omega = torch.arange(1, n_freq + 1) * math.pi / (B_OUT - A_IN)
        self.register_buffer("omega", omega)
        self.lin = torch.nn.Linear(2 * n_freq + 1, 1)
        torch.nn.init.normal_(self.lin.weight, std=0.1)

    def forward(self, r):
        r = r.reshape(-1, 1)
        z = self.omega * (r - A_IN)
        feats = torch.cat([torch.sin(z), torch.cos(z), r], dim=1)
        return self.lin(feats).reshape(-1)


# ==============================================================
# 4. 近似器 3: QPINN（データ再アップロード型 変分量子回路）
# ==============================================================
N_QUBITS = 4
N_LAYERS = 3
N_ENCODE = 2          # 再アップロード回数 = フーリエ周波数の上限を決める

# エンコード周波数の付け方:
#   "uniform" : 全 qubit に同じ角度 r_scaled を入れる（低周波）
#   "ladder"  : qubit w に (w+1)·r_scaled を入れる（周波数を稼ぐ定番設計）
# ladder は表現力は上がるが、初期状態で u'' が O(10) になり PDE 残差の最小化が
# 「高周波成分の抑制」に支配されて自明解に落ちる（rel_L2 ≳ 1）。
# 「エンコードが関数クラスの周波数を決める」という QPINN の性質がそのまま出る箇所で、
# 12_qpinn_ablation.py で定量化している。
FREQ_MODE = "uniform"

_DEV_CACHE = {}
_QNODE_CACHE = {}


def _get_qnode(n_qubits, n_layers, n_encode, freq_mode):
    """回路構成ごとに QNode を作ってキャッシュする（アブレーション用）"""
    key = (n_qubits, n_layers, n_encode, freq_mode)
    if key in _QNODE_CACHE:
        return _QNODE_CACHE[key]

    if n_qubits not in _DEV_CACHE:
        _DEV_CACHE[n_qubits] = qml.device("default.qubit", wires=n_qubits)
    dev = _DEV_CACHE[n_qubits]

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(r_scaled, theta):
        """r_scaled: (batch,) / theta: (n_encode, n_layers, n_qubits, 2) -> ⟨Z₀⟩ (batch,)"""
        for e in range(n_encode):
            # エンコード層
            for w in range(n_qubits):
                angle = r_scaled if freq_mode == "uniform" else (w + 1) * r_scaled
                qml.RY(angle, wires=w)
            # 変分層
            for l in range(n_layers):
                for w in range(n_qubits):
                    qml.RY(theta[e, l, w, 0], wires=w)
                    qml.RZ(theta[e, l, w, 1], wires=w)
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0))

    _QNODE_CACHE[key] = circuit
    return circuit


class QPINN(torch.nn.Module):
    def __init__(self, seed=0, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                 n_encode=N_ENCODE, freq_mode=FREQ_MODE, shift0=1.5):
        super().__init__()
        self.cfg = (n_qubits, n_layers, n_encode, freq_mode)
        self.circuit = _get_qnode(n_qubits, n_layers, n_encode, freq_mode)
        g = torch.Generator().manual_seed(seed)
        self.theta = torch.nn.Parameter(
            torch.rand(n_encode, n_layers, n_qubits, 2, generator=g) * 2 * math.pi
        )
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        # 解の平均値程度に初期化しておく（出力スケーリングの初期値は収束に効く）
        self.shift = torch.nn.Parameter(torch.tensor(float(shift0)))

    def forward(self, r):
        # r ∈ [a,b] → [0, π] に正規化（角度エンコードの定番）
        r_scaled = math.pi * (r.reshape(-1) - A_IN) / (B_OUT - A_IN)
        out = self.circuit(r_scaled, self.theta)
        return self.scale * out.reshape(-1) + self.shift


# ==============================================================
# 5. PINN の loss（3つの近似器で共通）
# ==============================================================
N_COLLOC = 64
W_BC = 10.0       # 境界条件の重み


def _d(y, r):
    return torch.autograd.grad(y, r, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def pinn_loss(model, r_colloc, r_bc):
    r = r_colloc.clone().requires_grad_(True)
    u = model(r)
    du = _d(u, r)
    d2u = _d(du, r)
    # Euler 型に整理した残差（r² を掛ける）。素の u''+u'/r-u/r² より条件数が良く、
    # 古典・量子どちらも収束が安定する。
    res = r**2 * d2u + r * du - u            # Navier（軸対称1D）
    l_pde = torch.mean(res**2)

    rb = r_bc.clone().requires_grad_(True)
    ub = model(rb)
    dub = _d(ub, rb)
    sig = sigma_r_from_u(ub, dub, rb)
    target = torch.tensor([-P_IN, 0.0])      # σ_r(a) = -p_i, σ_r(b) = 0
    l_bc = torch.mean((sig - target) ** 2)

    return l_pde + W_BC * l_bc, l_pde.item(), l_bc.item()


def rel_l2_u(model, n=200):
    rs = torch.linspace(A_IN, B_OUT, n)
    with torch.no_grad():
        pred = model(rs).numpy()
    ref = u_exact(rs.numpy())
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def rel_l2_sigma(model, n=200):
    """応力（1階微分）の誤差。変位より厳しく、応力集中評価の前段チェック。"""
    rs = torch.linspace(A_IN, B_OUT, n).requires_grad_(True)
    u = model(rs)
    du = _d(u, rs)
    sig = sigma_r_from_u(u, du, rs).detach().numpy()
    ref = sigma_r_exact(rs.detach().numpy())
    return float(np.linalg.norm(sig - ref) / np.linalg.norm(ref))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train(name, model, steps=4000, lr=0.01, log_every=1000):
    r_colloc = torch.linspace(A_IN, B_OUT, N_COLLOC)
    r_bc = torch.tensor([A_IN, B_OUT])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 終盤の学習率を落とさないと loss が暴れる（PDE 残差は高周波成分が支配的）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        loss, l_pde, l_bc = pinn_loss(model, r_colloc, r_bc)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"    step {step:4d}  loss={loss.item():.3e} "
                  f"(pde={l_pde:.2e}, bc={l_bc:.2e})  rel_L2(u)={rel_l2_u(model):.3e}",
                  flush=True)
    dt = time.time() - t0
    res = {"name": name, "n_param": n_params(model), "u_err": rel_l2_u(model),
           "s_err": rel_l2_sigma(model), "time": dt, "model": model}
    print(f"  → {name}: rel_L2(u)={res['u_err']:.3e}, rel_L2(σ_r)={res['s_err']:.3e}, "
          f"params={res['n_param']}, {dt:.1f}s", flush=True)
    return res


# ==============================================================
# 6. 実行
# ==============================================================
def main(steps=4000, q_steps=3000, q_lr=0.02, seed=0):
    _print_header()
    torch.manual_seed(seed)
    results = []

    print("\n" + "=" * 62)
    print("  1. 古典 PINN (MLP 16x2)")
    print("=" * 62)
    results.append(train("古典PINN", MLPPINN(), steps=steps))

    print("\n" + "=" * 62)
    print("  2. Fourier feature PINN (8 周波数)")
    print("=" * 62)
    results.append(train("Fourier PINN", FourierPINN(), steps=steps))

    print("\n" + "=" * 62)
    print(f"  3. QPINN ({N_QUBITS} qubit, {N_LAYERS} layer, {N_ENCODE} re-upload)")
    print("=" * 62)
    # QPINN はシミュレータのコストが重いのでステップ数を分けている
    results.append(train("QPINN", QPINN(seed=seed), steps=q_steps, lr=q_lr, log_every=500))

    print("\n" + "=" * 62)
    print("  比較表")
    print("=" * 62)
    print(f"{'モデル':<14}{'params':>8}{'rel_L2(u)':>13}{'rel_L2(σ_r)':>14}{'時間[s]':>10}")
    for r in results:
        print(f"{r['name']:<14}{r['n_param']:>8}{r['u_err']:>13.3e}"
              f"{r['s_err']:>14.3e}{r['time']:>10.1f}")

    print("\n読み方:")
    print("  - QPINN が少パラメータで同程度の u 誤差に届けば parameter efficiency の主張材料。")
    print("  - ただし σ_r（微分量）の誤差は u より悪くなるのが普通 → 応力集中（#3）の前兆。")
    print("  - 時間はシミュレータのコスト込み。qubit 数を増やすと 2ⁿ で効くので併記が必要。")
    print("\n注意: これは最小ベンチ。発表用には collocation 点数・ステップ数・")
    print("      シード複数本での平均±標準偏差が必要（docs/qpinn_rocket_research_plan.md §5）。")
    return results


def main_multiseed(n_seeds=5, steps=4000, q_steps=3000, q_lr=0.02, with_qpinn=False):
    """複数シードで平均±標準偏差を出す。

    Fourier feature PINN は初期値依存が強く（単一シードの結果は信用できない）、
    発表用の数字はこちらで取る。QPINN はシミュレータが重いので既定では外してある。
    """
    _print_header()
    specs = [("古典PINN", MLPPINN, steps, 0.01),
             ("Fourier PINN", FourierPINN, steps, 0.01)]
    if with_qpinn:
        specs.append(("QPINN", QPINN, q_steps, q_lr))

    table = {}
    for name, ctor, st, lr in specs:
        us, ss = [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            model = ctor(seed=seed) if ctor is QPINN else ctor()
            r = train(f"{name} seed={seed}", model, steps=st, lr=lr, log_every=10**9)
            us.append(r["u_err"]); ss.append(r["s_err"])
            n_p = r["n_param"]
        table[name] = (n_p, np.mean(us), np.std(us), np.mean(ss), np.std(ss))

    print("\n" + "=" * 62)
    print(f"  比較表（{n_seeds} シード, 平均 ± 標準偏差）")
    print("=" * 62)
    print(f"{'モデル':<14}{'params':>8}{'rel_L2(u)':>24}{'rel_L2(σ_r)':>24}")
    for name, (n_p, mu, su, ms, ss_) in table.items():
        print(f"{name:<14}{n_p:>8}{mu:>13.3e} ± {su:<8.1e}{ms:>13.3e} ± {ss_:<8.1e}")
    return table


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000, help="古典モデルの学習ステップ数")
    ap.add_argument("--q-steps", type=int, default=3000, help="QPINN の学習ステップ数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=0,
                    help="N>0 なら N シードで平均±標準偏差を出す（既定は単一シード）")
    ap.add_argument("--multiseed-qpinn", action="store_true",
                    help="複数シードモードに QPINN も含める（重い）")
    args = ap.parse_args()
    if args.seeds > 0:
        main_multiseed(n_seeds=args.seeds, steps=args.steps, q_steps=args.q_steps,
                       with_qpinn=args.multiseed_qpinn)
    else:
        main(steps=args.steps, q_steps=args.q_steps, seed=args.seed)
