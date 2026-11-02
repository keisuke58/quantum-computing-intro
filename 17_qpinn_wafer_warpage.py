# -*- coding: utf-8 -*-
"""
17 - ウェーハの反り（warpage）: 薄い円板の熱変形を QPINN / PINN で解く

半導体（研削・研磨）への応用に向けた最初のベンチ。研削熱で板厚方向に温度差が生じると
ウェーハが反る。その最小モデルとして、拘束のない円板に板厚方向の線形な温度分布

    T(z) = ΔT · z / h        （z ∈ [−h/2, h/2]，上面が +ΔT/2，下面が −ΔT/2）

を与える。このとき次が 3 次元で厳密に成り立つ（薄板近似は不要）:

- 熱ひずみ α T δ_ij は適合条件を満たすので，**応力は恒等的に 0**
- 変位は球面状の曲げ:
      u_x = κ x z,   u_y = κ y z,   u_z = −κ (x² + y²)/2 + κ z²/2 + c
- 曲率 κ = α ΔT / h，**反り量（中心と外周の高さの差） w = κ R² / 2**
  （c は剛体並進を固定するための定数で，反り量には影響しない）

厳密解がアスペクト比によらず成り立つので，「板を薄くすると解けなくなるか」だけを
純粋に測れる。実際のウェーハは直径 300 mm・厚さ 50 µm で R/h ≈ 3000 に達する。

## 何を比べるか

1. **アスペクト比 R/h**: 1 → 10 → 100 → 1000 と薄くしていく
2. **変位の表し方**:
   - none        : 入出力をそのまま使う
   - anisotropic : 入力を半径方向は R，板厚方向は h/2 で別々に正規化し，
                   出力も面内変位 α ΔT R・面外変位 α ΔT R²/h の代表値で割る
   - kirchhoff   : Kirchhoff 板の運動学（u_x = −z ∂w/∂x など）を組み込む
   最初は anisotropic で足りると考えたが，R/h = 100 で none・anisotropic とも完全に失敗した．
   原因は FEM の薄板で知られる「せん断ロッキング」と同じで，kirchhoff がその対策
3. **モデル**: 古典 MLP，ハイブリッド QPINN（14_qpinn_dem_thermo.py と同じ構成）

## 損失（DEM：ひずみエネルギー最小化）

    Π[u] = ∫ W(ε(u) − α T I) dV  ＋ 剛体モードの罰則

自由境界（どこも拘束しない）は DEM では自然境界条件になるので罰則が要らない。
厳密解では W ≡ 0 なので，Π の最小値は 0。

## 評価指標

- 反り量 w の相対誤差（工学的に一番重要）
- 変位の相対 L2 誤差（面内 u_x, u_y と面外 u_z を分けて）
- 疑似応力: 本来 0 の応力が RMS でどれだけ出たか（E α ΔT で無次元化）
"""

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "dem_thermo", os.path.join(_HERE, "14_qpinn_dem_thermo.py"))
dem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dem)

# ==============================================================
# 1. 問題設定（無次元: R = 1, E = 1, α = 1, ΔT = 1）
# ==============================================================
R_PLATE = 1.0
E_MOD, NU, ALPHA, DT = 1.0, 0.3, 1.0, 1.0
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU = E_MOD / (2 * (1 + NU))


def thickness(aspect):
    return R_PLATE / aspect


def temperature(x, h):
    """T(z) = ΔT z / h"""
    return DT * x[:, 2] / h


def curvature(h):
    return ALPHA * DT / h


def warpage_exact(h):
    return curvature(h) * R_PLATE**2 / 2


def u_exact(x, h):
    """厳密解の変位（体積平均で u_z の平均が 0 になるよう定数 c を選ぶ）"""
    k = curvature(h)
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    c = k / 2 * (R_PLATE**2 / 2 - h**2 / 12)   # mean(x²+y²)=R²/2, mean(z²)=h²/12
    return torch.stack([k * X * Z, k * Y * Z, -k / 2 * (X**2 + Y**2) + k / 2 * Z**2 + c], dim=1)


def sample_plate(n, h, seed=0):
    """円板（半径 R，厚さ h）の内部を体積一様にサンプリング"""
    g = torch.Generator().manual_seed(seed)
    r = R_PLATE * torch.sqrt(torch.rand(n, generator=g))
    th = 2 * math.pi * torch.rand(n, generator=g)
    z = h * (torch.rand(n, generator=g) - 0.5)
    return torch.stack([r * torch.cos(th), r * torch.sin(th), z], dim=1)


# ==============================================================
# 2. 厳密解の検証
# ==============================================================
def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def strain_of(fn, x):
    x = x.clone().requires_grad_(True)
    u = fn(x)
    G = torch.stack([_grad(u[:, i], x) for i in range(3)], dim=1)
    return 0.5 * (G + G.transpose(1, 2)), x


def stress_from_strain(eps, T):
    eye = torch.eye(3).expand(eps.shape[0], 3, 3)
    e = eps - ALPHA * T.reshape(-1, 1, 1) * eye
    tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
    return LAM * tr.reshape(-1, 1, 1) * eye + 2 * MU * e


def verify_exact_solution(aspects=(1, 10, 100, 1000), verbose=True):
    """厳密解が (1) 応力ゼロ (2) 反り量 κR²/2 を満たすことを自動微分で確認する"""
    ok = True
    for a in aspects:
        h = thickness(a)
        x = sample_plate(2000, h, seed=7)
        eps, xr = strain_of(lambda p: u_exact(p, h), x)
        S = stress_from_strain(eps, temperature(xr, h))
        s_rms = float(S.detach().pow(2).sum(dim=(1, 2)).mean().sqrt()) / (E_MOD * ALPHA * DT)
        # 反り量: 中心と外周（z=0）の u_z の差
        o = torch.zeros(1, 3)
        rim = torch.tensor([[R_PLATE, 0.0, 0.0]])
        w = float(u_exact(o, h)[0, 2] - u_exact(rim, h)[0, 2])
        w_err = abs(w - warpage_exact(h)) / warpage_exact(h)
        ok &= s_rms < 1e-10 and w_err < 1e-12
        if verbose:
            print(f"  R/h = {a:<5}  応力 RMS = {s_rms:.1e}（0 であるべき）  "
                  f"反り量 = {w:.4e}（理論 {warpage_exact(h):.4e}）")
    if verbose:
        print(f"  → {'OK' if ok else '*** 失敗 ***'}")
    return ok


# ==============================================================
# 3. 座標スケーリング付きのモデル
# ==============================================================
class Scaled(torch.nn.Module):
    """入力を方向別に正規化し，出力を方向別の代表値で戻すラッパ。

    薄い板では z の範囲が x, y の 1/aspect しかなく，そのままでは
    ネットワークにとって z 方向の変化がほぼ見えない。面外変位 u_z は面内変位の
    約 aspect 倍の大きさになるので，出力側の尺度も揃えておく。
    """

    def __init__(self, net, h, mode):
        super().__init__()
        self.net = net
        if mode == "anisotropic":
            self.register_buffer("s_in", torch.tensor([R_PLATE, R_PLATE, h / 2]))
            s_inplane = ALPHA * DT * R_PLATE
            s_out = ALPHA * DT * R_PLATE**2 / h
            self.register_buffer("s_out", torch.tensor([s_inplane, s_inplane, s_out]))
        else:
            self.register_buffer("s_in", torch.ones(3))
            self.register_buffer("s_out", torch.ones(3))

    def forward(self, x):
        return self.net(x / self.s_in) * self.s_out


class Kirchhoff(torch.nn.Module):
    """Kirchhoff 板の運動学を組み込んだモデル（せん断ロッキング対策）。

    薄い板では，せん断ひずみ ε_xz = ½(∂u_x/∂z + ∂u_z/∂x) が 0 になるには，
    大きさが真値の R/h 倍もある 2 つの項が打ち消し合う必要がある．一般の 3D 変位を
    出力するネットワークではこれが満たせず，「曲がらない」自明解に落ちる
    （FEM の薄板で知られるせん断ロッキングと同じ現象．R/h = 100 で実測した）．

    そこでネットワークは板の中立面 (x, y) の関数だけを出力し，3D 変位を

        u_x = u0_x − z ∂w/∂x,   u_y = u0_y − z ∂w/∂y,   u_z = w + (z²/2) φ

    として組み立てる．ε_xz は z² の項だけが残り，ほぼ 0 になる．
    厳密解は w = −κ(x²+y²)/2 + c，φ = κ，u0 = 0 で表現できる．
    """

    def __init__(self, net, h):
        super().__init__()
        self.net = net
        s_inplane = ALPHA * DT * R_PLATE
        s_w = ALPHA * DT * R_PLATE**2 / h
        s_phi = ALPHA * DT / h
        self.register_buffer("s_out", torch.tensor([s_inplane, s_inplane, s_w, s_phi]))

    def forward(self, x):
        with torch.enable_grad():
            xy = x[:, :2] if x.requires_grad else x[:, :2].detach().requires_grad_(True)
            z = x[:, 2]
            out = self.net(xy / R_PLATE) * self.s_out
            w = out[:, 2]
            gw = torch.autograd.grad(w, xy, grad_outputs=torch.ones_like(w), create_graph=True)[0]
            return torch.stack([out[:, 0] - z * gw[:, 0],
                                out[:, 1] - z * gw[:, 1],
                                w + z**2 / 2 * out[:, 3]], dim=1)


def make_model(kind, h, scaling, seed=0, n_qubits=4):
    d_in, d_out = (2, 4) if scaling == "kirchhoff" else (3, 3)
    if kind == "mlp":
        net = dem.MLP(d_in, d_out)
    elif kind == "small-mlp":
        net = dem.MLP(d_in, d_out, hidden=10, depth=2)
    elif kind == "qpinn":
        net = dem.QPINNHybrid(d_in, d_out, seed=seed, n_qubits=n_qubits)
    else:
        raise ValueError(kind)
    if scaling == "kirchhoff":
        return Kirchhoff(net, h)
    return Scaled(net, h, scaling)


# ==============================================================
# 4. 損失（DEM）と評価
# ==============================================================
def dem_loss(model, x, h, w_rb=100.0):
    eps, xr = strain_of(model, x)
    T = temperature(xr, h).detach()
    eye = torch.eye(3).expand(x.shape[0], 3, 3)
    e = eps - ALPHA * T.reshape(-1, 1, 1) * eye
    tr = e[:, 0, 0] + e[:, 1, 1] + e[:, 2, 2]
    W = 0.5 * LAM * tr**2 + MU * (e**2).sum(dim=(1, 2))
    energy = W.mean() / (E_MOD * (ALPHA * DT) ** 2)      # 無次元化した平均エネルギー密度

    # 剛体モード（並進 3・回転 3）を固定．変位の代表値で無次元化して重みを揃える
    u = model(x)
    s = ALPHA * DT * R_PLATE**2 / h
    pen = ((u.mean(dim=0) / s) ** 2).sum() + ((torch.cross(x, u, dim=1).mean(dim=0) / s) ** 2).sum()
    return energy + w_rb * pen, energy.item(), pen.item()


def evaluate(model, h):
    x = sample_plate(4000, h, seed=999)
    with torch.no_grad():
        up, ue = model(x), u_exact(x, h)
    inpl = float((up[:, :2] - ue[:, :2]).norm() / ue[:, :2].norm())
    outp = float((up[:, 2] - ue[:, 2]).norm() / ue[:, 2].norm())
    with torch.no_grad():
        o = torch.zeros(1, 3)
        rim = torch.stack([R_PLATE * torch.cos(torch.linspace(0, 2 * math.pi, 73)[:-1]),
                           R_PLATE * torch.sin(torch.linspace(0, 2 * math.pi, 73)[:-1]),
                           torch.zeros(72)], dim=1)
        w = float(model(o)[0, 2] - model(rim)[:, 2].mean())
    w_err = abs(w - warpage_exact(h)) / warpage_exact(h)
    eps, xr = strain_of(model, x[:1000])
    S = stress_from_strain(eps, temperature(xr, h))
    s_rms = float(S.detach().pow(2).sum(dim=(1, 2)).mean().sqrt()) / (E_MOD * ALPHA * DT)
    return {"w_err": w_err, "u_inplane": inpl, "u_outplane": outp, "stress_rms": s_rms, "w": w}


def train(model, h, steps=3000, lr=0.01, lbfgs=0, n_colloc=1024, resample_every=100):
    pts = {"k": 0}

    def resample():
        pts["k"] += 1
        pts["x"] = sample_plate(n_colloc, h, seed=1000 + pts["k"])
    resample()

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.01)
    for s in range(steps):
        if s > 0 and s % resample_every == 0:
            resample()
        opt.zero_grad()
        loss = dem_loss(model, pts["x"], h)[0]
        loss.backward()
        opt.step()
        sched.step()
    if lbfgs > 0:
        lopt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50,
                                 line_search_fn="strong_wolfe")

        def closure():
            lopt.zero_grad()
            l = dem_loss(model, pts["x"], h)[0]
            l.backward()
            return l
        for it in range(lbfgs):
            if it > 0:
                resample()
            if not torch.isfinite(lopt.step(closure)):
                break
    return time.time() - t0


# ==============================================================
# 5. 実行
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspects", default="1,10,100,1000", help="R/h をカンマ区切りで")
    ap.add_argument("--scalings", default="none,anisotropic,kirchhoff",
                    help="none / anisotropic（方向別スケーリング）/ kirchhoff（板の運動学を組み込む）")
    ap.add_argument("--models", default="mlp,small-mlp,qpinn")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--c-steps", type=int, default=3000)
    ap.add_argument("--q-steps", type=int, default=1000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    print("厳密解の検証:")
    if not verify_exact_solution() or a.verify_only:
        return

    rows = []
    for kind in a.models.split(","):
        for scaling in a.scalings.split(","):
            for asp in [float(s) for s in a.aspects.split(",")]:
                h = thickness(asp)
                for seed in range(a.seeds):
                    torch.manual_seed(seed)
                    m = make_model(kind, h, scaling, seed=seed)
                    is_q = kind == "qpinn"
                    t = train(m, h, steps=a.q_steps if is_q else a.c_steps,
                              lbfgs=0 if is_q else 30)
                    r = evaluate(m, h)
                    r.update({"model": kind, "scaling": scaling, "aspect": asp, "seed": seed,
                              "time": t, "n_param": sum(p.numel() for p in m.parameters())})
                    rows.append(r)
                    print(f"{kind:<10} {scaling:<12} R/h={asp:<6g} seed={seed}  "
                          f"反り誤差={r['w_err']:.2e}  u面内={r['u_inplane']:.2e}  "
                          f"u面外={r['u_outplane']:.2e}  疑似応力={r['stress_rms']:.2e}  {t:.0f}s",
                          flush=True)
                    if a.json:
                        json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)

    print("\n=== まとめ（反り量の相対誤差，シード平均） ===")
    for kind in a.models.split(","):
        for scaling in a.scalings.split(","):
            line = f"{kind:<10} {scaling:<12}"
            for asp in [float(s) for s in a.aspects.split(",")]:
                sel = [r["w_err"] for r in rows
                       if r["model"] == kind and r["scaling"] == scaling and r["aspect"] == asp]
                if sel:
                    line += f"  R/h={asp:g}: {np.mean(sel):.2e}"
            print(line)


if __name__ == "__main__":
    main()
