"""
Trained QPINNs with different input embeddings vs parameter-matched classical PINNs (Muramatsu's suggestion, 2026-11-17).

Embed-once circuits as in Berger, Hosters & Moeller, Sci. Rep. 15, 18823 (2025) (TE-QPINN):
    |0..0> -> prod_q Ry(angle_q(x)) -> L variational layers (Rx Ry Rz on every qubit + CNOT ring) -> <Z_q>
    u = head . <Z> + bias
Models
  Q-fixed    fixed embedding: angle_q = scale_q * a_c(q)   (Burgers: tower Chebyshev 2 (k+1) arccos a, as Kyriienko et al.)
  Q-TE       trainable embedding: an MLP (hidden 10, 10, tanh) gives phi_q(a); angle_q = phi_q(a) * a_c(q)   (Berger et al.)
  TE-twin    classical twin of Q-TE: the fixed circuit is replaced by a linear map on prod_q (1, cos angle_q, sin angle_q)
             (3^N terms), which spans every function a fixed embed-once circuit can output for these angles
  PINN-match tanh MLP with two hidden layers sized to the Q-TE parameter count
  PINN       tanh MLP 4 x 64 (reference size)
a = network input: Allen-Cahn (cos pi x, sin pi x, 2t - 1) (periodic built in); Burgers (x, t) rescaled to [-0.95, 0.95];
c(q) = q mod dim(a), k = q div dim(a).

  python 66_qpinn_embeddings.py --pde burgers --opt lbfgs --seeds 3      # reproduce the TE-QPINN paper setting first
  python 66_qpinn_embeddings.py --pde ac --opt adam --steps 20000 --seeds 5
"""
import argparse, json, os, time, importlib.util
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)
H = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ac", os.path.join(H, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac)
D_AC, NU_B = 1e-4, 0.01 / np.pi


# ---------------- inputs ----------------
def make_inp(pde):
    if pde == "ac":
        return lambda x, t: jnp.stack([jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x), 2 * t - 1])
    return lambda x, t: jnp.stack([0.95 * x, 0.95 * (2 * t / 0.95 - 1)])        # x in [-1, 1], t in [0, 0.95]


# ---------------- circuit (state vector) ----------------
def _apply1(psi, U, q, N):
    psi = psi.reshape((2,) * N)
    return jnp.moveaxis(jnp.tensordot(U, psi, axes=([1], [q])), 0, q).reshape(-1)


def _rx(a):
    c, s = jnp.cos(a / 2), jnp.sin(a / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(a):
    c, s = jnp.cos(a / 2), jnp.sin(a / 2)
    return jnp.array([[c, -s], [s, c]], dtype=complex)


def _rz(a):
    return jnp.array([[jnp.exp(-0.5j * a), 0], [0, jnp.exp(0.5j * a)]], dtype=complex)


def _cnot(psi, c, t, N):
    psi = psi.reshape((2,) * N)
    idx = [slice(None)] * N; idx[c] = 1
    sub = jnp.flip(psi[tuple(idx)], axis=t if t < c else t - 1)
    return psi.at[tuple(idx)].set(sub).reshape(-1)


def circuit_z(angles, th, N, L):
    """embed once with Ry(angles), then L layers of Rx Ry Rz + CNOT ring; returns <Z_q>"""
    psi = jnp.zeros(2**N, dtype=complex).at[0].set(1.0)
    for q in range(N):
        psi = _apply1(psi, _ry(angles[q]), q, N)
    for l in range(L):
        for q in range(N):
            psi = _apply1(psi, _rx(th[l, q, 0]), q, N)
            psi = _apply1(psi, _ry(th[l, q, 1]), q, N)
            psi = _apply1(psi, _rz(th[l, q, 2]), q, N)
        if N > 1:
            for q in range(N):
                psi = _cnot(psi, q, (q + 1) % N, N)
    prob = jnp.abs(psi.reshape((2,) * N)) ** 2
    return jnp.array([jnp.sum(jnp.take(prob, 0, axis=q)) - jnp.sum(jnp.take(prob, 1, axis=q)) for q in range(N)])


# ---------------- MLP ----------------
def mlp_init(key, widths):
    ps = []
    for i in range(len(widths) - 1):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (widths[i], widths[i + 1])) * np.sqrt(1 / widths[i]), jnp.zeros(widths[i + 1])))
    return ps


def mlp(ps, h):
    for W, b in ps[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = ps[-1]
    return h @ W + b


def n_params(p):
    return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(p)))


def match_hidden(dim, target):
    """two hidden layers of width h: dim*h + h + h*h + h + h + 1 parameters"""
    hs = np.arange(2, 200)
    return int(hs[np.argmin(np.abs(dim * hs + hs + hs * hs + hs + hs + 1 - target))])


# ---------------- models ----------------
def build(name, key, pde, N, L, te_hidden, target_params=None):
    inp = make_inp(pde); dim = 3 if pde == "ac" else 2
    comp = np.arange(N) % dim; k = np.arange(N) // dim
    k1, k2, k3 = jax.random.split(key, 3)
    qpart = lambda: {"th": jax.random.uniform(k1, (L, N, 3), minval=0, maxval=2 * np.pi),
                     "head": 0.5 * jax.random.normal(k2, (N,)), "bias": jnp.zeros(())}
    if name == "Q-fixed":
        p = qpart()
        if pde == "burgers":      # tower Chebyshev (Kyriienko et al.): Ry(2 (k+1) arccos a)
            ang = lambda a: 2 * (k + 1) * jnp.arccos(a[comp])
        else:                     # a = cos/sin pi x is +-1 inside the domain, where arccos is not smooth: linear tower
            ang = lambda a: (k + 1) * (np.pi / 2) * a[comp]
        f = lambda p, x, t: p["head"] @ circuit_z(ang(inp(x, t)), p["th"], N, L) + p["bias"]
    elif name in ("Q-TE", "TE-twin"):
        p = {"emb": mlp_init(k3, [dim] + te_hidden + [N])}
        ang = lambda p, a: mlp(p["emb"], a) * a[comp]
        if name == "Q-TE":
            p.update(qpart())
            f = lambda p, x, t: p["head"] @ circuit_z(ang(p, inp(x, t)), p["th"], N, L) + p["bias"]
        else:
            p.update({"C": 0.1 * jax.random.normal(k1, (3**N,)), "bias": jnp.zeros(())})

            def f(p, x, t):
                g = ang(p, inp(x, t)); v = jnp.ones((1,))
                for q in range(N):
                    v = jnp.kron(v, jnp.stack([jnp.ones(()), jnp.cos(g[q]), jnp.sin(g[q])]))
                return p["C"] @ v + p["bias"]
    else:
        if name == "PINN":
            widths = [dim, 64, 64, 64, 64, 1]
        else:                      # PINN-match
            h = match_hidden(dim, target_params); widths = [dim, h, h, 1]
        p = {"mlp": mlp_init(k3, widths)}
        f = lambda p, x, t: mlp(p["mlp"], inp(x, t))[0]
    return p, f


# ---------------- problems ----------------
def burgers_reference(x_eval, T_out, n_modes=2048, dt=2e-5):
    """u_t + u u_x = nu u_xx, u(0) = -sin pi x: odd and 2-periodic, so u(+-1) = 0 holds and a periodic spectral solver applies"""
    xs = -1 + 2 * np.arange(n_modes) / n_modes
    kk = np.fft.fftfreq(n_modes, d=2 / n_modes) * 2 * np.pi
    Lop = -NU_B * kk**2
    E, E2 = np.exp(dt * Lop), np.exp(dt * Lop / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, 1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, 1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, 1))
    Nf = lambda v: -0.5j * kk * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
    v = np.fft.fft(-np.sin(np.pi * xs)); out = {}
    want = {int(round(t / dt)): t for t in T_out}
    for n in range(1, int(round(max(T_out) / dt)) + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n in want:
            out[want[n]] = np.real(np.exp(1j * np.outer(x_eval + 1, kk)) @ (v / n_modes))
    return out


def setup(pde, sd, n_col):
    """collocation sets, residual and reference"""
    g = np.random.default_rng(100 + sd)
    if pde == "ac":
        xc = g.uniform(-1, 1, n_col); tc = g.uniform(0, 1, n_col)
        x0 = np.linspace(-1, 1, 512, endpoint=False)
        sets = {"ic": (x0, np.zeros_like(x0), x0**2 * np.cos(np.pi * x0), 100.0)}
        T_out = [0.25, 0.5, 1.0]
    else:                      # Berger et al.: 100 x 50 grid in (x, t), weight 10 on IC and BC
        X, Tt = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(0, 0.95, 50))
        xc, tc = X.ravel(), Tt.ravel()
        tb = np.linspace(0, 0.95, 50); x0 = np.linspace(-1, 1, 100)
        sets = {"ic": (x0, np.zeros_like(x0), -np.sin(np.pi * x0), 10.0),
                "bc": (np.concatenate([-np.ones(50), np.ones(50)]), np.concatenate([tb, tb]), np.zeros(100), 10.0)}
        T_out = [0.25, 0.5, 0.75, 0.95]
    return xc, tc, sets, T_out


def residual(pde, f):
    def r(p, x, t):
        u = f(p, x, t)
        ut = jax.grad(f, argnums=2)(p, x, t)
        ux = jax.grad(f, argnums=1)(p, x, t)
        uxx = jax.grad(jax.grad(f, argnums=1), argnums=1)(p, x, t)
        if pde == "ac":
            return ut - D_AC * uxx + 5 * u**3 - 5 * u
        return ut + u * ux - NU_B * uxx
    return r


# ---------------- optimisers ----------------
def adam(vg, p, steps, lr):
    m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p)
    b1, b2, eps = 0.9, 0.999, 1e-8; hist = []
    for i in range(1, steps + 1):
        val, g = vg(p)
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, g)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
        lr_i = lr * min(1.0, 0.5 * (1 + np.cos(np.pi * i / steps)) + 0.01)
        p = jax.tree_util.tree_map(lambda p, m, v: p - lr_i * (m / (1 - b1**i)) / (jnp.sqrt(v / (1 - b2**i)) + eps), p, m, v)
        if i % 1000 == 0 or i == 1:
            hist.append((i, float(val)))
    return p, hist


def lbfgs(vg, p, steps):
    from scipy.optimize import minimize
    w0, unravel = ravel_pytree(p); hist = []

    def fun(w):
        val, g = vg(unravel(jnp.asarray(w)))
        return float(val), np.asarray(ravel_pytree(g)[0], dtype=float)

    def cb(w):
        if len(hist) % 100 == 0:
            hist.append((len(hist), fun(w)[0]))
        else:
            hist.append(None)
    res = minimize(fun, np.asarray(w0), jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": steps, "maxfun": 2 * steps, "maxcor": 50, "ftol": 0, "gtol": 1e-12})
    return unravel(jnp.asarray(res.x)), [h for h in hist if h] + [(res.nit, float(res.fun))]


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pde", default="ac", choices=["ac", "burgers"])
    ap.add_argument("--models", default="Q-fixed,Q-TE,TE-twin,PINN-match,PINN")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--te_hidden", default="10,10")
    ap.add_argument("--opt", default="adam", choices=["adam", "lbfgs"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n_col", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bench", action="store_true", help="only time one gradient step per model")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/qpinn_emb_{a.pde}_{a.opt}.json"
    te_hidden = [int(s) for s in a.te_hidden.split(",")]
    M = 1024; xg = -1 + 2 * (np.arange(M) + 0.5) / M
    _, _, _, T_out = setup(a.pde, 0, a.n_col)
    t0 = time.time()
    ref = ac.reference(D_AC, xg, T_out) if a.pde == "ac" else burgers_reference(xg, T_out)
    print(f"reference {time.time() - t0:.0f}s", flush=True)
    # parameter target for PINN-match: the Q-TE model
    te_count = n_params(build("Q-TE", jax.random.PRNGKey(0), a.pde, a.N, a.L, te_hidden)[0])
    rows = []
    for name in a.models.split(","):
        for sd in range(a.seeds):
            p, f = build(name, jax.random.PRNGKey(sd), a.pde, a.N, a.L, te_hidden, te_count)
            xc, tc, sets, _ = setup(a.pde, sd, a.n_col)
            r = residual(a.pde, f)
            XC, TC = jnp.asarray(xc), jnp.asarray(tc)
            S = {k: tuple(jnp.asarray(v) for v in s[:3]) + (s[3],) for k, s in sets.items()}

            # Each loss term is jitted on its own with the points as arguments and the gradients are summed:
            # jitting the PDE term together with an (x, t)-vmapped term was 3x slower on GPU (see 56_ac_vqpinn.py).
            def loss_pde(p, xs, ts):
                return jnp.mean(jax.vmap(r, (None, 0, 0))(p, xs, ts) ** 2)

            def loss_fit(p, xs, ts, us, w):
                return w * jnp.mean((jax.vmap(f, (None, 0, 0))(p, xs, ts) - us) ** 2)
            vgs = [(jax.jit(jax.value_and_grad(loss_pde)), (XC, TC))] + \
                  [(jax.jit(jax.value_and_grad(loss_fit)), s) for s in S.values()]

            def vg(p):
                o = [g(p, *ar) for g, ar in vgs]
                return sum(v for v, _ in o), jax.tree_util.tree_map(lambda *gs: sum(gs), *[g for _, g in o])
            if a.bench:        # time one gradient step and stop
                jax.block_until_ready(vg(p)); t0 = time.time()
                for _ in range(10):
                    jax.block_until_ready(vg(p))
                print(f"bench {a.pde} {name} params={n_params(p)}: {(time.time() - t0) / 10 * 1000:.0f} ms/step "
                      f"on {jax.devices()[0]}", flush=True)
                break
            t0 = time.time()
            p, hist = adam(vg, p, a.steps, a.lr) if a.opt == "adam" else lbfgs(vg, p, a.steps)
            tt = time.time() - t0
            ev = jax.jit(jax.vmap(f, (None, 0, None)))
            errs = {t: float(np.linalg.norm(np.asarray(ev(p, jnp.asarray(xg), t)) - ref[t]) / np.linalg.norm(ref[t]))
                    for t in T_out}
            rows.append({"pde": a.pde, "model": name, "seed": sd, "params": n_params(p), "N": a.N, "L": a.L,
                         "opt": a.opt, "steps": a.steps, "time_s": tt, "loss": hist[-1][1], "loss_hist": hist,
                         **{f"t{t}": e for t, e in errs.items()}})
            print(f"{a.pde} {a.opt} {name:10s} seed={sd} params={n_params(p)} loss={hist[-1][1]:.2e} " +
                  " ".join(f"t={t}:{e:.1e}" for t, e in errs.items()) + f" {tt:.0f}s", flush=True)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
