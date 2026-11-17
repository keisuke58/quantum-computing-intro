"""
Baselines for Allen-Cahn (Raissi 2019 Eq. 12): space-time PINNs trained by Adam.
  - VQPINN: variational quantum circuit (data re-uploading, trainable embedding), JAX state vector
  - PINN:   classical tanh MLP (a standard size and a size matched to the VQPINN)
Same loss for both: mean PDE residual^2 + w_ic * mean IC error^2; periodicity is built in through
(cos pi x, sin pi x). Errors are measured against the spectral reference of 52_allen_cahn_qerc.py.

  python 56_ac_vqpinn.py --models VQPINN,PINN-small,PINN --steps 20000 --seeds 3
"""
import argparse, json, os, time, importlib.util
import numpy as np
import jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ac", os.path.join(_here, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac)
D = 1e-4


def inp(x, t):
    return jnp.stack([jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x), 2 * t - 1])


# ---------------- variational quantum circuit ----------------
def _apply1(psi, U, q, N):
    psi = psi.reshape((2,) * N)
    psi = jnp.moveaxis(jnp.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def _ry(a):
    c, s = jnp.cos(a / 2), jnp.sin(a / 2)
    return jnp.array([[c, -s], [s, c]], dtype=complex)


def _rz(a):
    return jnp.array([[jnp.exp(-0.5j * a), 0], [0, jnp.exp(0.5j * a)]], dtype=complex)


def _cnot(psi, c, t, N):
    psi = psi.reshape((2,) * N)
    idx1 = [slice(None)] * N; idx1[c] = 1
    sub = psi[tuple(idx1)]
    tt = t if t < c else t - 1
    sub = jnp.flip(sub, axis=tt)
    psi = psi.at[tuple(idx1)].set(sub)
    return psi.reshape(-1)


def vqc_init(key, N, L):
    k1, k2, k3 = jax.random.split(key, 3)
    return {"We": jax.random.normal(k1, (L, N, 3)), "be": jnp.zeros((L, N)),
            "th": jax.random.uniform(k2, (L, N, 2), minval=0, maxval=2 * np.pi),
            "head": 0.5 * jax.random.normal(k3, (N,)), "bias": jnp.zeros(())}


def vqc_u(p, x, t, N, L):
    a = inp(x, t)
    psi = jnp.zeros(2**N, dtype=complex).at[0].set(1.0)
    for l in range(L):
        for q in range(N):
            psi = _apply1(psi, _ry(p["We"][l, q] @ a + p["be"][l, q]), q, N)       # trainable encoding
            psi = _apply1(psi, _rz(p["th"][l, q, 0]), q, N)
            psi = _apply1(psi, _ry(p["th"][l, q, 1]), q, N)
        for q in range(N):
            psi = _cnot(psi, q, (q + 1) % N, N)
    prob = jnp.abs(psi.reshape((2,) * N)) ** 2
    z = jnp.array([jnp.sum(jnp.take(prob, 0, axis=q)) - jnp.sum(jnp.take(prob, 1, axis=q)) for q in range(N)])
    return p["head"] @ z + p["bias"]


# ---------------- classical MLP ----------------
def mlp_init(key, widths):
    ps = []
    for i in range(len(widths) - 1):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (widths[i], widths[i + 1])) * np.sqrt(1 / widths[i]), jnp.zeros(widths[i + 1])))
    return ps


def mlp_u(ps, x, t):
    h = inp(x, t)
    for W, b in ps[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = ps[-1]
    return (h @ W + b)[0]


def n_params(p):
    return int(sum(np.size(v) for v in jax.tree_util.tree_leaves(p)))


# ---------------- training ----------------
def split_value_and_grad(terms):
    """terms: [(loss_fn, args), ...]. Each term is jitted on its own and the values and gradients are summed.
    Jitting the PDE and data terms together made the data-assisted VQPINN step 3x slower on GPU
    (copaam 3090: 570 ms PDE, 37 ms data, 1789 ms joint); separately the cost is the sum."""
    vgs = [(jax.jit(jax.value_and_grad(f)), ar) for f, ar in terms]

    def vg(p):
        out = [f(p, *ar) for f, ar in vgs]
        val = sum(o[0] for o in out)
        g = jax.tree_util.tree_map(lambda *gs: sum(gs), *[o[1] for o in out])
        return val, g
    return vg


def adam(terms, p, steps, lr, log_every=1000):
    """terms: [(loss_fn, args), ...]; point arrays are passed as jit arguments (not baked in as constants)"""
    vg = split_value_and_grad(terms)
    m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    for i in range(1, steps + 1):
        val, g = vg(p)
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, g)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, g)
        lr_i = lr * min(1.0, 0.5 * (1 + np.cos(np.pi * i / steps)) + 0.01)      # cosine decay
        p = jax.tree_util.tree_map(lambda p, m, v: p - lr_i * (m / (1 - b1**i)) / (jnp.sqrt(v / (1 - b2**i)) + eps), p, m, v)
        if i % log_every == 0 or i == 1:
            hist.append((i, float(val)))
    return p, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="VQPINN,PINN-small,PINN")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n_col", type=int, default=4000)
    ap.add_argument("--w_ic", type=float, default=100.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_data", type=int, default=0, help="data points: n_data/10 random x at each t = 0.1, ..., 1.0")
    ap.add_argument("--w_data", type=float, default=100.0)
    ap.add_argument("--bench", action="store_true", help="only time one gradient step with / without the data term")
    ap.add_argument("--out", default="results/ac_vqpinn.json")
    a = ap.parse_args()
    M = 1024; xg = -1 + 2 * (np.arange(M) + 0.5) / M
    t_out = [0.25, 0.5, 1.0]
    t0 = time.time(); ref = ac.reference(D, xg, t_out); print(f"reference {time.time()-t0:.0f}s", flush=True)
    if a.n_data:   # same data points as 52_allen_cahn_qerc.py --n_data
        t_data = [round(0.1 * k, 6) for k in range(1, 11)]
        xd1 = np.sort(np.random.default_rng(7).uniform(-1, 1, a.n_data // 10))
        udd = ac.reference(D, xd1, t_data)
        xd = jnp.asarray(np.tile(xd1, len(t_data))); td = jnp.asarray(np.repeat(t_data, len(xd1)))
        ud = jnp.asarray(np.concatenate([udd[t] for t in t_data]))
    rows = []
    for name in a.models.split(","):
        for sd in range(a.seeds):
            key = jax.random.PRNGKey(sd)
            if name == "VQPINN":
                p = vqc_init(key, a.N, a.L); uf = lambda p, x, t: vqc_u(p, x, t, a.N, a.L)
            else:
                if name == "PINN":
                    widths = [3, 64, 64, 64, 64, 1]
                else:   # PINN-small: about as many parameters as the VQPINN
                    nq = n_params(vqc_init(key, a.N, a.L)); h = max(4, int((-(4 + 1) + np.sqrt(25 + 4 * nq)) / 2))
                    widths = [3, h, h, 1]
                p = mlp_init(key, widths); uf = mlp_u
            g = np.random.default_rng(100 + sd)
            xc = jnp.asarray(g.uniform(-1, 1, a.n_col)); tc = jnp.asarray(g.uniform(0, 1, a.n_col))
            x0 = jnp.asarray(np.linspace(-1, 1, 512, endpoint=False)); u0 = x0**2 * jnp.cos(jnp.pi * x0)

            def resid(p, x, t):
                u = uf(p, x, t)
                ut = jax.grad(uf, argnums=2)(p, x, t)
                uxx = jax.grad(jax.grad(uf, argnums=1), argnums=1)(p, x, t)
                return ut - D * uxx + 5 * u**3 - 5 * u

            def loss_pde(p, xc, tc, x0, u0):
                r = jax.vmap(resid, (None, 0, 0))(p, xc, tc)
                e0 = jax.vmap(uf, (None, 0, None))(p, x0, 0.0) - u0
                return jnp.mean(r**2) + a.w_ic * jnp.mean(e0**2)

            def loss_data(p, xd, td, ud):
                return a.w_data * jnp.mean((jax.vmap(uf, (None, 0, 0))(p, xd, td) - ud) ** 2)

            terms = [(loss_pde, (xc, tc, x0, u0))] + ([(loss_data, (xd, td, ud))] if a.n_data else [])
            if a.bench:     # time one step of the loss gradient with and without the data term, then stop
                for lab, tm in [("no data", terms[:1]), ("with data", terms)]:
                    vg = split_value_and_grad(tm); jax.block_until_ready(vg(p))
                    t0 = time.time()
                    for _ in range(20):
                        jax.block_until_ready(vg(p))
                    print(f"bench {name} {lab}: {(time.time() - t0) / 20 * 1000:.0f} ms/step on {jax.devices()[0]}", flush=True)
                continue

            t0 = time.time()
            p, hist = adam(terms, p, a.steps, a.lr)
            tt = time.time() - t0
            ev = jax.jit(jax.vmap(uf, (None, 0, None)))
            errs = {t: float(np.linalg.norm(np.asarray(ev(p, jnp.asarray(xg), t)) - ref[t]) / np.linalg.norm(ref[t]))
                    for t in t_out}
            rows.append({"model": name, "seed": sd, "params": n_params(p), "steps": a.steps, "time_s": tt, "n_data": a.n_data,
                         "loss_hist": hist, **{f"t{t}": e for t, e in errs.items()}})
            print(f"{name:10s} seed={sd} params={n_params(p)} " + " ".join(f"t={t}:{e:.1e}" for t, e in errs.items()) +
                  f" loss={hist[-1][1]:.1e} {tt:.0f}s", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
