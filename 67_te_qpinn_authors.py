"""
Runs the authors' TE-QPINN code as is, and compares embeddings and parameter-matched classical networks with it
(Muramatsu's suggestion, 2026-11-17).

  S. Berger, N. Hosters, M. Moeller, "Trainable embedding quantum physics informed neural networks for solving
  nonlinear PDEs", Sci. Rep. 15, 18823 (2025), doi:10.1038/s41598-025-02959-z
  Code: https://git.rwth-aachen.de/berger.st.11.11/te-qpinns  (commit f7d0177; PennyLane + PyTorch)

The circuit, embedding network, collocation points and loss are NOT re-implemented here: the code cells of the
authors' notebooks are read from a clone of their repository (--repo) and executed unchanged. Only these are ours:
  - the training loop is retyped from their loop cell (theirs also plots and loads a pickled reference): same optimiser
    call torch.optim.LBFGS(params, line_search_fn="strong_wolfe"), same number of outer steps, same stopping rule;
  - the reference solution (Fourier pseudo-spectral + ETDRK4) replaces their pickled interpolator;
  - torch.set_num_threads follows the cores reserved for the job instead of their 30;
  - two models they do not have: TE-twin and, for Allen-Cahn, the fixed embeddings and the PINNs (their Burgers code).

Models
  TE           trainable embedding (their FNN_BASIS circuit): Ry(phi_i(x) * x_c) with phi from a small tanh network
  NONE         fixed embedding Ry(x_c): the coordinates go in directly                      (their EMBEDDING "NONE")
  CHEBYSHEV    fixed embedding Ry(2 arccos x_c)                                             (their "CHEBYSHEV")
  TOWER        their "TOWER_CHEBYSHEV" as written (in the public code the tower factor is reset to 1 on every
               qubit, so it gives the same angles as CHEBYSHEV; kept because the paper's tables use it)
  NONE-match, CHEBYSHEV-match, TOWER-match   the same fixed embeddings with more variational layers, so that the
               parameter count matches the TE-QPINN (20 layers = 240 for Burgers, 17 layers = 204 for Allen-Cahn)
  PINN-3x10 / PINN-2x13   their classical PINNs for Burgers (261 / 235 parameters; the TE-QPINN has 244)
  PINN-match   Allen-Cahn: their FNN class sized to the TE-QPINN parameter count (2 x 12 = 205 vs 204)
  PINN-4x64    their FNN class, 4 x 64 (reference size)
  TE-twin      ours: the same embedding network, but the circuit is replaced by a trainable linear map on
               prod_i (1, cos a_i, sin a_i), a_i = phi_i(x) * x_c (3^n terms). Every output of a fixed circuit
               after a product Ry embedding is such a linear map, so this is the classical model class that
               contains the TE-QPINN.
Problems
  burgers   their Burgers notebooks (4 qubits, 5 layers, FNN 2 x 10, 100 x 50 grid, 500 outer L-BFGS steps)
  ch        Cahn-Hilliard of Wight & Zhao (2021) in mixed form: the loss is ours (see install_ch), everything else
            (circuit, embedding network, grid, causality training, optimiser) is their Allen-Cahn set-up
  ac        their Allen-Cahn notebook with causality training (Raissi's equation, 4 qubits, 5 layers, FNN 1 x 20,
            200 x 200 grid, initial-value weight 1e3, eps = 200, up to 300 outer L-BFGS steps)

  python 67_te_qpinn_authors.py --repo ~/te-qpinns --pde burgers --models TE,PINN-3x10,PINN-2x13 --seeds 3
  python 67_te_qpinn_authors.py --repo ~/te-qpinns --pde ac --models TE --seeds 3
"""
import argparse, json, os, subprocess, sys, time, types
import numpy as np
import matplotlib
matplotlib.use("Agg")

COMMIT = "f7d0177ed33ebd674fe9ed0fd846d18a5bf0129e"
FIXED = ("NONE", "CHEBYSHEV", "TOWER")
CH_G1, CH_G2 = 1e-6, 1e-2            # Cahn-Hilliard of Wight & Zhao (2021): gamma_1, gamma_2
NB_CMP = "PDE-Example/Burgers/burgers_pinn_comparison.ipynb"     # TE-QPINN vs PINN (paper Fig. 14)
NB_THS = "PDE-Example/Burgers/burgers_thesis.ipynb"              # embeddings (paper Fig. 11)
NB_AC = "CausalityTraining/allen_cahn.ipynb"
H = os.path.dirname(os.path.abspath(__file__))


# ---------------- the authors' cells ----------------
def cell(repo, rel, i, must_contain):
    c = json.load(open(os.path.join(repo, rel)))["cells"][i]
    src = "".join(c["source"])
    assert c["cell_type"] == "code" and must_contain in src, f"{rel} cell {i}: expected '{must_contain}'"
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))      # drop IPython magics only


def run_cell(ns, repo, rel, i, must_contain):
    exec(compile(cell(repo, rel, i, must_contain), f"{rel}[{i}]", "exec"), ns)


def check_repo(repo):
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == COMMIT, f"{repo} is at {head or '?'}; expected {COMMIT} (git -C {repo} checkout {COMMIT[:7]})"
    dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    assert not dirty, f"{repo} has local changes:\n{dirty}"


def ipython_stub():
    try:
        import IPython.display  # noqa: F401
    except ImportError:         # their Allen-Cahn cell imports it for live plots, which we do not draw
        m = types.ModuleType("IPython"); d = types.ModuleType("IPython.display")
        d.display = d.clear_output = lambda *a, **k: None
        m.display = d; sys.modules["IPython"] = m; sys.modules["IPython.display"] = d


# ---------------- reference solutions (numpy) ----------------
def etdrk4(u0_hat, Lop, Nf, dt, n_steps, every):
    """Kassam-Trefethen ETDRK4; yields the Fourier coefficients every `every` steps"""
    E, E2 = np.exp(dt * Lop), np.exp(dt * Lop / 2)
    M = 32; r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lop[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, 1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, 1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, 1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, 1))
    v = u0_hat
    for n in range(1, n_steps + 1):
        Nv = Nf(v); a = E2 * v + Q * Nv; Na = Nf(a); b = E2 * v + Q * Na; Nb = Nf(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nf(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if n % every == 0:
            yield v


def reference(pde, t_grid, x_grid, n_modes=None, sub=None):
    """u on the authors' (t, x) grid (t uniform from 0). Both problems are 2-periodic in x on [-1, 1]:
    Allen-Cahn by its boundary condition; Burgers because -sin(pi x) is odd, so u(+-1) = 0 is kept."""
    n_modes = n_modes or (4096 if pde == "burgers" else 1024)
    xs = -1 + 2 * np.arange(n_modes) / n_modes
    k = np.fft.fftfreq(n_modes, d=2 / n_modes) * 2 * np.pi
    dT = float(t_grid[1] - t_grid[0])
    sub = sub or int(np.ceil(dT / 1e-5))
    if pde == "ac":
        Lop = -1e-4 * k**2 + 5.0; Nf = lambda v: -5 * np.fft.fft(np.real(np.fft.ifft(v)) ** 3)
        u0 = xs**2 * np.cos(np.pi * xs)
    elif pde == "ch":     # phi_t = (G2 (phi^3 - phi) - G1 phi_xx)_xx
        Lop = CH_G2 * k**2 - CH_G1 * k**4; Nf = lambda v: -CH_G2 * k**2 * np.fft.fft(np.real(np.fft.ifft(v)) ** 3)
        u0 = -np.cos(2 * np.pi * xs)
    else:
        Lop = -(0.01 / np.pi) * k**2; Nf = lambda v: -0.5j * k * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        u0 = -np.sin(np.pi * xs)
    ev = np.exp(1j * np.outer(np.asarray(x_grid, float) + 1, k)) / n_modes
    v0 = np.fft.fft(u0); out = [np.real(ev @ v0)]
    for v in etdrk4(v0, Lop, Nf, dT / sub, sub * (len(t_grid) - 1), sub):
        out.append(np.real(ev @ v))
    return np.array(out)                      # (n_t, n_x)


def cached_reference(pde, t_grid, x_grid, smoke, outdir):
    fn = os.path.join(outdir, f"te_authors_ref_{pde}_{len(t_grid)}x{len(x_grid)}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    t0 = time.time(); u = reference(pde, t_grid, x_grid, sub=20 if smoke and pde != "ch" else None)   # CH is stiff: always dt = 1e-5
    print(f"reference {pde} {u.shape} {time.time() - t0:.0f}s", flush=True)
    if not smoke:
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        tmp = f"{fn}.{os.getpid()}.tmp.npy"; np.save(tmp, u); os.replace(tmp, fn)      # atomic: parallel jobs share it
    return u


# ---------------- set-up of the authors' namespaces ----------------
def setup(pde, repo, threads, smoke):
    ipython_stub()
    A = {"__name__": "te_authors"}
    if pde == "burgers":
        T = {"__name__": "te_authors_thesis"}
        for ns, rel in ((A, NB_CMP), (T, NB_THS)):
            run_cell(ns, repo, rel, 0, "import pennylane as qml")
            run_cell(ns, repo, rel, 1, "N_WIRES")
            if smoke:
                ns["T_COLLOC_POINTS"], ns["X_COLLOC_POINTS"] = 6, 8
            run_cell(ns, repo, rel, 2, "interior_colloc")
            run_cell(ns, repo, rel, 3, "def circuit")
            run_cell(ns, repo, rel, 5, "def pde_res_fnc")
        P = A                                   # their FNN class and model_pinn live in the comparison notebook
    else:
        run_cell(A, repo, NB_AC, 0, "USE_CAUSALITY")
        if smoke:
            A["T_COLOC_POINTS"], A["X_COLOC_POINTS"] = 6, 8
        run_cell(A, repo, NB_AC, 1, "interior_colloc")
        run_cell(A, repo, NB_AC, 3, "def circuit")
        run_cell(A, repo, NB_AC, 4, "def compute_temporal_weights")
        assert A["USE_CAUSALITY"] and A["N_WIRES"] == 4
        # fixed embeddings and PINNs for Allen-Cahn: the authors' Burgers model cells, on the Allen-Cahn domain
        T = {"__name__": "te_authors_thesis"}; P = {"__name__": "te_authors_cmp"}
        for ns, rel in ((T, NB_THS), (P, NB_CMP)):
            run_cell(ns, repo, rel, 0, "import pennylane as qml")
            run_cell(ns, repo, rel, 1, "N_WIRES")
            assert (ns["N_WIRES"], ns["N_LAYERS"]) == (A["N_WIRES"], A["N_LAYERS"])
            ns["domain_bounds"] = A["domain_bounds"]
            run_cell(ns, repo, rel, 3, "def circuit")
    torch = A["torch"]
    torch.set_num_threads(threads)
    return A, T, P, torch


# ---------------- models ----------------
def n_par(params):
    return int(sum(p.numel() for p in params))


def make_model(name, pde, A, T, P, torch):
    """returns (model, params, angle_fn or None, namespace whose loss functions see this model as their global)"""
    dev = A["device"]; nq, nl = A["N_WIRES"], A["N_LAYERS"]
    two_pi = 2 * torch.pi if pde != "burgers" else 1.0           # theta init: 2 pi rand (their AC cell) / rand (their Burgers cells)
    new_theta = lambda: (two_pi * torch.rand(nl, nq, 3, device=dev)).clone().detach().requires_grad_(True)
    db = A["domain_bounds"]
    resc = lambda x: 1.9 * (x - db[0]) / (db[1] - db[0]) - 0.95
    emb_net = (lambda: A["FNN"](2, 10).to(dev)) if pde == "burgers" else \
              (lambda: A["FNNBasisNet"](A["HIDDEN_LAYER_FNN"], A["NEURONS_FNN"]).to(dev))
    comp = [i % 2 for i in range(nq)]                       # qubit i takes x_rescaled[:, i % 2] (0: t, 1: x), as in their circuit
    if name == "TE":
        A["theta"] = new_theta(); A["basisNet"] = emb_net()
        ang = lambda x: A["basisNet"](resc(x)) * resc(x)[:, comp]
        return A["_model_te"], [A["theta"], *A["basisNet"].parameters()], ang
    if name.split("-")[0] in FIXED:
        base = name.split("-")[0]
        T["EMBEDDING"] = {"TOWER": "TOWER_CHEBYSHEV"}.get(base, base)
        # "-match": more variational layers, so that the parameter count matches the TE-QPINN (their circuit
        # reads the global N_LAYERS when it is called, so only the number is changed, not their code)
        T["N_LAYERS"] = nl
        if name.endswith("-match"):     # (only here: building the network draws random numbers)
            T["N_LAYERS"] = round(n_par([torch.zeros(nl, nq, 3), *emb_net().parameters()]) / (3 * nq))
        T["theta"] = (two_pi * torch.rand(T["N_LAYERS"], nq, 3, device=dev)).clone().detach().requires_grad_(True)
        return T["model"], [T["theta"]], None
    if name.startswith("PINN"):
        if name == "PINN-match":
            target = n_par([torch.zeros(nl, nq, 3), *emb_net().parameters()])
            h = min(range(2, 200), key=lambda h: abs(2 * h + h + h * h + h + h + 1 - target)); depth = 2
        else:
            depth, h = [int(s) for s in name.split("-")[1].split("x")]
        P["pinn"] = P["FNN"](depth, h, output_dim=1).to(dev)
        return P["model_pinn"], list(P["pinn"].parameters()), None
    if name == "TE-twin":
        net = emb_net()
        C = (0.1 * torch.randn(3**nq, device=dev)).requires_grad_(True)
        b = torch.zeros(1, device=dev, requires_grad=True)
        ang = lambda x: net(resc(x)) * resc(x)[:, comp]

        def twin(x):
            a = ang(x); f = torch.ones(a.shape[0], 1, device=dev)
            for i in range(nq):
                g = torch.stack([torch.ones_like(a[:, i]), torch.cos(a[:, i]), torch.sin(a[:, i])], 1)
                f = (f[:, :, None] * g[:, None, :]).reshape(a.shape[0], -1)
            return f @ C + b[0]
        return twin, [*net.parameters(), C, b], ang
    raise ValueError(name)


def bound(name, model, ang, A, T, P):
    """Freeze the globals the authors' model functions read (theta, basisNet, EMBEDDING, pinn), so that two
    networks of the same kind can coexist (Cahn-Hilliard needs one for phi and one for the chemical potential)."""
    if name == "TE":
        th, net = A["theta"], A["basisNet"]

        def use():
            A["theta"], A["basisNet"] = th, net
    elif name.split("-")[0] in FIXED:
        th, emb, nlay = T["theta"], T["EMBEDDING"], T["N_LAYERS"]

        def use():
            T["theta"], T["EMBEDDING"], T["N_LAYERS"] = th, emb, nlay
    elif name.startswith("PINN"):
        pn = P["pinn"]

        def use():
            P["pinn"] = pn
    else:
        return model, ang

    def f(x):
        use(); return model(x)

    def g(x):
        use(); return ang(x)
    return f, (g if ang is not None else None)


def install_ch(A, torch, phi, m):
    """Cahn-Hilliard in mixed form, in the structure of the authors' Allen-Cahn loss cell (ours; they have no CH):
        phi_t = G2 m_xx,   m = phi^3 - phi - (G1 / G2) phi_xx      (m = mu / G2, so that both outputs are O(1))
    phi and m are two separate networks, as the authors use two TE-QPINNs for the two fields of the cavity flow.
    Their loss_fnc, weighted_temporal_loss_fnc and compute_temporal_weights are kept; the three functions they
    call are replaced: the residual per time slice, the periodic boundary term and the initial-value term."""
    ic, top, bot, inter = (A[k] for k in ("init_val_colloc", "periodic_boundary_top_colloc",
                                          "periodic_boundary_bottom_colloc", "interior_colloc"))
    nt, nx = A["T_COLOC_POINTS"], A["X_COLOC_POINTS"]
    d = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    def boundary_initial_values():
        return torch.mean((phi(ic) - (-torch.cos(2 * torch.pi * ic[:, 1]))) ** 2)

    def boundary_periodic():
        res = 0.0
        for f in (phi, m):
            u1 = f(top); u3 = f(bot)
            res = res + torch.mean((u1 - u3) ** 2) + torch.mean((d(u1, top)[:, 1] - d(u3, bot)[:, 1]) ** 2)
        return res

    def temporal_loss_slices_fnc():
        p = phi(inter); dp = d(p, inter)
        p_t = dp[:, 0]; p_xx = d(dp[:, 1], inter)[:, 1]
        q = m(inter); q_xx = d(d(q, inter)[:, 1], inter)[:, 1]
        res = (p_t - CH_G2 * q_xx) ** 2 + (q - p**3 + p + (CH_G1 / CH_G2) * p_xx) ** 2
        loss_tmp = torch.zeros(nt, device=A["device"])
        loss_tmp[0] = A["INITIAL_VALUE_SCALING"] * boundary_initial_values()
        loss_tmp[1:] = torch.mean(res.reshape(nt - 1, nx - 2), axis=1)
        return loss_tmp
    A["boundary_initial_values"] = boundary_initial_values
    A["boundary_periodic"] = boundary_periodic
    A["temporal_loss_slices_fnc"] = temporal_loss_slices_fnc


# ---------------- training loops (retyped from the authors' loop cells) ----------------
def train_burgers(A, T, torch, model, params, in_thesis_ns, outer):
    if in_thesis_ns:      # burgers_thesis.ipynb: loss functions read the global `model`; stop when the loss stalls
        loss_fnc = T["loss_fnc"]
    else:                 # burgers_pinn_comparison.ipynb: loss functions take the model; no stopping rule
        loss_fnc = lambda: A["BOUNDARY_SCALE"] * (A["init_val_loss"](model) + A["dir_boundary_loss"](model)) + \
            A["pde_res_fnc"](model)
    opt = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = loss_fnc()
        l.backward()
        return l
    hist = []; previous_loss = float("inf")
    for i in range(outer):
        opt.step(closure)
        L = loss_fnc().item(); hist.append(L)
        if in_thesis_ns:
            if abs(previous_loss - L) < 1e-10:
                break
            previous_loss = L
        if i % 25 == 0:
            print(f"  step {i} loss {L:.3e}", flush=True)
    return hist, {}


def train_ac(A, torch, model, params, outer, mse_fn):
    A["model"] = model                       # their loss functions read the global `model`
    opt = torch.optim.LBFGS(params, line_search_fn="strong_wolfe")
    loss_fnc = A["loss_fnc"]

    def closure():
        opt.zero_grad()
        l = loss_fnc()
        l.backward()
        return l
    hist, mse = [], []
    A["temporal_weights"] = torch.ones(A["T_COLOC_POINTS"], device=A["device"])
    for eps in A["eps_array"]:
        A["eps"] = eps
        for j in range(outer):
            A["temporal_weights"] = A["compute_temporal_weights"]()
            opt.step(closure)
            hist.append(loss_fnc().item()); mse.append(mse_fn())
            if torch.min(A["temporal_weights"]) > A["sigma"]:
                break
            if j % 10 == 0:
                print(f"  step {j} loss {hist[-1]:.3e} MSE_ref {mse[-1]:.3e} min w {torch.min(A['temporal_weights']).item():.2f}",
                      flush=True)
    return hist, {"mse_hist": mse, "min_weight": float(torch.min(A["temporal_weights"]).item())}


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/te-qpinns"))
    ap.add_argument("--pde", default="burgers", choices=["burgers", "ac", "ch"])
    ap.add_argument("--models", default=None)
    ap.add_argument("--seeds", type=int, default=3, help="runs per model; run i uses torch.manual_seed(42 + i) (theirs: 42)")
    ap.add_argument("--outer", type=int, default=None, help="outer L-BFGS steps (theirs: 500 Burgers, 300 Allen-Cahn)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "2")))
    ap.add_argument("--smoke", action="store_true", help="tiny grid, to test that the cells run (not a result)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    check_repo(a.repo)
    models = (a.models or ("TE,TOWER,NONE,PINN-3x10,PINN-2x13,TE-twin" if a.pde == "burgers"
                           else "TE,CHEBYSHEV,NONE,PINN-match,PINN-4x64,TE-twin")).split(",")
    outer = a.outer or (500 if a.pde == "burgers" else 300)
    out = a.out or os.path.join(H, "results", f"te_authors_{a.pde}.json")
    A, T, P, torch = setup(a.pde, a.repo, a.threads, a.smoke)
    A["_model_te"] = A["model"]
    import pennylane
    ver = {"torch": torch.__version__, "pennylane": pennylane.__version__, "device": str(A["device"]), "commit": COMMIT[:7]}
    print(ver, flush=True)
    dom = A["input_domain"]                                   # product(t, x), columns (t, x)
    tg = np.unique(dom[:, 0].detach().cpu().numpy()); xg = np.unique(dom[:, 1].detach().cpu().numpy())
    ref = cached_reference(a.pde, tg, xg, a.smoke, os.path.dirname(out) or ".")
    ref_t = torch.tensor(ref.reshape(-1), device=A["device"])
    for ns in (A, T, P):
        ns["reference_values"] = ref_t                         # used by their compute_MSE_ref
    rows = json.load(open(out)) if os.path.exists(out) and not a.smoke else []
    done = {(r["model"], r["seed"]) for r in rows}
    for name in models:
        for i in range(a.seeds):
            seed = 42 + i
            if (name, seed) in done:
                continue
            torch.manual_seed(seed)
            model, params, ang = make_model(name, a.pde, A, T, P, torch)
            if a.pde == "ch":      # a second network of the same kind for m = mu / G2
                model, ang = bound(name, model, ang, A, T, P)
                m_model, m_params, _ = make_model(name, a.pde, A, T, P, torch)
                m_model, _ = bound(name, m_model, None, A, T, P)
                params = params + m_params
                install_ch(A, torch, model, m_model)
            print(f"{a.pde} {name} seed={seed} params={n_par(params)}", flush=True)

            def predict():
                with torch.no_grad():
                    return model(dom).detach().double().cpu().numpy().reshape(len(tg), len(xg))
            t0 = time.time()
            if a.pde == "burgers":
                hist, extra = train_burgers(A, T, torch, model, params, name.split("-")[0] in FIXED, outer)
            else:
                hist, extra = train_ac(A, torch, model, params, outer,
                                       lambda: float(np.mean((predict() - ref) ** 2)))
            tt = time.time() - t0
            u = predict(); e = u - ref
            row = {"pde": a.pde, "model": name, "seed": seed, "params": n_par(params), "outer_steps": len(hist),
                   "loss": hist[-1], "mse_ref": float(np.mean(e**2)), "lmax_ref": float(np.abs(e).max()),
                   "rel_l2": float(np.linalg.norm(e) / np.linalg.norm(ref)),
                   "rel_l2_T": float(np.linalg.norm(e[-1]) / np.linalg.norm(ref[-1])),
                   "time_s": tt, "loss_hist": hist, **extra, **ver}
            rows.append(row)
            print(f"{a.pde} {name:10s} seed={seed} params={row['params']} steps={len(hist)} loss={hist[-1]:.2e} "
                  f"MSE_ref={row['mse_ref']:.2e} relL2={row['rel_l2']:.2e} relL2(T)={row['rel_l2_T']:.2e} {tt:.0f}s", flush=True)
            if a.smoke:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(rows, open(out, "w"), indent=1)
            if i == 0:          # fields for figures: prediction and, for TE / TE-twin, the embedding angles
                fld = {"t": tg, "x": xg, "u": u, "ref": ref}
                if ang is not None:
                    with torch.no_grad():
                        fld["angles"] = ang(dom).detach().cpu().numpy().reshape(len(tg), len(xg), -1)
                np.savez_compressed(out.replace(".json", f"_{name}_fields.npz"), **fld)


if __name__ == "__main__":
    main()
