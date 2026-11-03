"""
Idea 4: evaluate the circuit once, solve many problems.
The feature matrices (Phi, Phi'') depend only on the features and the collocation points, not on the PDE
coefficients or the initial condition. Build them once, then solve Allen-Cahn for several diffusion
coefficients D and initial conditions, timing feature construction vs each solve.
On hardware, Phi and Phi'' are the only quantum part, so the circuit cost is paid once for all problems.

  python 57_ac_reuse.py --N 8 --scale 4
"""
import argparse, json, os, time, importlib.util
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ac", os.path.join(_here, "52_allen_cahn_qerc.py"))
ac = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ac)

ICS = {
    "x2cos": lambda x: x**2 * np.cos(np.pi * x),
    "cos2": lambda x: 0.5 * np.cos(2 * np.pi * x),
    "bumps": lambda x: 0.3 * np.sin(np.pi * x) + 0.2 * np.cos(3 * np.pi * x),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--model", default="QERC-Haar", choices=["QERC-Haar", "tanh-RF", "Fourier"])
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--Ds", default="1e-4,3e-4,1e-3")
    ap.add_argument("--ics", default="x2cos,cos2,bumps")
    ap.add_argument("--out", default="results/ac_reuse.json")
    a = ap.parse_args()
    m = 2**a.N; M = 4 * m
    x = -1 + 2 * (np.arange(M) + 0.5) / M
    ctor = {"QERC-Haar": lambda: ac.make_qerc(a.N, 0, a.scale), "tanh-RF": lambda: ac.make_tanh(m, 0, a.scale),
            "Fourier": lambda: ac.make_fourier(m, 0, a.scale)}[a.model]
    t0 = time.time(); Phi, Phixx = ac.feature_mats(ctor(), x); t_feat = time.time() - t0
    print(f"features ({a.model}, N={a.N}, x{a.scale}): {t_feat:.1f}s, once", flush=True)
    rows = []
    for icn in a.ics.split(","):
        for D in map(float, a.Ds.split(",")):
            ref = ac.reference(D, x, [0.5, 1.0], n_modes=512, dt=2e-5, u0=ICS[icn])
            t0 = time.time()
            _, out = ac.march(Phi, Phixx, x, D, 1e-3, 1.0, [0.5, 1.0], u0=ICS[icn])
            t_solve = time.time() - t0
            e = {t: float(np.linalg.norm(out[t] - ref[t]) / np.linalg.norm(ref[t])) if out else float("nan")
                 for t in (0.5, 1.0)}
            rows.append({"model": a.model, "N": a.N, "scale": a.scale, "ic": icn, "D": D,
                         "t0.5": e[0.5], "t1.0": e[1.0], "t_solve_s": t_solve, "t_feat_s": t_feat})
            print(f"ic={icn:6s} D={D:.0e}: t=0.5 {e[0.5]:.1e}  t=1 {e[1.0]:.1e}  solve {t_solve:.1f}s", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
