# nishioka-qpinn-solid

Quantum physics-informed neural networks (QPINN) and quantum extreme reservoir computing (QERC)
for solid mechanics, benchmarked against **size-matched and tuned classical models**.

The project asks a simple question: *when a quantum circuit is used to solve a PDE, what does it
actually contribute?* Every quantum model here is compared with classical controls of the same size,
including a model with the quantum circuit removed and a classical model that uses exactly the same
function space as the quantum features.

> Japanese introduction to the earlier tutorial scripts (01–10): [`README_ja.md`](README_ja.md).
> Detailed notes and all result tables (Japanese): [`docs/qpinn_next_plan.md`](docs/qpinn_next_plan.md).

## Main findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Variational QPINNs lose to size-matched classical models.** The circuit adds nothing that a same-size MLP or the same network with the circuit removed cannot do, in solid mechanics (Lamé, Kirsch, 2D thermoelasticity) and in fluids (cylinder wake, reproduction of G. Rakala's open-source QPINN). | `15`, `18`, `docs/qpinn_controls_results.md`, `figs/cyl_*` |
| 2 | **The bottleneck is optimisation, not expressivity.** The circuits can fit the exact solutions to ~1e-3, but PDE-loss training stalls at 5e-2–1. | `12`, `13` |
| 3 | **With a fixed reservoir and a linear readout, the reservoir does not change what can be represented.** Features p_k(x) = Tr[U†\|k⟩⟨k\|U ρ(x)] are linear in ρ(x), so their span is bounded by the encoding. The reservoir only matters when the encoding space is larger than 2^N (e.g. per-qubit inputs as in Ikeda et al. 2026); Clifford-only reservoirs are not enough. | `22`, `24`, `25`, `figs/qerc_reservoir.*` |
| 4 | **QERC-PINN (least-squares solve, no training loop) removes the optimisation wall.** On Kirsch it matches tuned tanh random features (~5e-5), but cosine features (Fourier PIELM, FS-PIELM) are ~40× better. | `26`, `29`, `figs/qerc_seeds.*`, `figs/qerc_fair_scales.*` |
| 5 | **On a Cu/Si TSV with a material interface, QERC beats 1-layer classical random features by ~10×, but a classical random projection of the same tensor-product space matches it exactly.** The gain comes from the tensor-product function space, not from anything specifically quantum. | `30`, `31`, `figs/tsv_controls.*` |
| 6 | **Shot noise is severe.** With parameter-shift derivatives, even 10^7 shots per circuit leave 4–11 % error and K_t = 2.5–2.8 (exact 3). | `27`, `figs/qerc_shot_noise.*` |
| 7 | **Random reservoirs fail in high-dimensional parametric problems.** For laminate wave dispersion with d = N layers, all random-feature models fail; random projections of an exponentially large space capture an exponentially small part of the target. | `33` |

## Repository layout

| Scripts | Content |
|---|---|
| `01`–`10` | Introductory quantum computing scripts (qubits, gates, VQE, QAOA, IBM hardware) |
| `11`–`17` | Variational QPINN for solid mechanics: Lamé cylinder, ablation, Kirsch plate (stress concentration), deep energy method for thermoelasticity, classical controls, data assimilation with noise, wafer warpage |
| `18`, `23` | Reproduction and improvement study of the open-source QPINN for the cylinder wake (G. Rakala, OIST), with figures |
| `19`–`22` | Physics-aware encodings, quantum extreme learning probes, QERC features, QERC-PINN solved by least squares |
| `24`–`29`, `32` | Reservoir-role analysis, Ikeda-type encoder, 5-seed comparisons, shot noise, fair scale tuning incl. Fourier PIELM / FS-PIELM, figures |
| `30`, `31` | Single TSV (Cu via in Si) thermal stress with domain decomposition, and classical controls |
| `33` | High-dimensional parametric benchmark (laminate wave dispersion) |
| `docs/` | Result notes and research plans (Japanese) |
| `figs/` | Figures (PNG and PDF) |
| `results/` | Raw results (JSON) |

Exact solutions (Lamé, Kirsch via Muskhelishvili potentials, thermoelastic annulus, single TSV) are
implemented in the scripts and verified numerically before use.

## Setup

```bash
pip install -r requirements.txt
pip install jax          # tested with JAX 0.10.2, PennyLane 0.45.1
```

All experiments run on CPU (state-vector simulation). Examples:

```bash
python 25_qerc_ikeda_pde.py --N 8          # QERC-PINN on Kirsch with different reservoirs
python 29_qerc_fair_scales.py --seeds 5    # fair comparison incl. Fourier PIELM / FS-PIELM
python 30_qerc_tsv_single.py --N 8         # single TSV thermal stress
```

The cylinder-wake scripts (`18`, `23`) need `cylinder_wake.mat` from
[github.com/GeetRakala/QPINN](https://github.com/GeetRakala/QPINN); pass its path with `--data`.

## References

- M. Raissi, P. Perdikaris, G. E. Karniadakis, *J. Comput. Phys.* 378 (2019) 686–707.
- G. Rakala, QPINN, [github.com/GeetRakala/QPINN](https://github.com/GeetRakala/QPINN).
- A. Ikeda, A. Sakurai, K. Nemoto, M. Muramatsu, *Quantum Extreme Reservoir Computing for Phase Classification of Polymer Alloy Microstructures*, arXiv:2601.02150 (2026).
- M. Schuld, R. Sweke, J. J. Meyer, *Phys. Rev. A* 103 (2021) 032430.
- X. Xiong et al., *Frequency Shift Physics-Informed Extreme Learning Machine*, arXiv:2607.01694 (2026).
- J. Hu, S. Jin, N. Liu, L. Zhang, *Quantum Random Feature Method for Solving PDEs*, arXiv:2510.07945 (2025).

## Author

Keisuke Nishioka — Keio University / Leibniz Universität Hannover
