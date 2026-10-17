#!/usr/bin/env python3
"""
CPU time comparison of Gaussian elimination with:
  1. Row swapping (swap only when the pivot is zero)
  2. Partial pivoting
  3. Total pivoting
"""
import argparse
import csv
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def back_substitution(U, c):
    """Solve U x = c, where U is upper triangular."""
    n = c.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def gauss_row_swap(A, b):
    """Gaussian elimination with row swapping only when the pivot is zero."""
    A = A.copy()
    b = b.copy()
    n = b.shape[0]
    for k in range(n - 1):
        if A[k, k] == 0.0:
            nz = np.flatnonzero(A[k + 1:, k])
            if nz.size == 0:
                raise np.linalg.LinAlgError("Singular matrix")
            p = k + 1 + nz[0]
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        m = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(m, A[k, k:])
        b[k + 1:] -= m * b[k]
    return back_substitution(A, b)


def gauss_partial(A, b):
    """Gaussian elimination with partial pivoting: largest |a_ik| for i >= k."""
    A = A.copy()
    b = b.copy()
    n = b.shape[0]
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        m = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(m, A[k, k:])
        b[k + 1:] -= m * b[k]
    return back_substitution(A, b)


def gauss_total(A, b):
    """Gaussian elimination with total pivoting: largest |a_ij| for i, j >= k."""
    A = A.copy()
    b = b.copy()
    n = b.shape[0]
    cols = np.arange(n)          # permutation of the unknowns
    for k in range(n - 1):
        sub = np.abs(A[k:, k:])
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        p, q = k + i, k + j
        if p != k:               # swap rows
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        if q != k:               # swap columns
            A[:, [k, q]] = A[:, [q, k]]
            cols[[k, q]] = cols[[q, k]]
        m = A[k + 1:, k] / A[k, k]
        A[k + 1:, k:] -= np.outer(m, A[k, k:])
        b[k + 1:] -= m * b[k]
    y = back_substitution(A, b)
    x = np.empty(n)
    x[cols] = y                  # restore the order of the unknowns
    return x


METHODS = {
    "Row swapping": gauss_row_swap,
    "Partial pivoting": gauss_partial,
    "Total pivoting": gauss_total,
}


def run_one(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, n))
    x_exact = rng.uniform(-1.0, 1.0, size=n)
    b = A @ x_exact

    rows = []
    for name, solver in METHODS.items():
        t0 = time.process_time()
        x = solver(A, b)
        cpu = time.process_time() - t0
        err = np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        rows.append({"n": n, "method": name, "cpu": cpu, "err": err, "res": res})
    return rows


def write_outputs(all_rows, extrap_n):
    # CSV
    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "method", "cpu", "err", "res"])
        w.writeheader()
        w.writerows(all_rows)

    # LaTeX table of results
    with open("results_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\begin{tabular}{rlrrr}\n\\toprule\n")
        f.write("$n$ & Method & CPU (s) & Relative error & Relative residual \\\\\n\\midrule\n")
        for r in all_rows:
            f.write(f"{r['n']} & {r['method']} & {r['cpu']:.4f} & "
                    f"{r['err']:.2e} & {r['res']:.2e} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Experimental results (CPU time and accuracy)}\n\\end{table}\n")

    # O(n^3) extrapolation from the largest measured size
    nmax = max(r["n"] for r in all_rows)
    with open("extrapolation.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\begin{tabular}{lrr}\n\\toprule\n")
        f.write(f"Method & Estimated time for $n={extrap_n}$ (s) & Estimated time (h) \\\\\n\\midrule\n")
        for name in METHODS:
            last = [r for r in all_rows if r["method"] == name and r["n"] == nmax][0]
            c = last["cpu"] / nmax**3
            est = c * extrap_n**3
            f.write(f"{name} & {est:.3e} & {est/3600:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"\\caption{{Extrapolated CPU time for $n={extrap_n}$ (model $T(n)=c\\,n^3$)}}\n")
        f.write("\\end{table}\n")

    # Log-log plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in METHODS:
        xs = [r["n"] for r in all_rows if r["method"] == name]
        ys = [r["cpu"] for r in all_rows if r["method"] == name]
        ax.loglog(xs, ys, marker="o", label=name)
    ax.set_xlabel("Size $n$")
    ax.set_ylabel("CPU time (s)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig("cpu_plot.pdf")


def main():
    ap = argparse.ArgumentParser(description="Comparison of Gaussian elimination pivoting strategies")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[100, 200, 400, 600, 800, 1000])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--extrapolate", type=int, default=20000)
    args = ap.parse_args()

    all_rows = []
    for n in args.sizes:
        for r in run_one(n, args.seed):
            all_rows.append(r)
            print(f"n={n:6d}  {r['method']:<18s}  CPU={r['cpu']:10.4f} s  "
                  f"error={r['err']:.2e}  residual={r['res']:.2e}")

    write_outputs(all_rows, args.extrapolate)

    N = args.extrapolate
    print(f"\nFor n = {N}: matrix = {8 * N * N / 1e9:.2f} GB, "
          f"about {2 * N**3 / 3:.2e} flops per method.")
    print("Generated files: results.csv, results_table.tex, "
          "extrapolation.tex, cpu_plot.pdf")


if __name__ == "__main__":
    main()
