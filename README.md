# Total-Pivoting

Comparison of the CPU time and accuracy of three variants of **Gaussian elimination** for solving linear systems `Ax = b` on random matrices:

1. **Row swapping**: rows are swapped only when the pivot is exactly zero
2. **Partial pivoting**: the pivot is the largest `|a_ik|` in the current column (`i >= k`)
3. **Total pivoting**: the pivot is the largest `|a_ij|` in the whole remaining submatrix (`i, j >= k`), with row and column swaps

Project for the *Devoir Mathématique*, INPT (Institut National des Postes et Télécommunications).

Author: **EL MEHDAOUI Chaimaa**

## How it works

For each size `n`, the script:

1. Generates a random matrix `A` with entries uniform on `[-1, 1]` (fixed seed for reproducibility)
2. Picks a random exact solution `x_ex` and sets `b = A @ x_ex`
3. Solves the system with each of the three methods and measures CPU time with `time.process_time()`
4. Computes the relative error `||x - x_ex|| / ||x_ex||` and the relative residual `||Ax - b|| / ||b||`

It then writes the results table, a log-log plot, and an extrapolation of the CPU time to `n = 20000` using the model `T(n) = c * n^3`.

> **Why extrapolate?** A dense `20000 x 20000` float64 matrix takes about 3.2 GB and elimination needs roughly 5.3 x 10^12 floating-point operations. Running it directly in pure Python/NumPy is impractical on a typical machine, so the script measures smaller sizes and extrapolates. Treat the `n = 20000` figures as orders of magnitude only.

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Usage

```bash
# Default sizes: 100, 200, 400, 600, 800, 1000, extrapolated to n = 20000
python gauss_comparison.py

# Custom sizes
python gauss_comparison.py --sizes 200 500 1000 1500

# Custom seed or extrapolation target
python gauss_comparison.py --seed 42 --extrapolate 20000
```

| Option | Default | Description |
|---|---|---|
| `--sizes` | `100 200 400 600 800 1000` | Matrix sizes to measure |
| `--seed` | `2026` | Random seed |
| `--extrapolate` | `20000` | Size for the O(n^3) CPU time estimate |

## Output files

| File | Content |
|---|---|
| `results.csv` | Raw measurements (size, method, CPU time, error, residual) |
| `results_table.tex` | Results table, ready to `\input` in LaTeX |
| `extrapolation.tex` | Estimated CPU time for `n = 20000` |
| `cpu_plot.pdf` | CPU time vs. size, log-log scale |

## Compiling the report

The LaTeX report reads the generated `.tex` and `.pdf` files, so run the script first:

```bash
python gauss_comparison.py
pdflatex devoir_mathematique.tex
pdflatex devoir_mathematique.tex
```

## Findings

- All three methods follow the same cubic growth in `n`.
- Row swapping and partial pivoting take about the same time; total pivoting is consistently the slowest because it searches the whole submatrix and permutes columns.
- Row swapping is the least accurate, since it does not protect against small pivots. Partial and total pivoting stay close to machine precision.
- Partial pivoting is the best trade-off between cost and stability, which is why libraries such as LAPACK use it.

Timings depend on your hardware, so your numbers will differ.

## Repository structure

```
Total-Pivoting/
├── gauss_comparison.py       # implementation and experiment
├── devoir_mathematique.pdf   # report
└── README.md
```
