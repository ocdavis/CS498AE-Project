# CS498AE Project (Relax-and-Cut methods for GMICs Reproduction + Extension)

A Python + Gurobi implementation of the relax-and-cut framework for Gomory mixed-integer cuts, based on:

> Fischetti and Salvagnin (2011), *A Relax-and-Cut Framework for Gomory Mixed-Integer Cuts*. Mathematical Programming Computation 3, 79-102.

Implements the `fast` and `subg` variants and compares them against a standard rank-1 GMI baseline on selected MIPLIB instances.

## Files

| File | Purpose |
|---|---|
| `gmic_lib.py` | GMIC generator, cut pool, `relax_and_cut_fast`, `relax_and_cut_subg`, `run_1gmi_baseline`|
| `run_benchmarks.py` | Benchmark runner|
| `gen_phylo.py` | Simulator of the minimum-conflict-free column selection problem|
| `data/` | MPS files |

## Requirements

- Python 3.10+
- A Gurobi license (that is not free/restricted)
- Packages: `gurobipy`, `numpy`, `scipy`

```bash
pip install gurobipy numpy scipy
```

## Usage


### Benchmark

Edit the `INSTANCES`, `METHODS`, `FAST_KWARGS`, and `SUBG_KWARGS` constants at the top of `run_benchmarks.py`, then:

```powershell
python run_benchmarks.py
```

## Output

`results.csv` columns:

| Column | Meaning |
|---|---|
| `instance` | Short name of the MIP instance. |
| `method` | `1gmi`, `fast`, `subg` |
| `n_vars`, `n_constrs` | Size of the instance |
| `z0` | Initial LP relaxation objective |
| `opt` | Best known integer optimum |
| `z_final` | LP value after adding all cuts from the method |
| `gap_closed_pct` | `100 * (z_final - z0) / (opt - z0)` |
| `runtime_s` | Runtime in seconds for the method on this instance |
| `status` | `ok` or `error`. |
| `error` | Summary of error (if one exists) |

## LLM usage

Disclaimers in `gmic_lib.py` and `run_benchmarks.py` document which parts were debugged/refactored/optimized with Claude
