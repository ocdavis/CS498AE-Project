"""
Generator for simulated tumor-phylogeny instances of the
minimum-conflict-free column selection problem.

Usage:
python gen_phylo.py --outdir [path to output directory] --seed [seed number]

Example Usage:
python gen_phylo.py --outdir "data2/phylo" --seed 42

"""

import argparse
import os
import numpy as np


def conflict_pairs(M):
    """Return list of (j, k) column-index pairs in conflict."""
    n, m = M.shape
    pairs = []
    for j in range(m):
        col_j = M[:, j]
        for k in range(j + 1, m):
            col_k = M[:, k]
            has_10 = np.any((col_j == 1) & (col_k == 0))
            has_01 = np.any((col_j == 0) & (col_k == 1))
            has_11 = np.any((col_j == 1) & (col_k == 1))
            if has_10 and has_01 and has_11:
                pairs.append((j, k))
    return pairs


def generate_matrix(n_samples, n_mutations, p, seed):
    """Sample M ~ Bernoulli(density), n x m."""
    rng = np.random.default_rng(seed)
    return (rng.random((n_samples, n_mutations)) < p).astype(np.int8)


def build_ilp(n_mutations, weights, conf_pairs, name="phylo"):
    """Build the conflict-free column selection ILP as a Gurobi model."""
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(name, env=env)
    model.ModelSense = GRB.MINIMIZE

    neg_w = (-weights).tolist()
    x = model.addVars(n_mutations, vtype=GRB.BINARY,
                      obj=neg_w, name="x")
    for (j, k) in conf_pairs:
        model.addConstr(x[j] + x[k] <= 1, name=f"c_{j}_{k}")
    model.update()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/phylo",
                    help="Where to write the .mps files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # [ Configuration ]

    configs = [
        # (size_tag,  name,        n_samples, n_mutations, sampling probability)
        ("small",  "phylo_small",       30,         50,         0.15),
        ("med",    "phylo_med",         50,        100,         0.10),
    ]

    rng = np.random.default_rng(args.seed)

    for tag, name, n_samples, n_mutations, density in configs:
        sub_seed = int(rng.integers(0, 2**31))
        M = generate_matrix(n_samples, n_mutations, density, sub_seed)
        cpairs = conflict_pairs(M)

        if len(cpairs) == 0:
            print(f"  {name}: zero conflicts — skipping (would be trivial)")
            continue

        weights = rng.integers(1, 11, size=n_mutations).astype(float)

        model = build_ilp(n_mutations, weights, cpairs, name=name)

        lp = model.relax()
        lp.setParam("OutputFlag", 0)
        lp.optimize()
        z_lp = lp.ObjVal

        model.setParam("OutputFlag", 0)
        model.optimize()
        if model.SolCount == 0:
            print(f"  {name}: ILP solver found no solution — skipping")
            continue
        opt = model.ObjVal
        print(f"Opt value (to use in benchmark): {opt}")
        print(f"z_lp = {z_lp}")

        mps_path = os.path.join(args.outdir, f"{name}.mps")
        model.write(mps_path)

if __name__ == "__main__":
    main()