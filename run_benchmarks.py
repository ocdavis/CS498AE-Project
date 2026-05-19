"""
Benchmark relax-and-cut GMICs against the 1gmi baseline on selected MIPLIB instances.

See "Configurations" section below to adjust instances and methods tested, results file, log path, and arguments of the relax-and-cut methods.

Writes results to results.csv incrementally.

Usage:
    python run_benchmarks.py

LMM Usage Disclaimer: Claude used to make the benchmarking more modular, and to add fail-safes (write incrementally, be able resume and skip already done instances,
 add a optimization limit, logs formatting issues or LP solving failures)
"""

import os
import sys
import csv
import time
import traceback
from pathlib import Path

# Set Gurobi license BEFORE importing gurobipy
os.environ.setdefault("GRB_LICENSE_FILE", r"C:\Users\davisoc\gurobi.lic")

import gurobipy as gp
from gurobipy import GRB

from gmic_lib import (
    run_1gmi_baseline,
    relax_and_cut_fast,
    relax_and_cut_subg,
)

# [ Configuration ]

RESULTS_CSV = Path("results_test.csv")
LOG_PATH = Path("run_test.log")

INSTANCES = [
    ("bell5",      "data/bell5.mps/bell5.mps",           8966406.49),
    ("p0282",      "data/p0282.mps/p0282.mps",           258411.0),
    ("mod008",     "data/mod008.mps/mod008.mps",         307.0),
    ("fiber",      "data/fiber.mps/fiber.mps",           405935.18),
    ("vpm2",       "data/vpm2.mps/vpm2.mps",             13.75),
    ("arki001",    "data/arki001.mps/arki001.mps",       7580813.0459),
    ("roll3000",   "data/roll3000.mps/roll3000.mps",     12890.0),
    ("phylo_small",  "data/phylo\phylo_small.mps",  -68.0),
    ("phylo_med",  "data/phylo\phylo_med.mps",  -114.0),
]

METHODS = ["1gmi", "fast", "subg"]

FAST_KWARGS = dict(L=10, mu=0.01, I_max=100, K=1, verbose=False)
SUBG_KWARGS = dict(mu0=10.0, I_max=10000, K=10, verbose=False)

OPT_TIMELIMIT = 300


# [ Configuration END ]

def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


def load_done(csv_path):
    """Return set of (instance, method) pairs already recorded as 'ok' in the CSV."""
    done = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((row["instance"], row["method"]))
    return done


def compute_opt(mps_path, env, time_limit):
    log(f"  Computing opt via Gurobi MIP (TimeLimit={time_limit}s)...")
    m = gp.read(mps_path, env=env)
    m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 0)
    m.optimize()
    if m.SolCount == 0:
        return None
    return float(m.ObjVal)


def initial_lp_value(mps_path, env):
    m = gp.read(mps_path, env=env)
    lp = m.relax()
    lp.setParam("OutputFlag", 0)
    lp.optimize()
    if lp.status != GRB.OPTIMAL:
        return None, None, None
    return float(lp.ObjVal), m.NumVars, m.NumConstrs


def run_method(method, mps_path, opt, z0, env, log_fh):
    """Run one method and return a result dict ready for CSV row."""
    t0 = time.time()
    try:
        if method == "1gmi":
            gap = run_1gmi_baseline(mps_path, opt, gp_env=env)
            z_final = z0 + (gap / 100.0) * (opt - z0) if gap is not None else None
        elif method == "fast":
            m = gp.read(mps_path, env=env)
            m.setParam("OutputFlag", 0)
            P = relax_and_cut_fast(m, opt, **FAST_KWARGS)
            if P is None:
                raise RuntimeError("relax_and_cut_fast returned None")
            P.optimize()
            z_final = float(P.ObjVal)
            gap = 100 * (z_final - z0) / (opt - z0) if abs(opt - z0) > 1e-9 else 0.0
        elif method == "subg":
            m = gp.read(mps_path, env=env)
            m.setParam("OutputFlag", 0)
            P = relax_and_cut_subg(m, opt, **SUBG_KWARGS)
            if P is None:
                raise RuntimeError("relax_and_cut_subg returned None")
            P.optimize()
            z_final = float(P.ObjVal)
            gap = 100 * (z_final - z0) / (opt - z0) if abs(opt - z0) > 1e-9 else 0.0
        else:
            raise ValueError(f"unknown method {method}")
        runtime = time.time() - t0
        log(f"    {method}: {gap:.2f}% in {runtime:.1f}s", log_fh)
        return {
            "z_final": z_final, "gap_closed_pct": gap,
            "runtime_s": runtime, "status": "ok", "error": "",
        }
    except KeyboardInterrupt:
        raise
    except Exception as e:
        runtime = time.time() - t0
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        log(f"    {method}: FAILED ({err}) after {runtime:.1f}s", log_fh)
        return {
            "z_final": "", "gap_closed_pct": "",
            "runtime_s": runtime, "status": "error", "error": err,
        }


# ---- Main ----

FIELDNAMES = [
    "instance", "method", "n_vars", "n_constrs",
    "z0", "opt", "z_final", "gap_closed_pct",
    "runtime_s", "status", "error",
]


def main():
    env = gp.Env()
    env.setParam("OutputFlag", 0)

    done = load_done(RESULTS_CSV)
    log_fh = LOG_PATH.open("a")
    log(f"\n===== Run started {time.ctime()}  ({len(done)} entries already done) =====", log_fh)

    csv_new = not RESULTS_CSV.exists()
    csv_fh = RESULTS_CSV.open("a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=FIELDNAMES)
    if csv_new:
        writer.writeheader()
        csv_fh.flush()

    try:
        for name, mps_path, opt in INSTANCES:
            log(f"\n{'=' * 60}\n{name}\n{'=' * 60}", log_fh)

            if not Path(mps_path).exists():
                log(f"  SKIP: {mps_path} not found", log_fh)
                continue

            if opt is None:
                opt = compute_opt(mps_path, env, OPT_TIMELIMIT)
                if opt is None:
                    log(f"  SKIP: could not compute opt for {name}", log_fh)
                    continue

            z0, n_vars, n_constrs = initial_lp_value(mps_path, env)
            if z0 is None:
                log(f"  SKIP: initial LP did not solve for {name}", log_fh)
                continue

            log(f"  n_vars={n_vars} n_constrs={n_constrs}  z0={z0:.4f}  opt={opt:.4f}", log_fh)

            methods_for_this = list(METHODS)

            for method in methods_for_this:
                if (name, method) in done:
                    log(f"  [skip] {method} already done", log_fh)
                    continue
                log(f"  Running {method}...", log_fh)
                result = run_method(method, mps_path, opt, z0, env, log_fh)
                writer.writerow({
                    "instance": name, "method": method,
                    "n_vars": n_vars, "n_constrs": n_constrs,
                    "z0": z0, "opt": opt,
                    **result,
                })
                csv_fh.flush()
    except KeyboardInterrupt:
        log("\nInterrupted by user. Partial results in results.csv.", log_fh)
    finally:
        csv_fh.close()
        log(f"===== Run ended {time.ctime()} =====", log_fh)
        log_fh.close()


if __name__ == "__main__":
    main()
