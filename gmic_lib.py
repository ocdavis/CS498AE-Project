"""GMIC generation and Lagrangian relax-and-cut framework, based on
Fischetti and Salvagnin (2011), "A Relax-and-Cut Framework for Gomory
Mixed-Integer Cuts".

LLM Usage Disclaimer: I implemented the main relax-and-cut framework
(Algorithm 1), the cut deduplication, the subgradient method with the
Polyak step-size rule, and initial try at generating rank 1 GMICs, the dynamism check, and the fractionality
threshold based on the paper. Claude helped debug the implementation
and adjust the tableau extraction to work with Gurobi's basis
representation (handling VBasis flags for variables nonbasic at upper
bound, free / super-basic variables, and equality slacks that can
appear basic in degenerate LPs), and refactored the inner loop for
performance (sparse cut pool, vectorized GMIC formula (_gmic_coeff), single-call
objective updates).

Needed performance adjustments even smaller runs were taking minutes
as opposed to seconds, and runs the run at an hour now were running at
over 10.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import time


def _gmic_coeff(a_bar, is_int, f0):
    """Vectorized mixed-integer Gomory cut coefficient formula.

    Applies the standard GMIC coefficient formula (Fischetti and
    Salvagnin Section 2, eq. 5) elementwise. Expects the tableau row to
    already be in y-space (i.e. with at-upper-bound complementation
    applied to a_bar if applicable).

    Parameters:
        a_bar: numpy array of tableau row coefficients for the non-basic variables
        is_int: boolean array, True where the variable is integer-typed
        f0: fractional part of the basic variable's LP value

    Returns:
        numpy array of GMIC coefficients
    """
    coeff = np.zeros_like(a_bar, dtype=float)
    f_j = a_bar - np.floor(a_bar)
    int_below = is_int & (f_j <= f0)
    int_above = is_int & (f_j > f0)
    cont_pos = (~is_int) & (a_bar >= 0)
    cont_neg = (~is_int) & (a_bar < 0)
    coeff[int_below] = f_j[int_below]
    coeff[int_above] = f0 * (1.0 - f_j[int_above]) / (1.0 - f0)
    coeff[cont_pos] = a_bar[cont_pos]
    coeff[cont_neg] = -f0 * a_bar[cont_neg] / (1.0 - f0)
    return coeff


def generate_rank1_gmics(model, orig_model, *,
                         frac_lo=0.001,
                         coeff_tol=1e-9,
                         dyn_max=1e6,
                         seen_hashes=None):
    """Generate rank-1 GMICs from the current optimal LP basis.

    For each basic integer variable whose LP value is fractional within
    [frac_lo, 1 - frac_lo], extracts the corresponding tableau row,
    applies the mixed-integer Gomory formula, projects away slack
    variables, and applies numerical guards (coefficient cleanup,
    fractionality threshold, dynamism limit).

    Parameters:
        model: optimized Gurobi LP model
        orig_model: the original MIP model, used to identify which
            variables are integer-typed.
        frac_lo: minimum fractional distance from an integer required
            for a basic variable to generate a cut
        coeff_tol: minimum absolute value of a tableau or cut
            coefficient considered nonzero
        dyn_max: maximum allowed dynamism ratio max|c| / min|c|
        seen_hashes: set of normalized cut hashes used for deduplication

    Returns:
        List of cut dicts. Each cut is
        {'cols': int ndarray of column indices,
         'coeffs': float ndarray of coefficients,
         'rhs': float}, meaning sum(coeffs * x[cols]) >= rhs.
    """
    cuts = []
    if model.status != GRB.OPTIMAL:
        return cuts

    lp_vars = model.getVars()
    constrs = model.getConstrs()
    n = len(lp_vars)
    m = len(constrs)

    A_csc = model.getA().tocsc()
    A_csr = A_csc.tocsr()

    vbasis = np.array(model.getAttr('VBasis', lp_vars))
    cbasis = np.array(model.getAttr('CBasis', constrs))
    lb = np.array(model.getAttr('LB', lp_vars))
    ub = np.array(model.getAttr('UB', lp_vars))
    xval = np.array(model.getAttr('X', lp_vars))
    rhs_arr = np.array(model.getAttr('RHS', constrs))
    senses = np.array([c.Sense for c in constrs])
    is_int = np.array([v.VType in (GRB.INTEGER, GRB.BINARY)
                       for v in orig_model.getVars()])

    AT_LB, AT_UB, FREE = -1, -2, -3

    basic_vars_idx = np.where(vbasis == GRB.BASIC)[0]
    basic_constrs_idx = np.where(cbasis == GRB.BASIC)[0]
    if len(basic_vars_idx) + len(basic_constrs_idx) != m:
        return cuts

    # LLM Adjustment/Debug Disclaimer: Equality constraints can be reported as CBasis=BASIC by Gurobi in
    # degenerate LPs. Their slack is identically zero, so the column
    # sign is arbitrary as long as B stays invertible.
    cols_list = [A_csc[:, j] for j in basic_vars_idx]
    for i in basic_constrs_idx:
        sign = 1.0 if senses[i] == GRB.LESS_EQUAL else -1.0
        cols_list.append(sp.csc_matrix(([sign], ([i], [0])), shape=(m, 1)))
    B = sp.hstack(cols_list, format='csc')

    try:
        lu = splinalg.splu(B)
    except Exception:
        return cuts

    nb_lb = (vbasis == AT_LB)
    nb_ub = (vbasis == AT_UB)
    nb_fr = (vbasis == FREE)
    nb_struct = nb_lb | nb_ub

    nb_constr = (cbasis != GRB.BASIC)
    nb_le = nb_constr & (senses == GRB.LESS_EQUAL)
    nb_ge = nb_constr & (senses == GRB.GREATER_EQUAL)

    for k, j_basic in enumerate(basic_vars_idx):
        if not is_int[j_basic]:
            continue
        x_v = xval[j_basic]
        f0 = x_v - np.floor(x_v)
        if f0 < frac_lo or f0 > 1.0 - frac_lo:
            continue

        e_k = np.zeros(m); e_k[k] = 1.0
        try:
            lam = lu.solve(e_k, trans='T')
        except Exception:
            continue

        a_bar = A_csr.T @ lam

        if np.any(nb_fr & (np.abs(a_bar) > coeff_tol)):
            continue

        eff_a_bar = np.where(nb_ub, -a_bar, a_bar)

        coeffs_struct = np.zeros(n)
        active = nb_struct & (np.abs(a_bar) > coeff_tol)
        if active.any():
            coeffs_struct[active] = _gmic_coeff(
                eff_a_bar[active], is_int[active], f0)
        coeffs_struct[nb_ub] = -coeffs_struct[nb_ub]

        bound_at = np.where(nb_ub, ub, lb)
        bound_at = np.where(np.isinf(bound_at), 0.0, bound_at)
        rhs_struct_adj = float(np.sum(coeffs_struct * bound_at))

        a_bar_slack = np.zeros(m)
        a_bar_slack[nb_le] = lam[nb_le]
        a_bar_slack[nb_ge] = -lam[nb_ge]
        nu = np.zeros(m)
        slack_active = (nb_le | nb_ge) & (np.abs(a_bar_slack) > coeff_tol)
        nu[slack_active] = np.where(
            a_bar_slack[slack_active] >= 0,
            a_bar_slack[slack_active],
            -f0 * a_bar_slack[slack_active] / (1.0 - f0),
        )
        v_coeffs = np.zeros(m)
        v_coeffs[nb_le] = -nu[nb_le]
        v_coeffs[nb_ge] = nu[nb_ge]
        rhs_slack_adj = float(-np.sum(nu[nb_le] * rhs_arr[nb_le]) +
                              np.sum(nu[nb_ge] * rhs_arr[nb_ge]))

        coeffs_full = coeffs_struct + A_csr.T @ v_coeffs
        final_rhs = f0 + rhs_struct_adj + rhs_slack_adj

        # LLM Efficiency Adjustment Disclaimer: SCIP-style coefficient cleanup (from sepa_gmi.c): zero entries
        # below max(1e-9, 1e-12 * max|coeff|) before the dynamism check.
        max_abs = float(np.max(np.abs(coeffs_full))) if coeffs_full.size else 0.0
        if max_abs < coeff_tol:
            continue
        cleanup = max(coeff_tol, 1e-12 * max_abs)
        coeffs_full[np.abs(coeffs_full) < cleanup] = 0.0

        nz = np.nonzero(coeffs_full)[0]
        if nz.size == 0:
            continue
        cvals = coeffs_full[nz]
        amax = float(np.max(np.abs(cvals)))
        amin = float(np.min(np.abs(cvals)))
        if amax / amin > dyn_max:
            continue

        if seen_hashes is not None:
            key = (
                tuple(nz.tolist()),
                tuple(np.round(cvals / amax, 6).tolist()),
                round(final_rhs / amax, 6),
            )
            if key in seen_hashes:
                continue
            seen_hashes.add(key)

        cuts.append({'cols': nz, 'coeffs': cvals, 'rhs': float(final_rhs)})

    return cuts


class CutPool:
    """Sparse cut pool with hash-based deduplication.

    Stores cuts as (row, column, value) triples so the cut matrix M
    (|pool| by n) can be assembled as a single scipy CSR matrix
    """

    def __init__(self, n):
        """Initialize an empty pool.
        """
        self.n = n
        self._rows = []
        self._cols = []
        self._vals = []
        self._rhs = []
        self.seen = set()
        self._M = None
        self._rhs_arr = None
        self._cache_m = -1

    def __len__(self):
        return len(self._rhs)

    def add(self, cut):
        """Insert a cut if it is not a duplicate.
        """
        cols, coeffs, rhs_val = cut['cols'], cut['coeffs'], cut['rhs']
        amax = float(np.max(np.abs(coeffs)))
        if amax < 1e-12:
            return False
        key = (
            tuple(cols.tolist()),
            tuple(np.round(coeffs / amax, 6).tolist()),
            round(rhs_val / amax, 6),
        )
        if key in self.seen:
            return False
        self.seen.add(key)
        i = len(self._rhs)
        self._rows.extend([i] * len(cols))
        self._cols.extend(int(c) for c in cols)
        self._vals.extend(float(v) for v in coeffs)
        self._rhs.append(float(rhs_val))
        self._cache_m = -1
        return True

    def add_many(self, cuts):
        """Insert a list of cuts, deduplicating each.
        """
        return sum(1 for c in cuts if self.add(c))

    def build_M(self):
        """Build or fetch the cached (M, alpha0) representation.

        Returns:
            Tuple (M, alpha0) where M is a scipy.sparse.csr_matrix of
            shape (|pool|, n) holding cut coefficients, and alpha0 is a
            numpy array of length |pool| holding cut right-hand sides.
        """
        m = len(self._rhs)
        if self._cache_m == m:
            return self._M, self._rhs_arr
        if m == 0:
            self._M = sp.csr_matrix((0, self.n))
            self._rhs_arr = np.zeros(0)
        else:
            self._M = sp.csr_matrix(
                (self._vals, (self._rows, self._cols)),
                shape=(m, self.n))
            self._rhs_arr = np.array(self._rhs)
        self._cache_m = m
        return self._M, self._rhs_arr

    def add_to_model(self, model):
        """Append all pool cuts to a Gurobi model as >= constraints.
        """
        M, rhs_arr = self.build_M()
        vars_arr = model.getVars()
        added = []
        if M.shape[0] == 0:
            return added
        indptr, indices, data = M.indptr, M.indices, M.data
        for i in range(M.shape[0]):
            start, end = indptr[i], indptr[i + 1]
            expr = gp.LinExpr(data[start:end].tolist(),
                              [vars_arr[j] for j in indices[start:end]])
            added.append(model.addConstr(expr >= rhs_arr[i]))
        model.update()
        return added


def relax_and_cut_fast(model, UB, *,
                       L=10, mu=0.01, I_max=100, K=1,
                       verbose=True, log_interval=25):
    """Fast variant of relax-and-cut for GMICs.

    Parameters:
        model: the MIP model (Gurobi)
        UB: upper bound on the optimal value of the first GMIC closure,
            used in the Polyak step
        L: number of main iterations
        mu: Polyak step-size scaling factor
        I_max: subgradient iterations per main iteration
        K: cut generation interval inside the subgradient loop
        verbose: if True, print per-iteration progress.
        log_interval: print every log_interval subgradient steps.

    Returns:
        A Gurobi LP relaxation of model augmented with all pool cuts.
        Re-optimizing it yields the relax-and-cut bound. Returns None
        if the initial LP is not optimal.
    """
    lp_model = model.relax()
    lp_model.setParam('OutputFlag', 0)
    lp_model.optimize()
    if lp_model.status != GRB.OPTIMAL:
        return None

    lp_vars = lp_model.getVars()
    n = len(lp_vars)
    c_orig = np.array(lp_model.getAttr('Obj', lp_vars))

    pool = CutPool(n)
    pool.add_many(generate_rank1_gmics(lp_model, model))

    for main_iter in range(L):
        if verbose:
            print(f"Main iter {main_iter + 1}/{L}   pool={len(pool)}")

        # LLM Debug Disclaimer: Build the large LP fresh from model.relax() rather than
        # copying lp_model, which would carry the old Lagrangian
        # objective from the previous inner loop.
        large_lp = model.relax()
        large_lp.setParam('OutputFlag', 0)
        added = pool.add_to_model(large_lp)
        large_lp.optimize()
        if large_lp.status != GRB.OPTIMAL:
            if verbose:
                print(f"  Large LP status {large_lp.status}, stopping.")
            break

        u = np.maximum(np.array([cr.Pi for cr in added]), 0.0)

        for i in range(I_max):
            M, alpha0 = pool.build_M()
            if len(u) < M.shape[0]:
                u = np.concatenate([u, np.zeros(M.shape[0] - len(u))])

            obj_coeffs = c_orig - M.T.dot(u)
            lp_model.setAttr('Obj', lp_vars, obj_coeffs.tolist())
            lp_model.optimize()

            if lp_model.status == GRB.UNBOUNDED:
                if verbose:
                    print(f"  Inner LP unbounded at step {i}; breaking inner loop.")
                break
            if lp_model.status != GRB.OPTIMAL:
                break

            x_vals = np.array(lp_model.getAttr('X', lp_vars))
            L_u = float(lp_model.ObjVal) + float(u @ alpha0)

            s = alpha0 - M.dot(x_vals)
            s_norm_sq = float(s @ s)
            step = (mu * max(UB - L_u, 0.0) / s_norm_sq) if s_norm_sq > 1e-12 else 0.0
            u = np.maximum(u + step * s, 0.0)

            if K > 0 and (i % K == 0):
                new_cuts = generate_rank1_gmics(lp_model, model)
                added_n = pool.add_many(new_cuts)
                if added_n:
                    u = np.concatenate([u, np.zeros(added_n)])

            if verbose and (i % log_interval == 0):
                print(f"  step {i:4d}/{I_max}  L(u)={L_u:.2f}  "
                      f"pool={len(pool)}  s_norm_sq={s_norm_sq:.2e}")

    P_prime = model.relax()
    P_prime.setParam('OutputFlag', 0)
    pool.add_to_model(P_prime)
    return P_prime


def relax_and_cut_subg(model, UB, *,
                       mu0=10.0, I_max=10000, K=10, u_pad_interval=50,
                       p_avg=100, bad_threshold=10, mu_cap=100.0,
                       verbose=True, log_interval=500):
    """Subgradient variant of relax-and-cut for GMICs

    Parameters:
        model: the MIP model (Gurobi)
        UB: upper bound on the optimal value of the first GMIC closure,
            used in the Polyak step
        L: number of main iterations
        mu0: Initial Polyak step-size scaling factor
        I_max: subgradient iterations per main iteration
        K: cut generation interval inside the subgradient loop
        verbose: if True, print per-iteration progress.
        log_interval: print every log_interval subgradient steps.
        u_pad_interval: how often to append zero entries to u for cuts
            added since the last padding
        p_avg: window length for the average-improvement check
        bad_threshold: number of consecutive iterations satisfying
            L(u) < bestLB - Delta that trigger halve-and-backtrack
        mu_cap: cap on mu after the 10x branch of the ladder fires.
        verbose: if True, print per-iteration progress.
        log_interval: print every log_interval subgradient steps

    Returns:
        A Gurobi LP relaxation augmented with all pool cuts, or None if
        the initial LP fails.
    """
    lp_model = model.relax()
    lp_model.setParam('OutputFlag', 0)
    lp_model.optimize()
    if lp_model.status != GRB.OPTIMAL:
        return None

    lp_vars = lp_model.getVars()
    n = len(lp_vars)
    c_orig = np.array(lp_model.getAttr('Obj', lp_vars))

    pool = CutPool(n)
    pool.add_many(generate_rank1_gmics(lp_model, model))

    u = np.zeros(len(pool))
    u_best = u.copy()
    mu = mu0
    bestLB = -np.inf
    bestLB_at_window_start = -np.inf
    LB_history = []
    bad_count = 0
    pending_new_cuts = 0

    for i in range(I_max):
        M, alpha0 = pool.build_M()
        if len(u) < M.shape[0]:
            u_full = np.concatenate([u, np.zeros(M.shape[0] - len(u))])
        else:
            u_full = u

        obj_coeffs = c_orig - M.T.dot(u_full)
        lp_model.setAttr('Obj', lp_vars, obj_coeffs.tolist())
        lp_model.optimize()

        if lp_model.status == GRB.UNBOUNDED:
            if verbose:
                print(f"  Inner LP unbounded at step {i}; breaking.")
            break
        if lp_model.status != GRB.OPTIMAL:
            break

        x_vals = np.array(lp_model.getAttr('X', lp_vars))
        L_u = float(lp_model.ObjVal) + float(u_full @ alpha0)
        LB_history.append(L_u)

        if L_u > bestLB:
            bestLB = L_u
            u_best = u.copy()

        s_full = alpha0 - M.dot(x_vals)
        s_active = s_full[:len(u)]
        s_norm_sq = float(s_active @ s_active)
        step = (mu * max(UB - L_u, 0.0) / s_norm_sq) if s_norm_sq > 1e-12 else 0.0
        u = np.maximum(u + step * s_active, 0.0)

        Delta = max(UB - bestLB, 1e-10)
        if L_u < bestLB - Delta:
            bad_count += 1
            if bad_count >= bad_threshold:
                mu *= 0.5
                u = u_best.copy()
                bad_count = 0
        else:
            bad_count = 0

        if (i + 1) % p_avg == 0 and len(LB_history) >= p_avg:
            window_improvement = bestLB - bestLB_at_window_start
            if window_improvement < 0.01 * Delta:
                avgLB = float(np.mean(LB_history[-p_avg:]))
                gap_to_avg = bestLB - avgLB
                if gap_to_avg < 0.001 * Delta:
                    mu *= 10.0
                elif gap_to_avg < 0.01 * Delta:
                    mu *= 2.0
                else:
                    mu *= 0.5
                mu = min(mu, mu_cap)
            bestLB_at_window_start = bestLB

        if K > 0 and (i % K == 0):
            new_cuts = generate_rank1_gmics(lp_model, model)
            pending_new_cuts += pool.add_many(new_cuts)

        if (i + 1) % u_pad_interval == 0 and pending_new_cuts > 0:
            u = np.concatenate([u, np.zeros(pending_new_cuts)])
            u_best = np.concatenate([u_best, np.zeros(pending_new_cuts)])
            pending_new_cuts = 0

        if verbose and (i % log_interval == 0):
            print(f"  step {i:5d}/{I_max}  L(u)={L_u:.2f}  best={bestLB:.2f}  "
                  f"pool={len(pool)}  mu={mu:.3f}")

    P_prime = model.relax()
    P_prime.setParam('OutputFlag', 0)
    pool.add_to_model(P_prime)
    return P_prime


def run_1gmi_baseline(mps_filepath, known_opt, gp_env=None):
    """Run the standard rank-1 GMI baseline on a single instance.

    Solves the LP relaxation, reads one round of rank-1 GMICs from the
    optimal basis, adds them all back to the LP, re-solves, and reports
    the percentage integrality gap closed.

    Args:
        mps_filepath: path to an MPS file containing the MIP.
        known_opt: best known integer optimal objective value.
        gp_env: optional pre-configured Gurobi environment. Creates a
            fresh one if None.

    Returns:
        Percentage gap closed (float), or None if the initial LP fails.
    """
    if gp_env is None:
        gp_env = gp.Env()
    model = gp.read(mps_filepath, env=gp_env)
    model.setParam('OutputFlag', 0)

    lp_model = model.relax()
    lp_model.setParam('OutputFlag', 0)
    lp_model.optimize()
    if lp_model.status != GRB.OPTIMAL:
        return None
    z_0 = lp_model.ObjVal
    print(f"Initial LP (z_0): {z_0:.4f}")

    cuts = generate_rank1_gmics(lp_model, model)
    print(f"Generated {len(cuts)} stable rank-1 GMICs.")

    lp_vars = lp_model.getVars()
    for cut in cuts:
        expr = gp.LinExpr(cut['coeffs'].tolist(),
                          [lp_vars[i] for i in cut['cols']])
        lp_model.addConstr(expr >= cut['rhs'])
    lp_model.optimize()
    z_1 = lp_model.ObjVal
    print(f"Final LP after 1 round (z_1): {z_1:.4f}")

    if abs(known_opt - z_0) < 1e-6:
        return 0.0
    gap_closed = 100 * (z_1 - z_0) / (known_opt - z_0)
    print(f"Baseline % integrality gap closed: {gap_closed:.2f}%")
    return gap_closed


def benchmark_instance(mps_filepath, known_opt, variant='fast',
                       gp_env=None, **kwargs):
    """Run a relax-and-cut variant on a single MPS instance.

    Parameters:
        mps_filepath: path to MPS file
        known_opt: best known integer optimal objective value
        variant: one of 'fast' or 'subg'
        gp_env: optional pre-configured Gurobi environment.
        **kwargs: arguments that are forwarded to the relax_and_cut function.

    Returns:
        Tuple (gap_closed_pct, runtime_seconds), or None if the LP
        fails.
    """
    print(f"\n{'='*50}\nBenchmarking: {mps_filepath}   (variant={variant})\n{'='*50}")
    if gp_env is None:
        gp_env = gp.Env()
        gp_env.setParam('OutputFlag', 0)
    model = gp.read(mps_filepath, env=gp_env)
    model.setParam('OutputFlag', 0)

    lp_model = model.relax()
    lp_model.setParam('OutputFlag', 0)
    lp_model.optimize()
    if lp_model.status != GRB.OPTIMAL:
        print("Initial LP did not solve.")
        return None
    z_0 = lp_model.ObjVal
    print(f"Opt (z*): {known_opt:.4f}   Initial LP (z_0): {z_0:.4f}")

    runners = {
        'fast':   lambda m, ub: relax_and_cut_fast(m, ub, **kwargs),
        'subg':   lambda m, ub: relax_and_cut_subg(m, ub, **kwargs),
    }
    if variant not in runners:
        raise ValueError(f"Unknown variant {variant!r}; pick from {sorted(runners)}")

    t0 = time.time()
    P_prime = runners[variant](model, known_opt)
    runtime = time.time() - t0

    if P_prime is None:
        print("Variant returned None.")
        return None
    P_prime.optimize()
    z = P_prime.ObjVal
    if abs(known_opt - z_0) < 1e-6:
        gap_closed = 0.0
    else:
        gap_closed = 100 * (z - z_0) / (known_opt - z_0)
    print(f"Final LP (z): {z:.4f}")
    print(f"Runtime: {runtime:.2f}s   % gap closed: {gap_closed:.2f}%")
    return gap_closed, runtime
