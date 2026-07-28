import json
import os
import pathlib
import time
import numpy as np
import pybamm
from scipy.optimize import differential_evolution
from batterytwin.digital_twin.dfn_model import build_dfn_model

def _load_calib_config():
    config_path = pathlib.Path(__file__).resolve().parents[3] / 'configs' / 'calibration.default.json'
    if config_path.exists():
        with open(config_path, encoding='utf-8') as f:
            return json.load(f)
    return {}

def _get_param_bounds(config: dict):
    pb = config.get('param_bounds', {})
    return [(pb[k][0], pb[k][1]) for k in PARAM_ORDER if k in pb] or PARAM_BOUNDS_DEFAULT

def _get_de_config(config: dict):
    dc = config.get('de_config', {})
    if not dc:
        return DE_CONFIG_DEFAULT.copy()
    merged = DE_CONFIG_DEFAULT.copy()
    merged.update(dc)
    if isinstance(merged.get('mutation'), list):
        merged['mutation'] = tuple(merged['mutation'])
    return merged

def _get_obj_weights(config: dict):
    return config.get('objective_weights', {'beta_T': 0.2, 'head_tail_weight': 10.0, 'grad_weight': 0.5})
PARAM_ORDER = ['h_el', 'w_el', 'kappa_scale', 'c_e0', 'i0n_scale', 'i0p_scale', 'Dn_scale', 'Dp_scale', 'De_scale', 'tplus', 'h_total']
PARAM_BOUNDS_DEFAULT = [(0.06, 0.07), (1.0, 1.5), (4.0, 15.0), (1000.0, 1900.0), (5.0, 30.0), (10.0, 80.0), (0.1, 5.0), (0.1, 5.0), (0.3, 4.0), (0.2, 0.55), (5.0, 200.0)]
DE_CONFIG_DEFAULT = dict(strategy='best1bin', maxiter=25, popsize=12, tol=0.001, mutation=(0.5, 1.0), recombination=0.7, polish=True, disp=True, workers=-1, updating='deferred')
PARAM_BOUNDS = PARAM_BOUNDS_DEFAULT
DE_CONFIG = DE_CONFIG_DEFAULT
PENALTY = 1000000.0

def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def build_param_update(x, base_params):
    d = dict(zip(PARAM_ORDER, map(float, x)))
    sigma_e_base = base_params['Electrolyte conductivity [S.m-1]']
    De_base = base_params['Electrolyte diffusivity [m2.s-1]']
    Dn_base = base_params['Negative particle diffusivity [m2.s-1]']
    Dp_base = base_params['Positive particle diffusivity [m2.s-1]']
    i0_n_base = base_params['Negative electrode exchange-current density [A.m-2]']
    i0_p_base = base_params['Positive electrode exchange-current density [A.m-2]']
    ks = d['kappa_scale']
    De_s = d['De_scale']
    Dn_s = d['Dn_scale']
    Dp_s = d['Dp_scale']
    i0n_s = d['i0n_scale']
    i0p_s = d['i0p_scale']

    def sigma_e_scaled(c_e, T):
        return ks * sigma_e_base(c_e, T)

    def De_scaled(c_e, T):
        return De_s * De_base(c_e, T)

    def Dn_scaled(sto, T):
        return Dn_s * Dn_base(sto, T)

    def Dp_scaled(sto, T):
        return Dp_s * Dp_base(sto, T)

    def i0_n_scaled(c_e, c_s_surf, c_s_max, T):
        return i0n_s * i0_n_base(c_e, c_s_surf, c_s_max, T)

    def i0_p_scaled(c_e, c_s_surf, c_s_max, T):
        return i0p_s * i0_p_base(c_e, c_s_surf, c_s_max, T)
    update = {'Electrode height [m]': d['h_el'], 'Electrode width [m]': d['w_el'], 'Electrolyte conductivity [S.m-1]': sigma_e_scaled, 'Electrolyte diffusivity [m2.s-1]': De_scaled, 'Initial concentration in electrolyte [mol.m-3]': d['c_e0'], 'Cation transference number': d['tplus'], 'Negative electrode exchange-current density [A.m-2]': i0_n_scaled, 'Positive electrode exchange-current density [A.m-2]': i0_p_scaled, 'Negative particle diffusivity [m2.s-1]': Dn_scaled, 'Positive particle diffusivity [m2.s-1]': Dp_scaled}
    if 'h_total' in d:
        update['Total heat transfer coefficient [W.m-2.K-1]'] = d['h_total']
    return update

def simulate_once(x, model, base_params, t_eval, current_fun):
    local_params = base_params.copy()
    local_params.update(build_param_update(x, base_params))
    local_params['Current function [A]'] = current_fun
    sim = pybamm.Simulation(model, parameter_values=local_params)

    def _make_casadi(mode):
        try:
            return pybamm.CasadiSolver(mode=mode, atol=1e-06, rtol=1e-06, dt_max=5.0, return_solution_if_failed_early=True)
        except TypeError:
            return pybamm.CasadiSolver(mode=mode, atol=1e-06, rtol=1e-06, dt_max=5.0)
    solvers = [None, _make_casadi('fast'), _make_casadi('safe')]
    sol = None
    for solver in solvers:
        try:
            sol = sim.solve(t_eval) if solver is None else sim.solve(t_eval, solver=solver)
            break
        except Exception:
            continue
    if sol is None:
        return (None, None)
    try:

        def _extract(var):
            arr = np.asarray(var(t_eval))
            if arr.ndim > 1:
                arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
            return arr
        V_sim = _extract(sol['Terminal voltage [V]'])
        T_sim = None
        for t_key in ('Cell temperature [K]', 'Volume-averaged cell temperature [K]', 'X-averaged cell temperature [K]'):
            try:
                T_sim = _extract(sol[t_key])
                break
            except KeyError:
                pass
        return (V_sim, T_sim)
    except Exception:
        return (None, None)

def make_objective(model, base_params, t_eval, current_fun, v_real, T_real=None, alpha_V=1.0, beta_T=0.2, head_tail_weight=10.0, grad_weight=0.5, cutoff_weight=0.0, cutoff_tol_s=20.0):
    v_real = np.asarray(v_real).reshape(-1)
    n = len(v_real)
    n_head = max(10, int(0.1 * n))
    n_tail = max(10, int(0.1 * n))
    v_cutoff = float(v_real[-1])
    t_end_real = float(t_eval[-1])
    if T_real is not None:
        T_real = np.asarray(T_real).reshape(-1)

    def objective(x):
        try:
            V_sim, T_sim = simulate_once(x, model, base_params, t_eval, current_fun)
            if V_sim is None:
                return PENALTY
            if np.any(np.isnan(V_sim)) or V_sim.min() < 1.0 or V_sim.max() > 5.0:
                return PENALTY
            V_sim = V_sim.reshape(-1)
            head_v = _mae(V_sim[:n_head], v_real[:n_head])
            mid_v = _mae(V_sim[n_head:n - n_tail], v_real[n_head:n - n_tail])
            tail_v = _mae(V_sim[n - n_tail:], v_real[n - n_tail:])
            grad_v = _mae(np.diff(V_sim), np.diff(v_real))
            loss = head_tail_weight * head_v + 1.0 * mid_v + head_tail_weight * tail_v + grad_weight * grad_v
            if cutoff_weight > 0:
                crossings = np.where(V_sim <= v_cutoff)[0]
                if len(crossings) > 0:
                    t_cutoff_sim = float(t_eval[crossings[0]])
                elif len(V_sim) >= 2:
                    dV_dt = (V_sim[-1] - V_sim[-2]) / max(1e-06, t_eval[-1] - t_eval[-2])
                    if dV_dt < -1e-08:
                        t_cutoff_sim = float(t_eval[-1]) + (v_cutoff - float(V_sim[-1])) / dV_dt
                    else:
                        t_cutoff_sim = t_end_real * 3.0
                else:
                    t_cutoff_sim = t_end_real * 3.0
                dt = abs(t_cutoff_sim - t_end_real)
                if dt > cutoff_tol_s:
                    loss += cutoff_weight * (dt - cutoff_tol_s) / t_end_real
            if T_real is not None and T_sim is not None:
                T_sim = T_sim.reshape(-1)
                if not np.any(np.isnan(T_sim)):
                    loss += beta_T * _mae(T_sim, T_real)
            return loss
        except Exception:
            return PENALTY
    return objective

class _Stage1Objective:

    def __init__(self, t_eval, current_data, v_real, T_real, ambient_K, beta_T, head_tail_weight, grad_weight, cutoff_weight, cutoff_tol_s):
        self.t_eval = np.asarray(t_eval)
        self.current_data = np.asarray(current_data)
        self.v_real = np.asarray(v_real).reshape(-1)
        self.T_real = np.asarray(T_real).reshape(-1) if T_real is not None else None
        self.ambient_K = ambient_K
        self.beta_T = beta_T
        self.head_tail_weight = head_tail_weight
        self.grad_weight = grad_weight
        self.cutoff_weight = cutoff_weight
        self.cutoff_tol_s = cutoff_tol_s
        self._model = None
        self._params = None
        self._current_fun = None

    def _init_model(self):
        model, params = build_dfn_model(self.ambient_K)
        self._model = model
        self._params = params
        self._current_fun = pybamm.Interpolant(self.current_data[:, 0], self.current_data[:, 1], pybamm.t)

    def __call__(self, x):
        if self._model is None:
            self._init_model()
        try:
            V_sim, T_sim = simulate_once(x, self._model, self._params, self.t_eval, self._current_fun)
            if V_sim is None:
                return PENALTY
            if np.any(np.isnan(V_sim)) or V_sim.min() < 1.0 or V_sim.max() > 5.0:
                return PENALTY
            V_sim = V_sim.reshape(-1)
            n = len(self.v_real)
            n_head = max(10, int(0.1 * n))
            n_tail = max(10, int(0.1 * n))
            head_v = _mae(V_sim[:n_head], self.v_real[:n_head])
            mid_v = _mae(V_sim[n_head:n - n_tail], self.v_real[n_head:n - n_tail])
            tail_v = _mae(V_sim[n - n_tail:], self.v_real[n - n_tail:])
            grad_v = _mae(np.diff(V_sim), np.diff(self.v_real))
            loss = self.head_tail_weight * head_v + 1.0 * mid_v + self.head_tail_weight * tail_v + self.grad_weight * grad_v
            if self.cutoff_weight > 0:
                v_cutoff = float(self.v_real[-1])
                t_end_real = float(self.t_eval[-1])
                crossings = np.where(V_sim <= v_cutoff)[0]
                if len(crossings) > 0:
                    t_cutoff_sim = float(self.t_eval[crossings[0]])
                elif len(V_sim) >= 2:
                    dV_dt = (V_sim[-1] - V_sim[-2]) / max(1e-06, self.t_eval[-1] - self.t_eval[-2])
                    if dV_dt < -1e-08:
                        t_cutoff_sim = float(self.t_eval[-1]) + (v_cutoff - float(V_sim[-1])) / dV_dt
                    else:
                        t_cutoff_sim = t_end_real * 3.0
                else:
                    t_cutoff_sim = t_end_real * 3.0
                dt = abs(t_cutoff_sim - t_end_real)
                if dt > self.cutoff_tol_s:
                    loss += self.cutoff_weight * (dt - self.cutoff_tol_s) / t_end_real
            if self.T_real is not None and T_sim is not None:
                T_sim = T_sim.reshape(-1)
                if not np.any(np.isnan(T_sim)):
                    loss += self.beta_T * _mae(T_sim, self.T_real)
            return loss
        except Exception:
            return PENALTY

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None
        state['_params'] = None
        state['_current_fun'] = None
        return state

def run_parameter_search(model, base_params, t_eval, current_fun, v_real, T_real=None, beta_T=0.2, de_config=None, bounds=None, verbose=True, current_data=None, ambient_K=298.15):
    config = _load_calib_config()
    if bounds is None:
        bounds = _get_param_bounds(config)
    base_de = _get_de_config(config)
    if de_config is not None:
        base_de.update(de_config)
        if isinstance(base_de.get('mutation'), list):
            base_de['mutation'] = tuple(base_de['mutation'])
    de_config = base_de
    obj_weights = _get_obj_weights(config)
    if beta_T == 0.2:
        beta_T = obj_weights.get('beta_T', 0.2)
    workers = de_config.get('workers', 1)
    if workers == -1:
        workers = max(1, os.cpu_count() // 2)
        de_config['workers'] = workers
    if workers != 1:
        de_config['updating'] = 'deferred'
    if verbose:
        print(f"[calib_config] iteration={config.get('_iteration', 0)}  beta_T={beta_T}  maxiter={de_config.get('maxiter')}  popsize={de_config.get('popsize')}  workers={workers}")
    use_parallel = workers != 1 and current_data is not None
    if use_parallel:
        objective = _Stage1Objective(t_eval=t_eval, current_data=current_data, v_real=v_real, T_real=T_real, ambient_K=ambient_K, beta_T=beta_T, head_tail_weight=obj_weights.get('head_tail_weight', 10.0), grad_weight=obj_weights.get('grad_weight', 0.5), cutoff_weight=obj_weights.get('cutoff_weight', 0.0), cutoff_tol_s=obj_weights.get('cutoff_tol_s', 20.0))
        if verbose:
            print(f'  Using parallel-safe _Stage1Objective (workers={workers})')
    else:
        if workers != 1 and current_data is None:
            de_config['workers'] = 1
            de_config['updating'] = 'immediate'
            if verbose:
                print('  WARNING: workers>1 but current_data not provided, falling back to workers=1')
        objective = make_objective(model, base_params, t_eval, current_fun, v_real, T_real, beta_T=beta_T, head_tail_weight=obj_weights.get('head_tail_weight', 10.0), grad_weight=obj_weights.get('grad_weight', 0.5), cutoff_weight=obj_weights.get('cutoff_weight', 0.0), cutoff_tol_s=obj_weights.get('cutoff_tol_s', 20.0))
    want_polish = de_config.get('polish', False)
    do_manual_polish = want_polish and de_config.get('workers', 1) != 1
    if do_manual_polish:
        de_config['polish'] = False
    t0 = time.time()
    result = differential_evolution(objective, bounds=bounds, **de_config)
    if do_manual_polish:
        from scipy.optimize import minimize
        if verbose:
            print(f'  DE done (obj={result.fun:.6f}), running L-BFGS-B polish...')
        polish_obj = make_objective(model, base_params, t_eval, current_fun, v_real, T_real, beta_T=beta_T, head_tail_weight=obj_weights.get('head_tail_weight', 10.0), grad_weight=obj_weights.get('grad_weight', 0.5), cutoff_weight=obj_weights.get('cutoff_weight', 0.0), cutoff_tol_s=obj_weights.get('cutoff_tol_s', 20.0))
        polish_result = minimize(polish_obj, result.x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 15})
        if polish_result.fun < result.fun:
            result.x = polish_result.x
            result.fun = polish_result.fun
            if verbose:
                print(f'  Polish improved: {polish_result.fun:.6f}')
        elif verbose:
            print(f'  Polish did not improve ({polish_result.fun:.6f} >= {result.fun:.6f})')
    elapsed = time.time() - t0
    x_best = result.x
    if verbose:
        print(f'Best objective : {result.fun:.6f}')
        print(f'Elapsed        : {elapsed:.1f} s')
        print('Best x_dict    :', dict(zip(PARAM_ORDER, [round(v, 6) for v in x_best])))
    return {'x_best': [float(v) for v in x_best], 'x_best_dict': {k: float(v) for k, v in zip(PARAM_ORDER, x_best)}, 'best_objective': float(result.fun), 'elapsed_s': elapsed, 'param_update': build_param_update(x_best, base_params)}
