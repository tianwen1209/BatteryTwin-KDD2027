import numpy as np
import pybamm
from .dfn_model import build_dfn_model
from batterytwin.calibration.optimization import build_param_update

def _safe_extract(sol, var_name, t=None):
    try:
        proc = sol[var_name]
    except KeyError:
        return None
    if t is not None:
        arr = np.asarray(proc(t))
    else:
        arr = np.asarray(proc.entries)
    if arr.ndim > 1:
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
    return arr

def _extract_temperature(sol, t=None):
    for key in ('Cell temperature [K]', 'Volume-averaged cell temperature [K]', 'X-averaged cell temperature [K]'):
        T = _safe_extract(sol, key, t)
        if T is not None:
            return T
    return None

def _extract_charge_scalars_from_cycle(cycle) -> np.ndarray:
    zeros = np.zeros(6, dtype=np.float32)
    try:
        steps = cycle.steps
        if len(steps) < 4:
            return zeros
        cc_step = steps[2]
        cv_step = steps[3]
    except (AttributeError, IndexError):
        return zeros
    cc_V = _safe_extract(cc_step, 'Terminal voltage [V]')
    cc_t = np.asarray(cc_step.t) if hasattr(cc_step, 't') else None
    if cc_V is None or cc_t is None or len(cc_V) < 3:
        return zeros
    cc_V_mean = float(np.mean(cc_V))
    cc_V_std = float(np.std(cc_V))
    cc_dur = float(cc_t[-1] - cc_t[0])
    cv_I = _safe_extract(cv_step, 'Current [A]')
    cv_t = np.asarray(cv_step.t) if hasattr(cv_step, 't') else None
    if cv_I is None or cv_t is None or len(cv_I) < 3:
        return zeros
    cv_I_abs = np.abs(cv_I)
    cv_I_mean = float(np.mean(cv_I_abs))
    cv_I_std = float(np.std(cv_I_abs))
    cv_dur = float(cv_t[-1] - cv_t[0])
    return np.array([cc_V_mean, cc_V_std, cc_dur, cv_I_mean, cv_I_std, cv_dur], dtype=np.float32)

def simulate_aging_trajectory(param_dict: dict, protocol: dict, ambient_temperature_K: float=298.15, extract_internal: bool=False, extract_charge_info: bool=False) -> dict:
    if param_dict.get('physical_baseline'):
        from .dfn_model import build_dfn_model_phys
        model, params = build_dfn_model_phys(param_dict.get('phys_param_set', 'Chen2020'), ambient_temperature_K, degradation=True)
    else:
        model, params = build_dfn_model(ambient_temperature_K, degradation=True)
        x_best = param_dict['x_best']
        param_update = build_param_update(x_best, params)
        params.update(param_update)
    ocp_shifts = param_dict.get('ocp_shifts', {})
    for ocp_key, shift_val in ocp_shifts.items():
        if ocp_key in params and callable(params[ocp_key]):
            base_fn = params[ocp_key]
            params[ocp_key] = lambda *a, _f=base_fn, _s=shift_val: _f(*a) + _s
    deg_params = param_dict.get('degradation_params', {})
    for k, v in deg_params.items():
        params[k] = v
    Q_nom = protocol.get('nominal_capacity_Ah', 4.0)
    params['Nominal cell capacity [A.h]'] = Q_nom
    C_rate_val = protocol.get('C_rate', 1.0)
    C_charge = protocol.get('C_rate_charge', C_rate_val)
    C_discharge = protocol.get('C_rate_discharge', C_rate_val)
    n_cycles = protocol.get('n_cycles', 200)
    V_min = float(params['Lower voltage cut-off [V]'])
    V_max = float(params['Upper voltage cut-off [V]'])
    I_discharge_A = protocol.get('I_discharge_A')
    I_charge_A = protocol.get('I_charge_A')
    dis_str = f'Discharge at {I_discharge_A} A until {V_min} V' if I_discharge_A else f'Discharge at {C_discharge}C until {V_min} V'
    chg_str = f'Charge at {I_charge_A} A until {V_max} V' if I_charge_A else f'Charge at {C_charge}C until {V_max} V'
    cycle_steps = (dis_str, f'Rest for 10 minutes', chg_str, f'Hold at {V_max} V until C/50')
    experiment = pybamm.Experiment([cycle_steps] * n_cycles)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    try:
        solver = pybamm.CasadiSolver(mode='safe', dt_max=10.0, return_solution_if_failed_early=True)
    except TypeError:
        solver = pybamm.CasadiSolver(mode='safe', dt_max=10.0)
    sol = sim.solve(solver=solver)
    soh_list = []
    voltage_curves = []
    time_curves = []
    temperature_curves = []
    charge_scalars_list = [] if extract_charge_info else None
    internal_c_e = [] if extract_internal else None
    internal_c_s = [] if extract_internal else None
    Q_init = None
    for cycle in sol.cycles:
        try:
            step = cycle.steps[0]
        except (AttributeError, IndexError):
            step = cycle
        V = _safe_extract(step, 'Terminal voltage [V]')
        if V is not None:
            voltage_curves.append(V)
            t_step = np.asarray(step.t)
            time_curves.append(t_step - t_step[0])
        T = _extract_temperature(step)
        if T is not None:
            temperature_curves.append(T)
        Q_arr = _safe_extract(step, 'Discharge capacity [A.h]')
        if Q_arr is not None and len(Q_arr) > 0:
            Q_cycle = float(Q_arr[-1])
            if Q_init is None:
                Q_init = Q_cycle
            soh_list.append(Q_cycle / Q_init if Q_init > 0 else 1.0)
        else:
            soh_list.append(np.nan)
        if extract_charge_info:
            scalars = _extract_charge_scalars_from_cycle(cycle)
            charge_scalars_list.append(scalars)
        if extract_internal:
            c_e = _safe_extract(step, 'Electrolyte concentration [mol.m-3]')
            if c_e is not None:
                internal_c_e.append(c_e)
            c_s_n = _safe_extract(step, 'Negative particle surface concentration [mol.m-3]')
            c_s_p = _safe_extract(step, 'Positive particle surface concentration [mol.m-3]')
            if c_s_n is not None and c_s_p is not None:
                internal_c_s.append(np.concatenate([c_s_n, c_s_p]))
            elif c_s_n is not None:
                internal_c_s.append(c_s_n)
    result = {'soh': np.array(soh_list), 'voltage_curves': voltage_curves, 'time_curves': time_curves, 'temperature_curves': temperature_curves}
    if extract_charge_info:
        result['charge_scalars'] = np.array(charge_scalars_list, dtype=np.float32)
    if extract_internal:
        result['internal_states'] = {'c_e': internal_c_e, 'c_s_surf': internal_c_s}
    return result

def simulate_single_cycle(param_dict: dict, t_eval: np.ndarray, current_A: np.ndarray, ambient_temperature_K: float=298.15, nominal_capacity_Ah: float=4.0) -> dict:
    model, params = build_dfn_model(ambient_temperature_K)
    x_best = param_dict['x_best']
    param_update = build_param_update(x_best, params)
    params.update(param_update)
    params['Nominal cell capacity [A.h]'] = nominal_capacity_Ah
    current_fun = pybamm.Interpolant(t_eval, current_A, pybamm.t)
    params['Current function [A]'] = current_fun
    sim = pybamm.Simulation(model, parameter_values=params)
    solvers = [None, pybamm.CasadiSolver(mode='fast', atol=1e-06, rtol=1e-06, dt_max=5.0), pybamm.CasadiSolver(mode='safe', atol=1e-06, rtol=1e-06, dt_max=5.0)]
    sol = None
    for s in solvers:
        try:
            sol = sim.solve(t_eval) if s is None else sim.solve(t_eval, solver=s)
            break
        except Exception:
            continue
    if sol is None:
        return {'voltage': None, 'temperature': None}
    V = _safe_extract(sol, 'Terminal voltage [V]', t=t_eval)
    T = _extract_temperature(sol, t=t_eval)
    return {'voltage': V, 'temperature': T}
