import argparse
import json
import os
import pathlib
import time
from datetime import datetime
import numpy as np
import pandas as pd
import pybamm
from scipy.optimize import differential_evolution
from batterytwin.digital_twin.dfn_model import build_dfn_model
from .optimization import build_param_update

def extract_soh_from_csv(csv_path: str, n_cycles: int):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    cycles = sorted(df['Cycle'].unique())[:n_cycles]
    Q_list = []
    C_rate_est = None
    for c in cycles:
        dis = df[(df['Cycle'] == c) & (df['Current'] < 0)]
        if dis.empty:
            Q_list.append(np.nan)
            continue
        cap = abs(float(dis['Capacity'].values[-1] - dis['Capacity'].values[0]))
        Q_list.append(cap)
        if C_rate_est is None and cap > 0:
            avg_current = abs(dis['Current'].mean())
            C_rate_est = avg_current / cap
    Q_real = np.array(Q_list)
    Q1 = Q_real[0]
    soh_real = Q_real / Q1
    return (list(cycles), soh_real, Q_real, C_rate_est or 1.0)
DEGRADATION_PARAM_ORDER = ['SEI kinetic rate constant [m.s-1]', 'Lithium plating kinetic rate constant [m.s-1]', 'Dead lithium decay constant [s-1]']
DEGRADATION_BOUNDS_LOG10 = [(-16, -11), (-13, -8), (-10, -5)]

class _DegradationObjective:

    def __init__(self, stage1_params_json, csv_path, soh_real, n_cycles, C_rate, V_min, V_max, ambient_K):
        self.stage1_params_json = stage1_params_json
        self.csv_path = csv_path
        self.soh_real = soh_real
        self.n_cycles = n_cycles
        self.C_rate = C_rate
        self.V_min = V_min
        self.V_max = V_max
        self.ambient_K = ambient_K
        self._model = None
        self._params = None

    def _init_model(self):
        model, params = build_dfn_model(self.ambient_K, degradation=True)
        stage1 = json.loads(self.stage1_params_json)
        params.update(build_param_update(stage1['x_best'], params))
        for ocp_key, shift_val in stage1.get('ocp_shifts', {}).items():
            if ocp_key in params and callable(params[ocp_key]):
                base_fn = params[ocp_key]
                params[ocp_key] = lambda *a, _f=base_fn, _s=shift_val: _f(*a) + _s
        self._model = model
        self._params = params

    def __call__(self, x_log10):
        if self._model is None:
            self._init_model()
        PENALTY = 1000000.0
        params = self._params.copy()
        for name, val_log in zip(DEGRADATION_PARAM_ORDER, x_log10):
            params[name] = 10.0 ** val_log
        cycle_steps = (f'Discharge at {self.C_rate}C until {self.V_min} V', 'Rest for 10 minutes', f'Charge at {self.C_rate}C until {self.V_max} V', f'Hold at {self.V_max} V until C/50')
        experiment = pybamm.Experiment([cycle_steps] * self.n_cycles)
        sim = pybamm.Simulation(self._model, parameter_values=params, experiment=experiment)
        try:
            solver = pybamm.CasadiSolver(mode='safe', dt_max=10.0)
            sol = sim.solve(solver=solver)
        except Exception:
            return PENALTY
        soh_list = []
        Q_init = None
        for cycle in sol.cycles:
            try:
                step = cycle.steps[0]
            except (AttributeError, IndexError):
                step = cycle
            try:
                Q_arr = np.asarray(step['Discharge capacity [A.h]'].entries)
                Q_cycle = float(Q_arr[-1]) if len(Q_arr) > 0 else 0.0
            except Exception:
                Q_cycle = 0.0
            if Q_init is None:
                Q_init = Q_cycle
            soh_list.append(Q_cycle / Q_init if Q_init > 0 else 1.0)
        soh_sim = np.array(soh_list)
        n = min(len(soh_sim), len(self.soh_real))
        if n < 3:
            return PENALTY
        mae = float(np.mean(np.abs(soh_sim[:n] - self.soh_real[:n])))
        return mae

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_model'] = None
        state['_params'] = None
        return state

def calibrate_degradation(params_json_path: str, csv_path: str, n_cycles: int=100, popsize: int=10, maxiter: int=25, workers: int=-1, verbose: bool=True) -> dict:
    with open(params_json_path, encoding='utf-8') as f:
        stage1_params = json.load(f)
    cell_id = stage1_params['cell_id']
    out_dir = pathlib.Path(params_json_path).parent
    if verbose:
        print(f'[Stage 2] Cell {cell_id}, fitting degradation over {n_cycles} cycles')
    cycle_ids, soh_real, Q_real, C_rate_est = extract_soh_from_csv(csv_path, n_cycles)
    actual_n = len(soh_real)
    if verbose:
        print(f'  Real data: {actual_n} cycles, C_rate~{C_rate_est:.2f}')
        print(f'  SOH: {soh_real[0]:.4f} -> {soh_real[-1]:.4f} ({(1 - soh_real[-1]) * 100:.2f}% degradation)')
        print(f'  Q1 = {Q_real[0]:.4f} Ah')
    ambient_K = stage1_params.get('ambient_temperature_k', 298.15)
    _, temp_params = build_dfn_model(ambient_K, degradation=True)
    V_min = float(temp_params['Lower voltage cut-off [V]'])
    V_max = float(temp_params['Upper voltage cut-off [V]'])
    if workers == -1:
        workers = max(1, os.cpu_count() // 2)
    if verbose:
        print(f'  Workers: {workers}, popsize={popsize}, maxiter={maxiter}')
    stage1_json_str = json.dumps(stage1_params)
    objective = _DegradationObjective(stage1_params_json=stage1_json_str, csv_path=csv_path, soh_real=soh_real, n_cycles=actual_n, C_rate=C_rate_est, V_min=V_min, V_max=V_max, ambient_K=ambient_K)
    if verbose:
        print(f'  Starting DE optimization (log10 space)...')
        print(f'  Bounds: {DEGRADATION_BOUNDS_LOG10}')
    de_config = dict(strategy='best1bin', maxiter=maxiter, popsize=popsize, tol=1e-05, mutation=(0.5, 1.0), recombination=0.7, polish=False, disp=True, workers=workers, updating='deferred')
    t0 = time.time()
    result = differential_evolution(objective, bounds=DEGRADATION_BOUNDS_LOG10, **de_config)
    elapsed = time.time() - t0
    x_best_log10 = result.x
    x_best_linear = {name: 10.0 ** val for name, val in zip(DEGRADATION_PARAM_ORDER, x_best_log10)}
    if verbose:
        print(f'  Best objective (MAE SOH): {result.fun:.6f}')
        print(f'  Elapsed: {elapsed:.1f} s')
        for name, val in x_best_linear.items():
            print(f'    {name} = {val:.2e}')
    model, params = build_dfn_model(ambient_K, degradation=True)
    params.update(build_param_update(stage1_params['x_best'], params))
    for ocp_key, shift_val in stage1_params.get('ocp_shifts', {}).items():
        if ocp_key in params and callable(params[ocp_key]):
            base_fn = params[ocp_key]
            params[ocp_key] = lambda *a, _f=base_fn, _s=shift_val: _f(*a) + _s
    for name, val_log in zip(DEGRADATION_PARAM_ORDER, x_best_log10):
        params[name] = 10.0 ** val_log
    cycle_steps = (f'Discharge at {C_rate_est}C until {V_min} V', 'Rest for 10 minutes', f'Charge at {C_rate_est}C until {V_max} V', f'Hold at {V_max} V until C/50')
    experiment = pybamm.Experiment([cycle_steps] * actual_n)
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    soh_sim = None
    try:
        solver = pybamm.CasadiSolver(mode='safe', dt_max=10.0)
        sol = sim.solve(solver=solver)
        soh_list = []
        Q_init = None
        for cycle in sol.cycles:
            try:
                step = cycle.steps[0]
            except (AttributeError, IndexError):
                step = cycle
            try:
                Q_arr = np.asarray(step['Discharge capacity [A.h]'].entries)
                Q_cycle = float(Q_arr[-1]) if len(Q_arr) > 0 else 0.0
            except Exception:
                Q_cycle = 0.0
            if Q_init is None:
                Q_init = Q_cycle
            soh_list.append(Q_cycle / Q_init if Q_init > 0 else 1.0)
        soh_sim = np.array(soh_list)
    except Exception as e:
        if verbose:
            print(f'  WARNING: Final simulation failed: {e}')
    if soh_sim is not None:
        n_common = min(len(soh_sim), len(soh_real))
        if n_common >= 3:
            mae_soh = float(np.mean(np.abs(soh_sim[:n_common] - soh_real[:n_common])))
        else:
            mae_soh = None
            if verbose:
                print(f'  WARNING: simulation completed only {len(soh_sim)} cycles')
        if verbose:
            print(f'  Simulation completed {len(soh_sim)}/{actual_n} cycles')
    else:
        mae_soh = None
        n_common = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stage1_params['degradation_params'] = x_best_linear
    stage1_params['degradation_calibration'] = {'n_cycles_fit': actual_n, 'C_rate_est': C_rate_est, 'mae_soh': mae_soh, 'best_objective': float(result.fun), 'elapsed_s': elapsed, 'timestamp': timestamp}
    params_path = pathlib.Path(params_json_path)
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(stage1_params, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f'  Updated params JSON: {params_path}')
    log_path = out_dir / 'stage2_metrics.json'
    if log_path.exists():
        with open(log_path, encoding='utf-8') as f:
            existing_log = json.load(f)
    else:
        existing_log = {}
    existing_log['stage2_timestamp'] = timestamp
    existing_log['stage2_n_cycles_fit'] = actual_n
    existing_log['stage2_C_rate_est'] = C_rate_est
    existing_log['stage2_soh_real_start'] = float(soh_real[0])
    existing_log['stage2_soh_real_end'] = float(soh_real[-1])
    existing_log['stage2_mae_soh'] = mae_soh
    existing_log['stage2_best_objective'] = float(result.fun)
    existing_log['stage2_elapsed_s'] = elapsed
    existing_log['degradation_params'] = x_best_linear
    existing_log['degradation_params_log10'] = {name: float(val) for name, val in zip(DEGRADATION_PARAM_ORDER, x_best_log10)}
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(existing_log, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f'  Updated calibration log: {log_path}')
    if verbose:
        print(f'  All results in: {out_dir}/')
        if mae_soh is not None:
            print(f'  MAE(SOH) = {mae_soh:.4f}')
    return stage1_params
