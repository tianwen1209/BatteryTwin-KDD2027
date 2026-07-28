import json
import os
import pathlib
import numpy as np
import pandas as pd
from .config import DEFAULT_SEQUENCE_LENGTH
from batterytwin.digital_twin.simulator import simulate_single_cycle

def resample_cycle_inputs(t_raw, v_raw, i_raw, n_points=DEFAULT_SEQUENCE_LENGTH):
    t_raw = np.asarray(t_raw, dtype=float)
    v_raw = np.asarray(v_raw, dtype=float)
    i_raw = np.asarray(i_raw, dtype=float)
    if len(t_raw) < 2 or not np.all(np.diff(t_raw) > 0):
        raise ValueError(f'resample_cycle_inputs: t_raw must be strictly increasing (len={len(t_raw)})')
    t_grid = np.linspace(t_raw[0], t_raw[-1], n_points)
    v_rs = np.interp(t_grid, t_raw, v_raw)
    i_rs = np.interp(t_grid, t_raw, i_raw)
    return {'t': (t_grid - t_grid[0]).astype(np.float32), 'v': v_rs.astype(np.float32), 'i': (-i_rs).astype(np.float32)}

def extract_cycle_frames(csv_path, n_points=DEFAULT_SEQUENCE_LENGTH):
    df = pd.read_csv(csv_path)
    if 'Cycle' not in df.columns:
        raise ValueError(f"extract_cycle_frames: CSV missing 'Cycle' column: {csv_path}")
    df = df.dropna(subset=['Cycle'])
    cycles = sorted(df['Cycle'].unique())
    out = []
    for c in cycles:
        sub = df[df['Cycle'] == c]
        dis = sub[sub['Current'] < -1.0]
        if len(dis) < 10:
            out.append(None)
            continue
        rs = resample_cycle_inputs(dis['Time'].values, dis['Voltage'].values, dis['Current'].values, n_points)
        out.append(rs)
    return out
_WORKER_PARAMS = None

def _worker_init(params_dict):
    global _WORKER_PARAMS
    _WORKER_PARAMS = params_dict

def _sim_worker(args):
    idx, cyc, ambient_temperature_k = args
    if cyc is None:
        return (idx, None, None)
    res = simulate_single_cycle(_WORKER_PARAMS, cyc['t'], cyc['i'], ambient_temperature_K=ambient_temperature_k)
    return (idx, res['voltage'], res['temperature'])

def build_dt_reference_cache(csv_path, params_json, out_npz, n_workers=None, n_points=DEFAULT_SEQUENCE_LENGTH, ambient_temperature_k=298.15):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pathlib.Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    with open(params_json) as f:
        params = json.load(f)
    cycles = extract_cycle_frames(csv_path, n_points=n_points)
    n = len(cycles)
    dt_v = np.full((n, n_points), np.nan, dtype=np.float32)
    dt_t = np.full((n, n_points), np.nan, dtype=np.float32)
    tasks = [(i, c, ambient_temperature_k) for i, c in enumerate(cycles)]
    n_workers = n_workers or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init, initargs=(params,)) as pool:
        futures = [pool.submit(_sim_worker, t) for t in tasks]
        for fut in as_completed(futures):
            idx, v, T = fut.result()
            if v is not None and len(v) == n_points:
                dt_v[idx] = v
            if T is not None and len(T) == n_points:
                dt_t[idx] = T
    real_v = np.stack([c['v'] if c else np.full(n_points, np.nan, np.float32) for c in cycles])
    real_t = np.stack([c['t'] if c else np.full(n_points, np.nan, np.float32) for c in cycles])
    real_i = np.stack([c['i'] if c else np.full(n_points, np.nan, np.float32) for c in cycles])
    np.savez_compressed(out_npz, dt_v=dt_v, dt_T=dt_t, real_v=real_v, real_t=real_t, real_i=real_i)
    return out_npz

def load_dt_cache(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}
