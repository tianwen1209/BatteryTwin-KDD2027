import json
import pathlib
import numpy as np
__all__ = ['d1_monotonicity', 'd2_derivative_envelope', 'd3_pybamm_residual', 'd4_structural', 'd5_temperature', 'detect_cycle', 'detect_cycle_batch', 'calibrate_env', 'save_env', 'load_env']

def d1_monotonicity(v, eps=0.002):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f'd1_monotonicity expects 1-D array, got shape {v.shape}')
    L = len(v)
    out = np.zeros(L, dtype=bool)
    d = np.diff(v)
    bad = d > eps
    out[:-1] |= bad
    out[1:] |= bad
    return out

def d2_derivative_envelope(v, dv_mean, dv_std, k=3.0):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f'd2_derivative_envelope expects 1-D array, got shape {v.shape}')
    dv_mean = float(dv_mean)
    dv_std = float(dv_std)
    d = np.diff(v)
    z = np.abs(d - dv_mean) / max(dv_std, 1e-08)
    bad = z > k
    out = np.zeros(len(v), dtype=bool)
    out[:-1] |= bad
    out[1:] |= bad
    return out

def d3_pybamm_residual(v, dt_ref, z_threshold=3.0, window=5):
    v = np.asarray(v, dtype=np.float32)
    r = np.asarray(dt_ref, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f'd3_pybamm_residual expects 1-D v, got shape {v.shape}')
    if r.shape != v.shape:
        raise ValueError(f'dt_ref shape {r.shape} != v shape {v.shape}')
    if not np.isfinite(r).all():
        return np.zeros_like(v, dtype=bool)
    diff = np.abs(v - r)
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) + 1e-08
    z = (diff - med) / mad
    return z > z_threshold

def d4_structural(v, cutoff_idx_ref, knee_idx_ref, cutoff_v_ref=2.5, knee_tol_idx=15, cutoff_tol_v=0.05, min_knee_d2=0.0001):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f'd4_structural expects 1-D array, got shape {v.shape}')
    L = len(v)
    out = np.zeros(L, dtype=bool)
    if abs(v[-1] - cutoff_v_ref) > cutoff_tol_v:
        out[-min(20, L):] = True
    d2 = np.abs(np.diff(np.diff(v)))
    if d2.size > 0 and d2.max() > min_knee_d2:
        knee_idx = int(np.argmax(d2)) + 1
        if abs(knee_idx - knee_idx_ref) > knee_tol_idx:
            s = max(0, knee_idx - 5)
            e = min(L, knee_idx + 5)
            out[s:e] = True
    return out

def d5_temperature(T, direction='warming', max_drop=0.3, window=5):
    T = np.asarray(T, dtype=np.float32)
    if T.ndim != 1:
        raise ValueError(f'd5_temperature expects 1-D array, got shape {T.shape}')
    L = len(T)
    out = np.zeros(L, dtype=bool)
    if direction != 'warming':
        return out
    for k in range(window, L):
        if T[k] < T[k - window] - max_drop:
            out[k - window:k + 1] = True
    return out

def detect_cycle(v, dt_ref, T, env):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 1:
        raise ValueError(f'detect_cycle: v must be 1-D, got shape {v.shape}')
    m1 = d1_monotonicity(v)
    m2 = d2_derivative_envelope(v, env['dv_mean'], env['dv_std'], env['k'])
    m3 = d3_pybamm_residual(v, dt_ref, env['z_threshold'])
    m4 = d4_structural(v, env['cutoff_idx_ref'], env['knee_idx_ref'])
    if T is not None:
        m5 = d5_temperature(np.asarray(T, dtype=np.float32))
    else:
        m5 = np.zeros(len(v), dtype=bool)
    mask = m1 | m2 | m3 | m4 | m5
    sev = (m1 | m2).astype(np.int8) * 1
    sev = np.maximum(sev, m3.astype(np.int8) * 2)
    sev = np.maximum(sev, m4.astype(np.int8) * 2)
    sev = np.maximum(sev, (m3 & m4 | m5).astype(np.int8) * 3)
    return (mask, sev)

def detect_cycle_batch(V, DT_refs, T_arr, env):
    V = np.asarray(V, dtype=np.float32)
    DT_refs = np.asarray(DT_refs, dtype=np.float32)
    if V.ndim != 2:
        raise ValueError(f'detect_cycle_batch: V must be 2-D [N, L], got {V.shape}')
    N, L = V.shape
    masks = np.zeros((N, L), dtype=bool)
    sevs = np.zeros((N, L), dtype=np.int8)
    for i in range(N):
        T_i = T_arr[i] if T_arr is not None else None
        masks[i], sevs[i] = detect_cycle(V[i], DT_refs[i], T_i, env)
    return (masks, sevs)

def calibrate_env(V_clean):
    V = np.asarray(V_clean, dtype=np.float32)
    if V.ndim != 2:
        raise ValueError(f'calibrate_env expects 2-D array [N, L], got shape {V.shape}')
    if V.shape[0] < 1:
        raise ValueError('calibrate_env requires at least 1 clean cycle')
    dV = np.diff(V, axis=1)
    L = V.shape[1]
    offset = L // 4
    d2 = np.abs(np.diff(dV, axis=1))
    knee_idx = int(np.median(np.argmax(d2[:, offset:], axis=1))) + offset + 1
    return {'dv_mean': float(dV.mean()), 'dv_std': float(dV.std() + 1e-05), 'cutoff_idx_ref': int(V.shape[1] - 1), 'knee_idx_ref': knee_idx, 'z_threshold': 3.0, 'k': 3.0}

def save_env(env, path):
    import os
    import tempfile
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(env, f, indent=2)
    os.replace(tmp, path)

def load_env(path):
    with open(path) as f:
        return json.load(f)
