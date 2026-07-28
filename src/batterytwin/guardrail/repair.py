import warnings
import numpy as np

def repair_tier1_interp(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool) | np.isnan(v)
    if not mask.any():
        return v
    good = ~mask
    if not good.any():
        return v
    idx = np.arange(len(v))
    v[mask] = np.interp(idx[mask], idx[good], v[good])
    return v

def repair_tier2_projection(v: np.ndarray, mask: np.ndarray, dv_min: float=-0.5, dv_max: float=-0.0001, anchor_pad: int=2) -> np.ndarray:
    import cvxpy as cp
    v = np.asarray(v, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return v
    idx_bad = np.where(mask)[0]
    s = max(0, int(idx_bad.min()) - anchor_pad)
    e = min(len(v), int(idx_bad.max()) + 1 + anchor_pad)
    seg = v[s:e].astype(np.float64)
    n = len(seg)
    if n <= 1:
        return repair_tier1_interp(v, mask)
    x = cp.Variable(n)
    constraints = []
    if n >= 2:
        diffs = cp.diff(x)
        constraints.append(diffs >= dv_min)
        constraints.append(diffs <= dv_max)
    for k in range(n):
        global_idx = s + k
        if not mask[global_idx]:
            constraints.append(x[k] == float(seg[k]))
    obj = cp.Minimize(cp.sum_squares(x - seg))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, max_iter=5000)
        if prob.status not in ('optimal', 'optimal_inaccurate') or x.value is None:
            warnings.warn(f"repair_tier2: OSQP returned status '{prob.status}'; falling back to Tier 1 interpolation.", RuntimeWarning, stacklevel=2)
            return repair_tier1_interp(v, mask)
        v[s:e] = np.asarray(x.value, dtype=np.float32)
    except Exception as exc:
        warnings.warn(f'repair_tier2: CVXPY raised {exc!r}; falling back to Tier 1 interpolation.', RuntimeWarning, stacklevel=2)
        return repair_tier1_interp(v, mask)
    return v

def repair_tier3_patch(v: np.ndarray, mask: np.ndarray, dt_ref: np.ndarray, blend_pad: int=5) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return v
    if dt_ref is None or not np.isfinite(np.asarray(dt_ref)).all():
        warnings.warn('repair_tier3: dt_ref is None or non-finite; falling back to Tier 1 interpolation.', RuntimeWarning, stacklevel=2)
        return repair_tier1_interp(v, mask)
    dt_ref = np.asarray(dt_ref, dtype=np.float32)
    idx_bad = np.where(mask)[0]
    s = int(idx_bad.min())
    e = int(idx_bad.max()) + 1
    span = e - s
    left_anchor = float(v[s - 1]) if s > 0 else float(dt_ref[s])
    right_anchor = float(v[e]) if e < len(v) else float(dt_ref[e - 1])
    v[s:e] = dt_ref[s:e]
    left_blend = min(blend_pad, span)
    for j in range(left_blend):
        w = float(0.5 * (1.0 - np.cos(np.pi * (j + 1) / (blend_pad + 1))))
        v[s + j] = float((1.0 - w) * left_anchor + w * dt_ref[s + j])
    right_blend = min(blend_pad, span)
    for j in range(right_blend):
        idx = e - 1 - j
        if idx < s + left_blend:
            break
        w = float(0.5 * (1.0 - np.cos(np.pi * (j + 1) / (blend_pad + 1))))
        v[idx] = float((1.0 - w) * right_anchor + w * dt_ref[idx])
    return v
