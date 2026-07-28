import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pybamm

from batterytwin.digital_twin.dfn_model import build_dfn_model

from .optimization import PARAM_ORDER, run_parameter_search, simulate_once


def load_discharge_cycle(csv_path: str, cycle: int = 1):
    frame = pd.read_csv(csv_path)
    frame.columns = [column.strip() for column in frame.columns]
    cycle_frame = frame[frame["Cycle"] == cycle].copy()
    if cycle_frame.empty:
        raise ValueError(f"Cycle {cycle} was not found")
    discharge = cycle_frame[cycle_frame["Current"] < 0].copy()
    if discharge.empty:
        raise ValueError(f"Cycle {cycle} has no discharge rows")
    time_s = discharge["Time"].to_numpy(dtype=float)
    time_s = time_s - time_s[0]
    voltage_v = discharge["Voltage"].to_numpy(dtype=float)
    current_a = discharge["Current"].to_numpy(dtype=float)
    temperature_k = discharge["Temperature"].to_numpy(dtype=float) + 273.15
    capacity_ah = np.abs(discharge["Capacity"].to_numpy(dtype=float))
    capacity_ah = capacity_ah - capacity_ah[0]
    return time_s, voltage_v, temperature_k, current_a, capacity_ah


def calibrate_cell(
    csv_path: str,
    cell_id: str,
    cycle: int = 1,
    temperature_weight: float = 0.2,
    output_dir: str | Path = "calibration_output",
    optimizer_config: dict | None = None,
    verbose: bool = True,
) -> dict:
    time_s, voltage_v, temperature_k, current_a, _ = load_discharge_cycle(
        csv_path, cycle=cycle
    )
    initial_temperature_k = float(temperature_k[0])
    model, parameter_values = build_dfn_model(initial_temperature_k)
    parameter_values["Nominal cell capacity [A.h]"] = 4.0
    discharge_current_a = -current_a
    current_function = pybamm.Interpolant(time_s, discharge_current_a, pybamm.t)
    parameter_values["Current function [A]"] = current_function
    result = run_parameter_search(
        model=model,
        base_params=parameter_values,
        t_eval=time_s,
        current_fun=current_function,
        v_real=voltage_v,
        T_real=temperature_k,
        beta_T=temperature_weight,
        de_config=optimizer_config,
        verbose=verbose,
        current_data=np.column_stack([time_s, discharge_current_a]),
        ambient_K=initial_temperature_k,
    )
    simulated_voltage, simulated_temperature = simulate_once(
        result["x_best"], model, parameter_values, time_s, current_function
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"calibration_{cell_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    parameter_record = {
        "cell_id": cell_id,
        "fit_cycle": cycle,
        "ambient_temperature_k": initial_temperature_k,
        "parameter_order": PARAM_ORDER,
        "x_best": result["x_best"],
        "x_best_dict": result["x_best_dict"],
    }
    parameter_path = run_dir / "parameters.json"
    parameter_path.write_text(json.dumps(parameter_record, indent=2), encoding="utf-8")
    metrics = {
        "timestamp": timestamp,
        "initial_temperature_k": initial_temperature_k,
        "objective": result["best_objective"],
        "voltage_mae_v": (
            float(np.mean(np.abs(simulated_voltage - voltage_v)))
            if simulated_voltage is not None
            else None
        ),
        "temperature_mae_k": (
            float(np.mean(np.abs(simulated_temperature - temperature_k)))
            if simulated_temperature is not None
            else None
        ),
        "elapsed_s": result["elapsed_s"],
        "temperature_weight": temperature_weight,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        **parameter_record,
        "metrics": metrics,
        "run_dir": str(run_dir),
        "parameters_path": str(parameter_path),
        "metrics_path": str(metrics_path),
    }
