import numpy as np
import pybamm
from pybamm.input.parameters.lithium_ion.Chen2020 import nmc_LGM50_ocp_Chen2020, graphite_LGM50_ocp_Chen2020

def graphite_entropic_change_Mohtat2020(sto):
    du_dT = 10 ** (-3) * (0.28 - 1.56 * sto - 8.92 * sto ** 2 + 57.21 * sto ** 3 - 110.7 * sto ** 4 + 90.71 * sto ** 5 - 27.14 * sto ** 6)
    return du_dT

def NMC_entropic_change_Mohtat2020(sto):
    u_eq = 4.3452 - 1.6518 * sto + 1.6225 * sto ** 2 - 2.0843 * sto ** 3 + 3.5146 * sto ** 4 - 0.5623 * 10 ** (-4) * np.exp(109.451 * sto - 100.006)
    du_dT = (-800 + 779 * u_eq - 284 * u_eq ** 2 + 46 * u_eq ** 3 - 2.8 * u_eq ** 4) * 10 ** (-3)
    return du_dT

def build_dfn_model(ambient_temperature_K: float=298.15, use_chen2020_ocp: bool=True, degradation: bool=False):
    options = {'thermal': 'lumped'}
    if degradation:
        options['SEI'] = 'ec reaction limited'
        options['lithium plating'] = 'partially reversible'
    model = pybamm.lithium_ion.DFN(options)
    params = pybamm.ParameterValues('OKane2022')
    params['Ambient temperature [K]'] = ambient_temperature_K
    params['Initial temperature [K]'] = ambient_temperature_K
    if use_chen2020_ocp:
        params['Positive electrode OCP [V]'] = nmc_LGM50_ocp_Chen2020
        params['Negative electrode OCP [V]'] = graphite_LGM50_ocp_Chen2020
    params['Negative electrode OCP entropic change [V.K-1]'] = graphite_entropic_change_Mohtat2020
    params['Positive electrode OCP entropic change [V.K-1]'] = NMC_entropic_change_Mohtat2020
    return (model, params)
_OKANE_ONLY_PARAMS_CACHE: dict[str, dict] = {}
_MOHTAT_DUDT_SETS = {'Chen2020', 'OKane2022'}

def _okane_missing_params(base_set: str='Chen2020') -> dict:
    if base_set not in _OKANE_ONLY_PARAMS_CACHE:
        base = pybamm.ParameterValues(base_set)
        okane = pybamm.ParameterValues('OKane2022')
        missing_keys = set(okane.keys()) - set(base.keys())
        _OKANE_ONLY_PARAMS_CACHE[base_set] = {k: okane[k] for k in missing_keys}
    return dict(_OKANE_ONLY_PARAMS_CACHE[base_set])

def build_dfn_model_phys(param_set: str='Chen2020', ambient_temperature_K: float=298.15, degradation: bool=True):
    options = {'thermal': 'lumped'}
    if degradation:
        options['SEI'] = 'ec reaction limited'
        options['lithium plating'] = 'partially reversible'
    model = pybamm.lithium_ion.DFN(options)
    params = pybamm.ParameterValues(param_set)
    params.update(_okane_missing_params(param_set), check_already_exists=False)
    params['Ambient temperature [K]'] = ambient_temperature_K
    params['Initial temperature [K]'] = ambient_temperature_K
    if param_set in _MOHTAT_DUDT_SETS:
        params['Negative electrode OCP entropic change [V.K-1]'] = graphite_entropic_change_Mohtat2020
        params['Positive electrode OCP entropic change [V.K-1]'] = NMC_entropic_change_Mohtat2020
    return (model, params)

def build_dfn_model_chen2020_phys(ambient_temperature_K: float=298.15, degradation: bool=True):
    return build_dfn_model_phys('Chen2020', ambient_temperature_K, degradation)
