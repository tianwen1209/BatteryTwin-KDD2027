import numpy as np

from .detector import detect_cycle
from .repair import repair_tier1_interp, repair_tier2_projection, repair_tier3_patch


def process_cycle(voltage, dt_reference, environment, temperature=None):
    voltage = np.asarray(voltage, dtype=np.float32)
    mask, severity = detect_cycle(voltage, dt_reference, temperature, environment)
    highest_severity = int(severity.max()) if severity.size else 0
    if highest_severity >= 3:
        repaired = repair_tier3_patch(voltage, mask, dt_reference)
    elif highest_severity == 2:
        repaired = repair_tier2_projection(voltage, mask)
    elif highest_severity == 1:
        repaired = repair_tier1_interp(voltage, mask)
    else:
        repaired = voltage.copy()
    return {
        "voltage": repaired,
        "mask": mask,
        "severity": severity,
        "highest_severity": highest_severity,
    }
