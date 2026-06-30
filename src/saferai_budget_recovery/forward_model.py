"""OC3 Denial of Service P(success) forward model."""

from __future__ import annotations

from typing import Any

import numpy as np


STEP_LABEL_BY_VARIABLE = {
    "p_active": "T1595 - Reconnaissance: Active Scanning",
    "p_gather": "T1590 - Reconnaissance: Gather Victim Network Information",
    "p_acquire": "T1583.005 - Resource Development: Acquire Botnet",
    "p_build": "T1584.005 - Resource Development: Build/Compromise Botnet",
    "p_masquerading": "T1036 - Defense Evasion: Masquerading",
    "p_port": "T1571 - Defense Evasion: Non-Standard Port",
    "p_c2": "TA0011 - Command-and-Control",
    "p_direct": "T1498.001 - Impact: Direct Network Flood",
    "p_reflection": "T1498.002 - Impact: Reflection/Amplification Attack",
}

VARIABLE_BY_STEP_LABEL = {label: variable for variable, label in STEP_LABEL_BY_VARIABLE.items()}


def oc3_dos_tactic_probabilities(
    p_active: Any,
    p_gather: Any,
    p_acquire: Any,
    p_build: Any,
    p_masquerading: Any,
    p_port: Any,
    p_c2: Any,
    p_direct: Any,
    p_reflection: Any,
) -> dict[str, Any]:
    """Return OC3 DoS tactic-level probabilities for scalar or array inputs."""

    _validate_probabilities(
        p_active=p_active,
        p_gather=p_gather,
        p_acquire=p_acquire,
        p_build=p_build,
        p_masquerading=p_masquerading,
        p_port=p_port,
        p_c2=p_c2,
        p_direct=p_direct,
        p_reflection=p_reflection,
    )
    p_rec = p_active + p_gather - p_active * p_gather
    p_res = p_build + p_acquire - p_build * p_acquire
    p_def = p_port * p_masquerading
    p_imp = p_direct + p_reflection - p_direct * p_reflection
    return {
        "p_rec": p_rec,
        "p_res": p_res,
        "p_def": p_def,
        "p_c2_tactic": p_c2,
        "p_imp": p_imp,
    }


def oc3_dos_p_success(
    p_active: Any,
    p_gather: Any,
    p_acquire: Any,
    p_build: Any,
    p_masquerading: Any,
    p_port: Any,
    p_c2: Any,
    p_direct: Any,
    p_reflection: Any,
) -> Any:
    """Compute OC3 DoS P(success) from the nine elicited leaf probabilities."""

    tactics = oc3_dos_tactic_probabilities(
        p_active=p_active,
        p_gather=p_gather,
        p_acquire=p_acquire,
        p_build=p_build,
        p_masquerading=p_masquerading,
        p_port=p_port,
        p_c2=p_c2,
        p_direct=p_direct,
        p_reflection=p_reflection,
    )
    p_success = (
        tactics["p_rec"]
        * tactics["p_res"]
        * tactics["p_def"]
        * tactics["p_c2_tactic"]
        * tactics["p_imp"]
    )
    _validate_output_range(p_success)
    return p_success


def _validate_probabilities(**values: Any) -> None:
    for name, value in values.items():
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite probability values.")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1].")


def _validate_output_range(value: Any) -> None:
    arr = np.asarray(value, dtype=float)
    tolerance = 1e-12
    if np.any((arr < -tolerance) | (arr > 1.0 + tolerance)):
        raise AssertionError("OC3 DoS P(success) produced a value outside [0, 1].")

