from __future__ import annotations

import numpy as np
import pytest

from saferai_budget_recovery.forward_model import oc3_dos_p_success, oc3_dos_tactic_probabilities


def test_or_gate_values_are_used() -> None:
    tactics = oc3_dos_tactic_probabilities(
        p_active=0.2,
        p_gather=0.3,
        p_acquire=1.0,
        p_build=1.0,
        p_masquerading=1.0,
        p_port=1.0,
        p_c2=1.0,
        p_direct=1.0,
        p_reflection=1.0,
    )
    assert tactics["p_rec"] == pytest.approx(0.44)


def test_and_gate_values_are_used() -> None:
    tactics = oc3_dos_tactic_probabilities(
        p_active=1.0,
        p_gather=1.0,
        p_acquire=1.0,
        p_build=1.0,
        p_masquerading=0.4,
        p_port=0.5,
        p_c2=1.0,
        p_direct=1.0,
        p_reflection=1.0,
    )
    assert tactics["p_def"] == pytest.approx(0.2)


def test_all_zero_inputs_produce_zero_success() -> None:
    assert oc3_dos_p_success(0, 0, 0, 0, 0, 0, 0, 0, 0) == pytest.approx(0.0)


def test_all_one_inputs_produce_one_success() -> None:
    assert oc3_dos_p_success(1, 1, 1, 1, 1, 1, 1, 1, 1) == pytest.approx(1.0)


def test_vectorized_numpy_inputs_work() -> None:
    arr = np.array([0.0, 1.0])
    result = oc3_dos_p_success(arr, arr, arr, arr, arr, arr, arr, arr, arr)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.array([0.0, 1.0]))


def test_out_of_range_inputs_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="p_active must lie in \\[0, 1\\]"):
        oc3_dos_p_success(1.2, 1, 1, 1, 1, 1, 1, 1, 1)

