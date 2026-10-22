"""Tests for crisp_nam.metrics discrimination metrics.

With a single event type the competing-risks metrics reduce to their
standard survival counterparts, so they are checked against scikit-survival.
"""

import numpy as np
import pytest
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from crisp_nam.metrics import auc_td, truncated_concordance_td


def _single_risk_split(seed):
    rng = np.random.default_rng(seed)
    n = 1000
    x = rng.normal(size=n)
    event_times = rng.exponential(np.exp(-x))
    censor_times = rng.exponential(1.5, size=n)
    times = np.minimum(event_times, censor_times)
    events = (event_times <= censor_times).astype(int)
    train, test = slice(0, n // 2), slice(n // 2, n)
    return x[test], times[train], events[train], times[test], events[test]


@pytest.mark.parametrize("seed", [0, 1])
def test_auc_td_matches_sksurv_without_competing_risks(seed):
    scores, t_train, e_train, t_test, e_test = _single_risk_split(seed)
    tau = np.quantile(t_test, 0.5)

    auc, _ = auc_td(
        e_test, t_test, scores.reshape(-1, 1), np.array([tau]), tau, km=(e_train, t_train)
    )
    expected, _ = cumulative_dynamic_auc(
        Surv.from_arrays(e_train == 1, t_train),
        Surv.from_arrays(e_test == 1, t_test),
        scores,
        [tau],
    )
    assert auc == pytest.approx(expected[0], abs=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_truncated_concordance_td_matches_sksurv_without_competing_risks(seed):
    scores, t_train, e_train, t_test, e_test = _single_risk_split(seed)
    tau = np.quantile(t_test, 0.5)

    c_index, _ = truncated_concordance_td(
        e_test, t_test, scores.reshape(-1, 1), np.array([tau]), tau, km=(e_train, t_train)
    )
    expected = concordance_index_ipcw(
        Surv.from_arrays(e_train == 1, t_train),
        Surv.from_arrays(e_test == 1, t_test),
        scores,
        tau=tau,
    )[0]
    assert c_index == pytest.approx(expected, abs=1e-6)


def test_truncated_concordance_td_counts_competing_events_as_controls():
    # Subject 1 had a competing event before subject 0's cause-1 event, so the
    # pair is comparable and concordant when subject 0 has the higher risk.
    e_test = np.array([1, 2])
    t_test = np.array([2.0, 1.0])
    risks = np.array([[0.9], [0.1]])
    c_index, _ = truncated_concordance_td(e_test, t_test, risks, np.array([3.0]), 3.0)
    assert c_index == 1.0


def test_auc_td_handles_a_single_case():
    # lifelines returns a scalar rather than an array for one time point
    e_test = np.array([2, 0, 0, 1, 0])
    t_test = np.array([1.0, 2.0, 5.0, 6.0, 7.0])
    risks = np.array([[0.9], [0.1], [0.2], [0.3], [0.4]])
    auc, _ = auc_td(
        e_test, t_test, risks, np.array([3.0]), 3.0, km=(e_test, t_test), primary_risk=2
    )
    assert auc == 1.0
