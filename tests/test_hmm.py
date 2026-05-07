import numpy as np
import pytest

from strategies.hmm import GaussianHMM
from strategies.regime import Regime, RegimeClassifier, log_returns_from_close


def _synthetic_returns(seed: int = 0, n_per_state: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Generate returns drawn from three well-separated Gaussian regimes.

    Returns (returns, true_states_in_emission_order). State order is bear, chop, bull
    so the classifier (which sorts by emission mean) should align with it.
    """
    rng = np.random.default_rng(seed)
    means = [-0.004, 0.0, 0.004]
    stds = [0.010, 0.004, 0.008]

    segments = []
    truth = []
    for _ in range(3):
        for k in range(3):
            seg = rng.normal(means[k], stds[k], size=n_per_state)
            segments.append(seg)
            truth.append(np.full(n_per_state, k, dtype=int))
    return np.concatenate(segments), np.concatenate(truth)


def test_log_returns_from_close_basic():
    closes = np.array([100.0, 101.0, 99.0, 100.0])
    r = log_returns_from_close(closes)
    assert r.shape == (3,)
    assert np.allclose(r, np.log(closes[1:] / closes[:-1]))


def test_log_returns_short_input_returns_empty():
    assert log_returns_from_close(np.array([100.0])).size == 0
    assert log_returns_from_close(np.array([])).size == 0


def test_hmm_recovers_well_separated_regimes():
    returns, truth = _synthetic_returns(seed=1)
    hmm = GaussianHMM(n_states=3, n_iter=200, random_state=0).fit(returns)
    states = hmm.predict(returns)

    # State 0 = lowest mean = bear, etc. Allow for a few mislabelled bars.
    accuracy = float((states == truth).mean())
    assert accuracy > 0.75, f"HMM Viterbi accuracy too low: {accuracy:.3f}"

    # Means should be sorted ascending (bear < chop < bull).
    assert hmm.params is not None
    assert hmm.params.means[0] < hmm.params.means[1] < hmm.params.means[2]


def test_hmm_predict_proba_sums_to_one():
    returns, _ = _synthetic_returns(seed=2)
    hmm = GaussianHMM(n_states=3, n_iter=50, random_state=0).fit(returns)
    proba = hmm.predict_proba(returns)
    assert proba.shape == (returns.size, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1 + 1e-9).all()


def test_hmm_score_returns_finite():
    returns, _ = _synthetic_returns(seed=3)
    hmm = GaussianHMM(n_states=3, n_iter=50, random_state=0).fit(returns)
    ll = hmm.score(returns)
    assert np.isfinite(ll)


def test_hmm_rejects_too_few_samples():
    hmm = GaussianHMM(n_states=3)
    with pytest.raises(ValueError):
        hmm.fit(np.array([0.01, -0.01, 0.0]))


def test_regime_classifier_snapshot_fields():
    returns, _ = _synthetic_returns(seed=4)
    rc = RegimeClassifier(n_iter=50).fit(returns)
    snap = rc.snapshot(returns)

    assert isinstance(snap.regime, Regime)
    assert snap.proba.shape == (3,)
    assert np.isclose(snap.proba.sum(), 1.0, atol=1e-6)
    assert np.isfinite(snap.expected_return)
    assert snap.expected_vol >= 0


def test_regime_classifier_state_means_sorted():
    returns, _ = _synthetic_returns(seed=5)
    rc = RegimeClassifier(n_iter=50).fit(returns)
    means = rc.state_means
    assert means[0] <= means[1] <= means[2]
    # Transition matrix rows must sum to 1.
    A = rc.transition_matrix
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)


def test_regime_classifier_label_mapping():
    assert RegimeClassifier.label_for_state(0) == "bear"
    assert RegimeClassifier.label_for_state(1) == "chop"
    assert RegimeClassifier.label_for_state(2) == "bull"


# --------------------------------------------------------------------------- #
#  Edge cases                                                                  #
# --------------------------------------------------------------------------- #


def test_log_returns_two_close_inputs():
    """Two closes produce exactly one log return."""
    r = log_returns_from_close(np.array([100.0, 105.0]))
    assert r.shape == (1,)
    assert r[0] == pytest.approx(np.log(1.05))


def test_log_returns_negative_close_returns_nan():
    """Log of a negative close should produce a NaN — the function does not assert."""
    with np.errstate(invalid="ignore"):
        r = log_returns_from_close(np.array([100.0, -50.0]))
    assert np.isnan(r[0])


def test_hmm_score_increases_or_stays_after_more_iterations():
    """EM training should monotonically improve the log-likelihood (loose bound)."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=400)

    fast = GaussianHMM(n_states=3, n_iter=5, random_state=0).fit(returns)
    slow = GaussianHMM(n_states=3, n_iter=80, random_state=0).fit(returns)

    # More EM iterations should not produce a worse log-likelihood (within tolerance).
    assert slow.score(returns) + 1e-3 >= fast.score(returns)


def test_hmm_sort_invariant_after_multiple_fits():
    """Means must remain sorted ascending after each independent fit."""
    rng = np.random.default_rng(9)
    returns_a = rng.normal(0.0, 0.005, size=300)
    returns_b = rng.normal(0.001, 0.005, size=300)

    hmm = GaussianHMM(n_states=3, n_iter=50, random_state=0)
    hmm.fit(returns_a)
    assert hmm.params is not None
    assert hmm.params.means[0] <= hmm.params.means[1] <= hmm.params.means[2]

    hmm.fit(returns_b)
    assert hmm.params is not None
    assert hmm.params.means[0] <= hmm.params.means[1] <= hmm.params.means[2]


def test_hmm_predict_proba_on_single_observation_is_normalised():
    """After fitting, predict_proba on a length-1 input must sum to 1."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.005, size=200)
    hmm = GaussianHMM(n_states=3, n_iter=50, random_state=0).fit(returns)

    proba = hmm.predict_proba(np.array([0.001]))
    assert proba.shape == (1, 3)
    assert proba[0].sum() == pytest.approx(1.0, abs=1e-6)


def test_predict_proba_without_fit_raises():
    hmm = GaussianHMM(n_states=3)
    with pytest.raises(RuntimeError):
        hmm.predict_proba(np.array([0.001, -0.001, 0.0005]))


def test_state_means_property_without_fit_raises():
    rc = RegimeClassifier()
    with pytest.raises(RuntimeError):
        _ = rc.state_means
    with pytest.raises(RuntimeError):
        _ = rc.state_variances
    with pytest.raises(RuntimeError):
        _ = rc.transition_matrix


def test_hmm_predict_returns_only_valid_state_indices():
    """Viterbi output must only contain values in [0, n_states)."""
    rng = np.random.default_rng(13)
    returns = rng.normal(0.0, 0.005, size=200)
    hmm = GaussianHMM(n_states=3, n_iter=30, random_state=0).fit(returns)

    states = hmm.predict(returns)
    assert states.shape == (returns.size,)
    assert states.min() >= 0
    assert states.max() <= 2
    assert states.dtype.kind in ("i", "u")
