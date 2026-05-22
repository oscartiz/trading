"""3-state regime classifier built on top of the Gaussian HMM.

Conventions (after sorting states by emission mean ascending):
    state 0 → BEAR (lowest-mean log returns)
    state 1 → CHOP (middle)
    state 2 → BULL (highest-mean log returns)

Typical use:
    >>> rc = RegimeClassifier()
    >>> rc.fit(returns_array)
    >>> proba = rc.predict_proba(returns_array)   # shape (T, 3)
    >>> rc.label_for_state(2)                      # "bull"
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .hmm import GaussianHMM


class Regime(IntEnum):
    BEAR = 0
    CHOP = 1
    BULL = 2

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass
class RegimeSnapshot:
    regime: Regime
    proba: np.ndarray            # shape (3,) — P(bear), P(chop), P(bull)
    expected_return: float       # E[r] under the posterior, in log-return units
    expected_vol: float          # sqrt(E[var]) under the posterior

    @property
    def label(self) -> str:
        return self.regime.label


class RegimeClassifier:
    """Fits a 3-state Gaussian HMM on log returns and exposes posteriors."""

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-4,
        min_variance: float = 1e-10,
        random_state: int = 0,
        n_seeds: int = 1,
    ) -> None:
        self.hmm = GaussianHMM(
            n_states=3,
            n_iter=n_iter,
            tol=tol,
            min_variance=min_variance,
            random_state=random_state,
            n_seeds=n_seeds,
        )

    # ---------- training ----------
    def fit(self, returns: np.ndarray) -> "RegimeClassifier":
        self.hmm.fit(np.asarray(returns, dtype=np.float64).ravel())
        return self

    # ---------- inference ----------
    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        return self.hmm.predict_proba(np.asarray(returns, dtype=np.float64).ravel())

    def predict(self, returns: np.ndarray) -> np.ndarray:
        return self.hmm.predict(np.asarray(returns, dtype=np.float64).ravel())

    def snapshot(self, returns: np.ndarray) -> RegimeSnapshot:
        """Run smoothing on the full series and report the latest bar's regime."""
        proba = self.predict_proba(returns)[-1]
        params = self.hmm.params
        assert params is not None  # fit() ensures this
        means = params.means
        variances = params.variances
        regime = Regime(int(np.argmax(proba)))
        expected_return = float(np.dot(proba, means))
        expected_vol = float(np.sqrt(max(np.dot(proba, variances), 0.0)))
        return RegimeSnapshot(
            regime=regime,
            proba=proba,
            expected_return=expected_return,
            expected_vol=expected_vol,
        )

    # ---------- helpers ----------
    @staticmethod
    def label_for_state(state: int) -> str:
        return Regime(state).label

    @property
    def state_means(self) -> np.ndarray:
        params = self.hmm.params
        if params is None:
            raise RuntimeError("Classifier is not fitted yet.")
        return params.means

    @property
    def state_variances(self) -> np.ndarray:
        params = self.hmm.params
        if params is None:
            raise RuntimeError("Classifier is not fitted yet.")
        return params.variances

    @property
    def transition_matrix(self) -> np.ndarray:
        params = self.hmm.params
        if params is None:
            raise RuntimeError("Classifier is not fitted yet.")
        return params.trans_mat

    @property
    def log_likelihood(self) -> float | None:
        """Log-likelihood of the most recent fit (None before fit())."""
        return self.hmm.log_likelihood_

    @property
    def n_iter_run(self) -> int:
        """EM iterations the most recent fit needed before convergence/cap."""
        return self.hmm.n_iter_run_


def log_returns_from_close(close: np.ndarray) -> np.ndarray:
    """log(p_t / p_{t-1}) — the canonical input for the regime model."""
    close = np.asarray(close, dtype=np.float64).ravel()
    if close.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.diff(np.log(close))
