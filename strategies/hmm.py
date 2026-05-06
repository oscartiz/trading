"""Gaussian Hidden Markov Model — numpy-only.

Used by the regime-switching strategy to classify each return bar into one of
N latent states. Implementation: forward-backward (in log space for stability),
Viterbi for MAP path, and Baum-Welch (EM) for parameter estimation.

States are returned sorted by emission mean ascending, so for n_states=3 the
caller can rely on state 0 = lowest-mean (bear), state 1 = middle (chop),
state 2 = highest-mean (bull).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_2PI = float(np.log(2 * np.pi))
_EPS = 1e-12


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + _EPS) + a_max
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


@dataclass
class HMMParams:
    start_prob: np.ndarray   # (K,)
    trans_mat: np.ndarray    # (K, K) row-stochastic
    means: np.ndarray        # (K,)
    variances: np.ndarray    # (K,)


class GaussianHMM:
    """Univariate Gaussian-emission HMM trained with Baum-Welch."""

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-4,
        min_variance: float = 1e-8,
        random_state: int = 0,
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be ≥ 2")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_variance = min_variance
        self.random_state = random_state
        self.params: HMMParams | None = None
        self.log_likelihood_: float | None = None
        self.n_iter_run_: int = 0

    # ---------- public API ----------
    def fit(self, x: np.ndarray) -> "GaussianHMM":
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size < self.n_states * 5:
            raise ValueError(f"Need at least {self.n_states * 5} samples to fit {self.n_states}-state HMM")

        params = self._init_params(x)
        prev_ll = -np.inf

        for it in range(self.n_iter):
            log_emit = self._log_emissions(x, params)
            log_alpha, ll = self._forward(log_emit, params)
            log_beta = self._backward(log_emit, params)
            params = self._m_step(x, log_emit, log_alpha, log_beta, params)
            self.n_iter_run_ = it + 1
            if np.isfinite(ll) and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        params = self._sort_by_mean(params)
        self.params = params
        self.log_likelihood_ = float(prev_ll)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Smoothed posteriors P(state_t | x_{1..T}) — shape (T, K)."""
        params = self._require_fitted()
        x = np.asarray(x, dtype=np.float64).ravel()
        log_emit = self._log_emissions(x, params)
        log_alpha, _ = self._forward(log_emit, params)
        log_beta = self._backward(log_emit, params)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        return np.exp(log_gamma)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Viterbi most-likely state sequence — shape (T,)."""
        params = self._require_fitted()
        x = np.asarray(x, dtype=np.float64).ravel()
        log_emit = self._log_emissions(x, params)
        T, K = log_emit.shape
        log_pi = np.log(params.start_prob + _EPS)
        log_A = np.log(params.trans_mat + _EPS)

        delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=np.int64)
        delta[0] = log_pi + log_emit[0]
        for t in range(1, T):
            scores = delta[t - 1, :, None] + log_A
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emit[t]

        states = np.empty(T, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def score(self, x: np.ndarray) -> float:
        """Log-likelihood of the observation sequence."""
        params = self._require_fitted()
        x = np.asarray(x, dtype=np.float64).ravel()
        log_emit = self._log_emissions(x, params)
        _, ll = self._forward(log_emit, params)
        return float(ll)

    # ---------- internals ----------
    def _require_fitted(self) -> HMMParams:
        if self.params is None:
            raise RuntimeError("HMM is not fitted yet — call fit() first.")
        return self.params

    def _init_params(self, x: np.ndarray) -> HMMParams:
        """Quantile-based initialisation: split sorted x into K equal chunks."""
        rng = np.random.default_rng(self.random_state)
        K = self.n_states
        sorted_x = np.sort(x)
        chunks = np.array_split(sorted_x, K)
        means = np.array([float(c.mean()) for c in chunks])
        variances = np.array([max(float(c.var()), self.min_variance) for c in chunks])
        # Tiny noise so the symmetry breaks if all chunks happen to be identical.
        means += rng.normal(0, 1e-6, size=K)
        # Persistent transitions encourage regime stickiness.
        trans = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(trans, 0.95)
        start = np.full(K, 1.0 / K)
        return HMMParams(start_prob=start, trans_mat=trans, means=means, variances=variances)

    def _log_emissions(self, x: np.ndarray, p: HMMParams) -> np.ndarray:
        """Log Gaussian PDF for each (t, k) — shape (T, K)."""
        T = x.size
        K = self.n_states
        var = np.maximum(p.variances, self.min_variance)
        diff = x[:, None] - p.means[None, :]
        log_emit = -0.5 * (_LOG_2PI + np.log(var)[None, :] + diff**2 / var[None, :])
        # Clip wildly negative values that occur for outliers vs. tight states.
        return np.clip(log_emit, -1e6, 1e6).reshape(T, K)

    def _forward(self, log_emit: np.ndarray, p: HMMParams) -> tuple[np.ndarray, float]:
        T, K = log_emit.shape
        log_pi = np.log(p.start_prob + _EPS)
        log_A = np.log(p.trans_mat + _EPS)
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_emit[0]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t - 1, :, None] + log_A, axis=0) + log_emit[t]
        ll = float(_logsumexp(log_alpha[-1]))
        return log_alpha, ll

    def _backward(self, log_emit: np.ndarray, p: HMMParams) -> np.ndarray:
        T, K = log_emit.shape
        log_A = np.log(p.trans_mat + _EPS)
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_A + (log_emit[t + 1] + log_beta[t + 1])[None, :], axis=1
            )
        return log_beta

    def _m_step(
        self,
        x: np.ndarray,
        log_emit: np.ndarray,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        p: HMMParams,
    ) -> HMMParams:
        T = log_emit.shape[0]
        # γ_t(i) = P(state_t = i | x)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # ξ_t(i,j) = P(state_t=i, state_{t+1}=j | x)
        log_A = np.log(p.trans_mat + _EPS)
        log_xi = (
            log_alpha[:-1, :, None]
            + log_A[None, :, :]
            + (log_emit[1:] + log_beta[1:])[:, None, :]
        )
        log_xi -= _logsumexp(log_xi.reshape(T - 1, -1), axis=1)[:, None, None]
        xi = np.exp(log_xi)

        start_prob = gamma[0] / (gamma[0].sum() + _EPS)
        trans_mat = xi.sum(axis=0) / (gamma[:-1].sum(axis=0)[:, None] + _EPS)
        # Renormalise rows defensively.
        trans_mat = trans_mat / (trans_mat.sum(axis=1, keepdims=True) + _EPS)

        gamma_sum = gamma.sum(axis=0) + _EPS
        means = (gamma * x[:, None]).sum(axis=0) / gamma_sum
        diffs = x[:, None] - means[None, :]
        variances = (gamma * diffs**2).sum(axis=0) / gamma_sum
        variances = np.maximum(variances, self.min_variance)

        return HMMParams(
            start_prob=start_prob,
            trans_mat=trans_mat,
            means=means,
            variances=variances,
        )

    def _sort_by_mean(self, p: HMMParams) -> HMMParams:
        order = np.argsort(p.means)
        return HMMParams(
            start_prob=p.start_prob[order],
            trans_mat=p.trans_mat[np.ix_(order, order)],
            means=p.means[order],
            variances=p.variances[order],
        )
