"""Fully Bayesian counterpart of the UC model in PyMC, diagnosed with ArviZ.

Same structure as ``uc.py`` (random-walk level with constant drift + fixed trigonometric seasonal
+ observation noise) so the two are directly comparable, but with explicit priors, a full posterior
over every parameter, and MCMC instead of point-estimated variances.

    level_s  = level0 + drift * s + sigma_level * W_s      W = standard random walk, W_0 = 0
    season_t = sum_k a_k cos(2 pi k m_t / 12) + b_k sin(2 pi k m_t / 12)    m_t = calendar month
    y_t      = level + season + Normal(0, sigma_obs)       (y = logit rate, centred)

Marginalising the latent random walk gives  y ~ MvNormal(level0 + drift*s + X beta, K),
K_ij = sigma_level^2 * min(s_i, s_j) + sigma_obs^2 * 1[i=j].  NUTS therefore samples only six
scalars/vectors. (A first version sampled the 24 latent innovations explicitly; that funnel gave
dozens of divergences and R-hat > 1.05 with only 15 observations.) Forecasts are exact
Gaussian conditionals of the future given the data, drawn once per posterior sample, so they carry
parameter uncertainty as well as state and observation noise.
"""
import numpy as np
import pandas as pd

from .. import config


def fourier_design(months: np.ndarray, harmonics: int) -> np.ndarray:
    cols = []
    for k in range(1, harmonics + 1):
        ang = 2 * np.pi * k * months / config.SEASONAL_PERIOD
        cols += [np.cos(ang), np.sin(ang)]
    return np.column_stack(cols)


def build_model(y: pd.Series, harmonics: int = config.SEASONAL_HARMONICS):
    import pymc as pm
    import pytensor.tensor as pt

    n = len(y)
    idx = pd.date_range(y.index[0], periods=n, freq="MS")
    X = fourier_design(idx.month.values, harmonics)
    s = np.arange(n, dtype=float)
    y_c = y.values - y.values.mean()
    min_s = np.minimum.outer(s, s)
    with pm.Model() as model:
        # Weakly informative priors in logit units (0.05 logit ~ 1 percentage point near a 30% rate).
        sigma_obs = pm.HalfNormal("sigma_obs", 0.05)
        sigma_level = pm.HalfNormal("sigma_level", 0.05)
        drift = pm.Normal("drift", 0.0, 0.03)  # per-month logit drift
        level0 = pm.Normal("level0", 0.0, 0.5)
        beta = pm.Normal("beta", 0.0, 0.15, shape=X.shape[1])  # seasonal Fourier coefficients
        mean = level0 + drift * s + pt.dot(X, beta)
        cov = sigma_level**2 * min_s + sigma_obs**2 * np.eye(n)
        pm.MvNormal("y", mu=mean, cov=cov, observed=y_c)
    return model, y.values.mean()


def _sample_future(post: dict, y_c: np.ndarray, n: int, horizon: int, harmonics: int,
                   months_full: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One forecast path of y_future per posterior draw -> array (horizon, n_draws)."""
    X = fourier_design(months_full, harmonics)
    s = np.arange(n + horizon, dtype=float)
    min_s = np.minimum.outer(s, s)
    out = np.empty((horizon, len(post["drift"])))
    for i in range(out.shape[1]):
        mu = post["level0"][i] + post["drift"][i] * s + X @ post["beta"][i]
        K = post["sigma_level"][i] ** 2 * min_s + post["sigma_obs"][i] ** 2 * np.eye(n + horizon)
        Koo, Kfo, Kff = K[:n, :n], K[n:, :n], K[n:, n:]
        A = np.linalg.solve(Koo, Kfo.T).T
        m = mu[n:] + A @ (y_c - mu[:n])
        C = Kff - A @ Kfo.T
        out[:, i] = rng.multivariate_normal(m, (C + C.T) / 2, method="cholesky")
    return out


def pymc_forecast(y: pd.Series, horizon: int, seed: int = config.SEED, draws: int = 1000,
                  tune: int = 1000, chains: int = 4, target_accept: float = 0.99,
                  harmonics: int = config.SEASONAL_HARMONICS, progress: bool = False):
    """Returns (quantile DataFrame on the logit scale, arviz InferenceData)."""
    import pymc as pm

    model, ybar = build_model(y, harmonics)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
                          target_accept=target_accept, random_seed=seed, progressbar=progress)
    post = {k: idata.posterior[k].stack(s=("chain", "draw")).transpose("s", ...).values
            for k in ("sigma_obs", "sigma_level", "drift", "level0", "beta")}
    n = len(y)
    full_idx = pd.date_range(y.index[0], periods=n + horizon, freq="MS")
    fut = _sample_future(post, y.values - ybar, n, horizon, harmonics, full_idx.month.values,
                         np.random.default_rng(seed)) + ybar
    out = pd.DataFrame({config.qname(q): np.quantile(fut, q, axis=1) for q in config.QUANTILES},
                       index=full_idx[n:])
    return out, idata


def diagnostics(idata) -> dict:
    """Headline sampler health numbers + the ArviZ summary table."""
    import arviz as az

    summ = az.summary(idata, var_names=["sigma_obs", "sigma_level", "drift", "level0", "beta"])
    return {"summary": summ, "max_rhat": float(summ["r_hat"].max()),
            "min_ess_bulk": float(summ["ess_bulk"].min()),
            "min_ess_tail": float(summ["ess_tail"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum())}
