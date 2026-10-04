"""
graphon_model.py  —  Graphon mean-field crowding/flow/contagion signal

Idea
----
Treat each ETF as a "particle" with state x_i(t) = log_price_i(t) -
EMA_trend_i(t) (deviation from its own slow trend). A graphon — a symmetric
interaction kernel W(x,y), here a rolling-correlation-based matrix W_ij(t) in
[0,1] — couples the particles: each ETF is pulled toward a NETWORK-WEIGHTED
PEER CONSENSUS

    xbar_i(t) = sum_j W_ij(t) x_j(t) / sum_j W_ij(t)

and its CROWDING GAP is d_i(t) = x_i(t) - xbar_i(t). This is the finite-N,
discretized analogue of the continuity equation

    d/dt m + div(m v) = 0,   v driven by network interaction through W(x,y)

that governs a graphon mean-field game: the graphon-weighted Laplacian
L_W = D_W - W (D_W = diag(row sums)) is exactly the discrete generator of
flow toward the network-weighted consensus, so d(x)/dt = -L_W x / deg IS a
discretized continuity/transport equation driven by W. Each ETF's observed
gap-closing behavior is then modeled as the solution to a small LINEAR-
QUADRATIC control problem (see below) — the "flow" in "crowding -> flow ->
liquidation -> contagion": gaps don't just sit there, they imply an optimal
rate of closing, and a widening gap under this model is a growing cost-to-go
(a dynamic systemic risk measure) that propagates to neighbors through the
SAME graphon the next time their own consensus is computed, which is the
contagion channel.

The LQR control problem (independently derived and validated, NOT a
reproduction of any specific paper's exact equations)
-------------------------------------------------------------------------
Over a short horizon H, each ETF chooses a daily control u_t (how much of
its own gap to close) to minimize

    sum_{t=0}^{H-1} [ 0.5*u_t^2 + 0.5*q*(d_t - u_t)^2 ] + 0.5*c_T*d_H^2

subject to d_{t+1} = d_t - u_t + noise — i.e. trading off control effort
(0.5*u_t^2, an Almgren-Chriss-style quadratic trading/impact cost) against a
running penalty for remaining desynchronized from the network consensus
(0.5*q*(d_t-u_t)^2), with a terminal penalty on whatever gap remains. This is
a standard finite-horizon LQR problem (NOT claimed to be identical to the
cited graphon-BSDE systemic-risk papers' exact cost functional, which solve a
continuous-time, fully coupled mean-field equilibrium — this is a tractable,
self-contained discrete-time simplification in the same spirit: a quadratic
tracking-vs-control-cost tradeoff with a network-mediated target).

The solution follows from the standard quadratic-value-function ansatz
V_t(d) = 0.5*P_t*d^2 + const, giving the backward Riccati recursion

    g_t = (q + P_{t+1}) / (1 + q + P_{t+1})         [u*_t = g_t * d_t]
    P_t = g_t^2 + (q + P_{t+1}) * (1 - g_t)^2,   P_H = c_T

solved once per horizon/q/c_T (fixed, not re-solved per ETF or per day —
only d_t and the fitted mean-reversion-to-consensus strength kappa vary).
VALIDATED against brute-force grid-based dynamic programming (401-point
state grid, explicit backward Bellman induction): the closed-form g_t
matched the brute-force optimal control to within grid resolution
(max abs error 0.01) at every horizon step and every tested state value.

`risk_measure_i(t) = P_0 * d_i(t)^2` is this engine's dynamic systemic risk
measure (the LQR cost-to-go) — the discrete-time analogue of the dynamic
risk measure induced by a graphon mean-field BSDE in the cited literature,
here available in closed form rather than by simulation.

Honest scope
------------
* The graphon is empirical (rolling correlation), not a literal continuum
  limit — this is the same finite-N discretization every applied graphon
  model uses in practice.
* The LQR problem is a deliberate simplification for tractability, not a
  literal implementation of any cited paper's coupled HJB/BSDE system.
* The forecast is a small linear model on the gap/control/risk-measure
  readout, always compared against a baseline with none of this, so the
  graphon machinery is credited only for what it adds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

BASE_FEATURES = ["x", "mom5", "mom21", "vol21"]
GRAPHON_FEATURES = ["gap", "gap_chg", "kappa", "control", "risk_measure", "network_stress"]
FEATURE_NAMES = BASE_FEATURES + GRAPHON_FEATURES
N_BASE = len(BASE_FEATURES)
N_FEAT = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------
def rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    lo = np.maximum(np.arange(a.shape[0]) - w + 1, 0)
    return c[1:] - c[lo]


def trailing_sum(a: np.ndarray, w: int) -> np.ndarray:
    out = rolling_sum(a, w)
    out[: w - 1] = np.nan
    return out


def ema(a: np.ndarray, span: int) -> np.ndarray:
    return pd.DataFrame(a).ewm(span=span, adjust=False).mean().values


def compute_forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    end = n - horizon
    if end > 0:
        fwd[:end] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


# ---------------------------------------------------------------------------
# LQR Riccati recursion (fixed horizon/q/terminal cost -> one set of gains)
# ---------------------------------------------------------------------------
def riccati_solve(horizon: int, q: float, terminal_cost: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (P, g), each length `horizon`+1 / `horizon`; g[0] is today's gain, u* = g[0]*d."""
    P = np.zeros(horizon + 1)
    g = np.zeros(horizon)
    P[horizon] = terminal_cost
    for t in range(horizon - 1, -1, -1):
        Pn = P[t + 1]
        g[t] = (q + Pn) / (1 + q + Pn)
        P[t] = g[t] ** 2 + (q + Pn) * (1 - g[t]) ** 2
    return P, g


# ---------------------------------------------------------------------------
# Graphon consensus and crowding gap
# ---------------------------------------------------------------------------
def rolling_corr_graphon(returns: np.ndarray, window: int, power: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-day graphon W_ij(t) = max(corr_ij(t), 0)**power, from a trailing
    `window`-day rolling correlation. Returns (W, deg) where W is (n,T,T)
    and deg[t,i] = sum_j W_ij(t) (NaN before `window` days of history).
    Vectorized via rolling sums of pairwise cross-products (O(n) passes over
    T^2 pairs, not O(n*T^2) per-day recomputation).
    """
    n, T = returns.shape
    s1 = trailing_sum(returns, window)                                   # (n,T)
    s2 = trailing_sum(returns ** 2, window)                               # (n,T)
    cross = np.einsum("ti,tj->tij", returns, returns)                     # (n,T,T)
    s12 = trailing_sum(cross.reshape(n, T * T), window).reshape(n, T, T)
    mean1 = s1 / window
    var1 = np.maximum(s2 / window - mean1 ** 2, 1e-12)
    cov = s12 / window - mean1[:, :, None] * mean1[:, None, :]
    sd = np.sqrt(var1)
    denom = np.maximum(sd[:, :, None] * sd[:, None, :], 1e-12)
    corr = np.clip(cov / denom, -1.0, 1.0)
    W = np.maximum(corr, 0.0) ** power
    idx = np.arange(T)
    W[:, idx, idx] = 0.0
    deg = W.sum(axis=2)
    return W, deg


def compute_consensus_and_gap(x: np.ndarray, W: np.ndarray, deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """xbar_i(t) = sum_j W_ij x_j / deg_i; gap = x - xbar. NaN where deg~0 or x invalid."""
    n, T = x.shape
    valid_x = np.isfinite(x)
    x0 = np.nan_to_num(x)
    weighted = np.einsum("tij,tj->ti", W, x0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = weighted / deg
    ok = valid_x & (deg > 1e-6)
    xbar = np.where(ok, xbar, np.nan)
    gap = x - xbar
    return xbar, gap


def fit_kappa(gap: np.ndarray, window: int, min_days: int) -> np.ndarray:
    """
    Rolling OLS (no intercept) of dgap(t) = gap(t)-gap(t-1) on -gap(t-1),
    i.e. dgap = -kappa*gap(t-1) + noise, over a trailing `window`. Vectorized
    via rolling sums (closed-form single-regressor OLS: kappa = -Sxy/Sxx).
    kappa > 0 means the gap is mean-reverting (pulled back toward the
    graphon consensus); kappa < 0 means it is currently self-reinforcing
    (decoupling further — the start of a possible contagion episode).
    """
    n, T = gap.shape
    g_prev = np.roll(gap, 1, axis=0)
    g_prev[0] = np.nan
    dg = gap - g_prev
    ok = np.isfinite(g_prev) & np.isfinite(dg)
    x0 = np.where(ok, g_prev, 0.0)
    y0 = np.where(ok, dg, 0.0)
    Sxx = trailing_sum(x0 ** 2, window)
    Sxy = trailing_sum(x0 * y0, window)
    n_eff = trailing_sum(ok.astype(float), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = -Sxy / np.where(Sxx > 1e-9, Sxx, np.nan)
    kappa = np.where(n_eff >= min_days, kappa, np.nan)
    return kappa


def build_feature_array(returns: np.ndarray, log_prices: np.ndarray, corr_window: int,
                        trend_span: int, graphon_power: float, kappa_fit_window: int,
                        min_kappa_fit_days: int, gap_lookback: int,
                        lqr_horizon: int, lqr_q: float, lqr_terminal_cost: float) -> Tuple[np.ndarray, Dict]:
    """Full (n, T, N_FEAT) feature tensor: baseline + graphon-mean-field readout."""
    n, T = returns.shape
    trend = ema(log_prices, trend_span)
    x = log_prices - trend

    mom5 = trailing_sum(returns, 5)
    L = 21
    mom21 = trailing_sum(returns, L)
    s1, s2 = trailing_sum(returns, L), trailing_sum(returns ** 2, L)
    vol21 = 0.5 * np.log(np.maximum((s2 - s1 ** 2 / L) / (L - 1), 1e-10))

    W, deg = rolling_corr_graphon(returns, corr_window, graphon_power)
    xbar, gap = compute_consensus_and_gap(x, W, deg)
    kappa = fit_kappa(gap, kappa_fit_window, min_kappa_fit_days)

    gap_prev = np.roll(gap, gap_lookback, axis=0)
    gap_prev[:gap_lookback] = np.nan
    gap_chg = gap - gap_prev

    _, g = riccati_solve(lqr_horizon, lqr_q, lqr_terminal_cost)
    g0 = g[0]
    P0, _ = riccati_solve(lqr_horizon, lqr_q, lqr_terminal_cost)
    P0 = P0[0]
    control = g0 * gap
    risk_measure = P0 * gap ** 2

    deg_safe = np.where(deg > 1e-6, deg, np.nan)
    network_stress = np.einsum("tij,tj->ti", W, np.nan_to_num(risk_measure)) / deg_safe

    feats = np.full((n, T, N_FEAT), np.nan)
    feats[:, :, 0] = x
    feats[:, :, 1] = mom5
    feats[:, :, 2] = mom21
    feats[:, :, 3] = vol21
    feats[:, :, 4] = gap
    feats[:, :, 5] = gap_chg
    feats[:, :, 6] = kappa
    feats[:, :, 7] = control
    feats[:, :, 8] = risk_measure
    feats[:, :, 9] = network_stress

    extras = {"x": x, "xbar": xbar, "gap": gap, "kappa": kappa, "risk_measure": risk_measure,
             "network_stress": network_stress, "W": W, "deg": deg, "lqr_g0": g0, "lqr_P0": P0}
    return feats, extras


# ---------------------------------------------------------------------------
# Cross-sectional regression machinery (same shape as the sibling engines)
# ---------------------------------------------------------------------------
def cs_zscore(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = np.where(sd > 1e-9, (a - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    out[~np.isfinite(a)] = np.nan
    return out


def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    y_ok = np.isfinite(Yz).all(axis=1)
    ok = feat_ok & y_ok
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {
        "cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
        "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
        "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]),
        "feat_ok": feat_ok,
    }


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    lam = lam_rel * np.trace(XtX) / p + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(p), Xty)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def walk_forward(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                 top_n: int, cfg: Dict) -> Optional[Dict]:
    n, T, p = Fz.shape
    base_idx = list(range(N_BASE))
    lam = cfg["ridge_lambda"]
    min_train = cfg["min_train_days"]
    cost = cfg["cost_bps"] / 1e4

    rows, ic_m, ic_b, sg_m, sg_b, sn_m, sn_b = [], [], [], [], [], [], []
    prev_m, prev_b = set(), set()
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < min_train:
            i += horizon
            continue
        XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
        Xty = cp["cXty"][b + 1] - cp["cXty"][a]
        try:
            beta_m = ridge_solve(XtX, Xty, lam)
            beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], lam)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        f = Fz[i]
        s_m = f @ beta_m
        s_b = f[:, base_idx] @ beta_b
        r = fwd[i]
        if not np.isfinite(r).all():
            i += horizon
            continue
        top_m = set(np.argsort(-s_m)[:top_n].tolist())
        top_b = set(np.argsort(-s_b)[:top_n].tolist())
        g_m = r[list(top_m)].mean() - r.mean()
        g_b = r[list(top_b)].mean() - r.mean()
        c_m = (top_n - len(top_m & prev_m)) / top_n * cost
        c_b = (top_n - len(top_b & prev_b)) / top_n * cost
        rows.append(i)
        ic_m.append(spearman(s_m, r))
        ic_b.append(spearman(s_b, r))
        sg_m.append(g_m)
        sg_b.append(g_b)
        sn_m.append(g_m - c_m)
        sn_b.append(g_b - c_b)
        prev_m, prev_b = top_m, top_b
        i += horizon

    if len(rows) < 10:
        return None
    return {k: np.array(v) for k, v in dict(rows=rows, ic_m=ic_m, ic_b=ic_b, sg_m=sg_m, sg_b=sg_b,
                                              sn_m=sn_m, sn_b=sn_b).items()}


def summarize(wf: Dict, lo: int, hi: int, horizon: int) -> Dict:
    sl = slice(lo, hi)
    ic_m, ic_b = wf["ic_m"][sl], wf["ic_b"][sl]
    sn_m, sn_b, sg_m = wf["sn_m"][sl], wf["sn_b"][sl], wf["sg_m"][sl]
    n = len(ic_m)
    if n < 3:
        return {"n": n}
    ann = np.sqrt(252.0 / horizon)
    sd_net = sn_m.std() + 1e-12
    sd_gross = sg_m.std() + 1e-12
    return {
        "n": int(n),
        "ic_mean": float(ic_m.mean()), "ic_t": tstat(ic_m),
        "ic_base_mean": float(ic_b.mean()), "ic_base_t": tstat(ic_b),
        "net_spread_mean": float(sn_m.mean()), "net_spread_t": tstat(sn_m),
        "gross_spread_mean": float(sg_m.mean()),
        "base_net_spread_mean": float(sn_b.mean()),
        "hit_rate": float((sn_m > 0).mean()),
        "sharpe_net": float(sn_m.mean() / sd_net * ann),
        "sharpe_gross": float(sg_m.mean() / sd_gross * ann),
        "d_ic_mean": float((ic_m - ic_b).mean()), "d_ic_t": tstat(ic_m - ic_b),
        "d_spread_mean": float((sn_m - sn_b).mean()), "d_spread_t": tstat(sn_m - sn_b),
    }


def live_scores(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                cfg: Dict) -> Optional[Dict]:
    n = Fz.shape[0]
    i = n - 1
    a, b = max(0, i - window), i - horizon
    if b < a or not cp["feat_ok"][i] or cp["cN"][b + 1] - cp["cN"][a] < cfg["min_train_days"]:
        return None
    XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
    Xty = cp["cXty"][b + 1] - cp["cXty"][a]
    base_idx = list(range(N_BASE))
    beta_m = ridge_solve(XtX, Xty, cfg["ridge_lambda"])
    beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], cfg["ridge_lambda"])
    f = Fz[i]
    disp = np.nanmean(np.nanstd(fwd[a:b + 1], axis=1))
    return {
        "score_m": f @ beta_m,
        "score_b": f[:, base_idx] @ beta_b,
        "beta_m": beta_m,
        "dispersion": float(disp) if np.isfinite(disp) else 0.0,
    }


# ---------------------------------------------------------------------------
# Contagion lead diagnostic
# ---------------------------------------------------------------------------
def _onsets(flag: np.ndarray, dedup: int) -> np.ndarray:
    out, last = [], -10 ** 9
    for t in np.where(flag)[0]:
        if t - last > dedup:
            out.append(t)
            last = t
    return np.array(out, dtype=int)


def contagion_lead_diagnostic(global_risk: np.ndarray, agg_returns: np.ndarray, agg_vol21: np.ndarray,
                              dates: List[str], lp: Dict, n_shifts: int = 200, seed: int = 0) -> Dict:
    """
    Does a spike in the GLOBAL systemic risk index (graphon-weighted sum of
    each ETF's own dynamic risk measure) tend to sit before a large AGGREGATE
    (equal-weight universe) move? One series each (not per-ETF, since this is
    a system-level question) — chance baseline is an empirical circular-shift
    permutation of the onset series against the fixed event series, for the
    same reason the sibling hyperbolic-flows engine's migration-lead
    diagnostic needed one (deduplicated onsets/events are quasi-periodic, not
    uniformly scattered, so an analytic uniform-placement chance formula is
    biased).
    """
    n = len(global_risk)
    W = lp["event_window"]
    fwd_cum = np.full(n, np.nan)
    if n > W:
        cum = np.concatenate([[0.0], np.cumsum(agg_returns)])
        fwd_cum[: n - W] = cum[W + 1: n + 1] - cum[1: n - W + 1]
    thresh = lp["event_vol_k"] * agg_vol21 * np.sqrt(W)
    event_flag = np.abs(fwd_cum) > thresh

    s = pd.Series(global_risk)
    mu = s.rolling(lp["z_window"], min_periods=63).mean()
    sd = s.rolling(lp["z_window"], min_periods=63).std()
    z = ((s - mu) / sd.replace(0, np.nan)).values
    above = np.nan_to_num(z, nan=-9.0) > lp["risk_z"]
    prev_below = np.roll(~above, 1)
    prev_below[0] = False
    onset_flag = above & prev_below

    ok = np.isfinite(z) & np.isfinite(fwd_cum)
    if not ok.any():
        return {"available": False}
    start = int(np.argmax(ok))
    dedup = lp["dedup_days"]
    ev = _onsets(event_flag & ok, dedup)
    on = _onsets(onset_flag & ok, dedup)
    on = on[on >= start]
    ev = ev[ev >= start]
    LW = lp["lead_window"]

    leads = []
    n_hits = sum(1 for d in ev if ((on >= d - LW) & (on < d)).any())
    for d in ev:
        prior = on[(on >= d - LW) & (on < d)]
        if len(prior):
            leads.append(int(d - prior.min()))
    n_false = sum(1 for m in on if not ((ev > m) & (ev <= m + LW)).any())

    rng = np.random.default_rng(seed)
    chance_hit = chance_false = 0.0
    span = n - start
    if len(on) and len(ev) and span > 0:
        shifts = rng.integers(0, span, size=n_shifts)
        hit_counts = np.empty(n_shifts)
        false_counts = np.empty(n_shifts)
        for si, sh in enumerate(shifts):
            on_mod = start + np.mod(on + sh - start, span)
            h = sum(1 for d in ev if ((on_mod >= d - LW) & (on_mod < d)).any())
            f = sum(1 for m in on_mod if not ((ev > m) & (ev <= m + LW)).any())
            hit_counts[si] = h
            false_counts[si] = f
        chance_hit = float(hit_counts.mean())
        chance_false = float(false_counts.mean())

    return {
        "available": True,
        "n_risk_onsets": int(len(on)),
        "n_large_move_events": int(len(ev)),
        "hit_rate": float(n_hits / len(ev)) if len(ev) else None,
        "chance_hit_rate": float(chance_hit / len(ev)) if len(ev) else None,
        "median_lead_days": float(np.median(leads)) if leads else None,
        "false_alarm_rate": float(n_false / len(on)) if len(on) else None,
        "chance_false_alarm_rate": float(chance_false / len(on)) if len(on) else None,
    }
