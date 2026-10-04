"""
trainer.py  —  P2 GRAPHON-MEAN-FIELD trainer

Builds a graphon mean-field model of each universe's ETFs: a rolling-
correlation graphon couples each ETF to a network-weighted peer consensus,
the resulting crowding gap feeds a closed-form (Riccati-solved) optimal
"liquidation" control and dynamic risk measure, and asks whether that
readout helps rank ETFs by forward return relative to the universe average,
over and above a baseline with none of it. See graphon_model.py for the
theory and the validation that led to each design choice (the Riccati
recursion checked against brute-force dynamic programming; kappa-fitting
checked against a known planted mean-reversion-to-consensus process; the
baseline ablation checked with a clean target-only signal injection after an
earlier engine's injection design gave a false negative).

Design choices (same shape as the sibling engines, for the same reasons):
  * Cross-sectional target: features and forward return are z-scored across
    ETFs each day, so market drift/beta cancel.
  * Baseline ablation: every configuration is evaluated for the graphon
    model AND a no-graphon baseline on the same dates; only the difference
    is credit.
  * Non-overlapping test points spaced `horizon` days apart; Sharpe
    annualized by sqrt(252/horizon); cost charged on the fraction of the
    top-N replaced.
  * Selection/holdout split, with a sample-size-aware selection gate (an
    absolute AND relative period-count floor) and confidence capped at Low
    whenever that gate didn't pass — both carried over from the sibling
    delay-agent-dynamics / hyperbolic-flows engines after real-data runs
    there showed what goes wrong without them.
  * The "contagion" claim is tested explicitly (a global-risk-index-leads-
    large-moves diagnostic, with an empirical permutation chance level —
    the analytic-formula bug found in the hyperbolic-flows engine is not
    repeated here), not assumed.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import graphon_model as gmod
from data_manager import load_master_data, validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d: Dict) -> Dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in d.items()}


def confidence_label(hold: Dict, gate_passed: bool = True) -> str:
    """
    Graded on the HOLDOUT of the selected configuration, and capped at Low
    whenever `gate_passed` is False — a configuration that never cleared its
    own selection-segment eligibility bar is not good evidence of anything
    systematic no matter how its holdout looks.
      High   : holdout rank-IC t >= IC_T_HIGH, net top-N spread > 0, and beats
               the baseline on both IC and net spread.
      Medium : IC > 0, net spread > 0, and beats the baseline on both.
      Low    : anything else, fewer than MIN_HOLDOUT_PERIODS test periods, or
               gate_passed is False.
    """
    if not gate_passed:
        return "Low"
    if hold.get("n", 0) < config.MIN_HOLDOUT_PERIODS:
        return "Low"
    ic, sp = hold["ic_mean"], hold["net_spread_mean"]
    d_ic, d_sp = hold["d_ic_mean"], hold["d_spread_mean"]
    beats_baseline = d_ic > 0 and d_sp > 0
    if hold["ic_t"] >= config.IC_T_HIGH and sp > 0 and beats_baseline:
        return "High"
    if ic > 0 and sp > 0 and beats_baseline:
        return "Medium"
    return "Low"


def select_config(grid: List[Dict], corr_window: Optional[int] = None):
    """
    Choose a configuration from the SELECTION segment only: eligible if
    rank-IC and net top-N spread are both positive there, AND the config's
    selection-segment period count clears both an absolute floor
    (MIN_SELECTION_N) and a floor relative to the largest period count in
    this pool (MIN_SELECTION_N_FRACTION). Among eligible configs, rank-sum of
    the two t-stats. If nothing is eligible, fall back to the size-filtered
    pool and report gate_passed=False.
    """
    pool = [g for g in grid if (corr_window is None or g["corr_window"] == corr_window)
            and g["selection"].get("n", 0) >= config.MIN_SELECTION_N]
    if not pool:
        return None, False
    max_n = max(g["selection"]["n"] for g in pool)
    pool = [g for g in pool if g["selection"]["n"] >= config.MIN_SELECTION_N_FRACTION * max_n]
    if not pool:
        return None, False
    elig = [g for g in pool if g["selection"]["ic_mean"] > 0 and g["selection"]["net_spread_mean"] > 0]
    gate = bool(elig)
    cand = elig if gate else pool
    ic_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["ic_t"]))}
    sp_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["net_spread_t"]))}
    best = max(cand, key=lambda g: (ic_order[id(g)] + sp_order[id(g)], g["selection"]["ic_t"]))
    return best, gate


def make_picks(scores: np.ndarray, tickers: List[str], dispersion: float, horizon: int,
               confidence: str, top_n: int) -> List[Dict]:
    order = np.argsort(-scores)[:top_n]
    return [{
        "ticker": tickers[j],
        "expected_return": round(float(scores[j] * dispersion * 100), 3),
        "score_z": round(float(scores[j]), 4),
        "horizon_days": int(horizon),
        "confidence": confidence,
    } for j in order]


def pattern_diagnostics(extras: Dict, tickers: List[str], dates: List[str]) -> Dict:
    """Today's snapshot + global systemic risk index series, for the dashboard."""
    n = len(dates)
    gap, kappa, risk, net_stress = extras["gap"], extras["kappa"], extras["risk_measure"], extras["network_stress"]
    W, deg = extras["W"], extras["deg"]
    last = n - 1
    out = {}
    valid_today = np.isfinite(gap[last])
    if valid_today.any():
        j = np.where(valid_today)[0]
        snapshot = [{"ticker": tickers[k], "gap": _f(gap[last, k]), "kappa": _f(kappa[last, k]),
                    "risk_measure": _f(risk[last, k]), "network_stress": _f(net_stress[last, k])}
                   for k in j]
        out["snapshot"] = snapshot
        out["n_decoupling"] = int(np.sum(np.nan_to_num(kappa[last, j], nan=1.0) < 0))
        out["n_covered"] = int(len(j))
        # top network edges today (strongest graphon ties), for a simple edge list diagnostic
        Wt = W[last]
        iu = np.triu_indices(len(tickers), k=1)
        strengths = Wt[iu]
        order = np.argsort(-strengths)[:10]
        edges = [{"a": tickers[iu[0][o]], "b": tickers[iu[1][o]], "weight": _f(strengths[o])}
                for o in order if strengths[o] > 0]
        out["top_edges"] = edges

    global_risk = np.nanmean(risk, axis=1)
    valid_hist = np.isfinite(global_risk)
    idx = np.arange(0, n, config.SERIES_STEP)
    idx = idx[valid_hist[idx]]
    out["series"] = {
        "dates": [dates[i] for i in idx],
        "global_risk_index": [_f(global_risk[i]) for i in idx],
    }
    out["global_risk_series"] = global_risk
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px_df = prices_df[available]
    px_df = px_df[~px_df.isna().any(axis=1)]
    if len(px_df) < 400:
        logger.warning(f"Not enough data for {name}")
        return None

    values = px_df.values
    returns = np.diff(np.log(values), axis=0)
    log_prices = np.log(values[1:])
    dates = [d.strftime("%Y-%m-%d") for d in px_df.index[1:]]
    n = len(dates)

    cfg = dict(ridge_lambda=config.RIDGE_LAMBDA, min_train_days=config.MIN_TRAIN_DAYS, cost_bps=config.TRADING_COST_BPS)
    fwd_by_h = {h: gmod.compute_forward_returns(returns, h) for h in config.HORIZONS}
    yz_by_h = {h: gmod.cs_zscore(fwd_by_h[h]) for h in config.HORIZONS}

    agg_returns = np.nanmean(returns, axis=1)
    L = 21
    s1, s2 = gmod.trailing_sum(agg_returns.reshape(-1, 1), L), gmod.trailing_sum((agg_returns ** 2).reshape(-1, 1), L)
    agg_vol21 = np.sqrt(np.maximum((s2 - s1 ** 2 / L) / (L - 1), 1e-10)).flatten()

    grid, wf_store, live_ctx, patterns = [], {}, {}, {}
    for cw in config.CORR_WINDOWS:
        logger.info(f"  [{name}] corr_window={cw}: building graphon feature series...")
        Fz_raw, extras = gmod.build_feature_array(
            returns, log_prices, cw, config.TREND_SPAN, config.GRAPHON_POWER,
            config.KAPPA_FIT_WINDOW, config.MIN_KAPPA_FIT_DAYS, config.GAP_CHANGE_LOOKBACK,
            config.LQR_HORIZON, config.LQR_Q, config.LQR_TERMINAL_COST,
        )
        Fz = gmod.cs_zscore(Fz_raw)
        pat = pattern_diagnostics(extras, available, dates)
        pat["contagion_lead"] = gmod.contagion_lead_diagnostic(pat.pop("global_risk_series"), agg_returns, agg_vol21, dates, config.LEAD_PARAMS)
        patterns[cw] = pat

        for h in config.HORIZONS:
            cp = gmod.build_cross_products(Fz, yz_by_h[h])
            for window in config.WINDOWS:
                wf = gmod.walk_forward(Fz, cp, fwd_by_h[h], h, window, config.TOP_N, cfg)
                key = (cw, h, window)
                live_ctx[key] = (Fz, cp)
                if wf is None:
                    continue
                m = len(wf["rows"])
                split = int(m * config.SELECTION_FRACTION)
                entry = {
                    "corr_window": cw, "horizon": h, "window": window, "n_periods": m,
                    "selection": _clean(gmod.summarize(wf, 0, split, h)),
                    "holdout": _clean(gmod.summarize(wf, split, m, h)),
                    "full": _clean(gmod.summarize(wf, 0, m, h)),
                }
                grid.append(entry)
                wf_store[key] = (wf, split)
        logger.info(f"  [{name}] corr_window={cw}: done ({len([g for g in grid if g['corr_window'] == cw])} configs)")

    if not grid:
        return None

    def live_for(g):
        Fz, cp = live_ctx[(g["corr_window"], g["horizon"], g["window"])]
        return gmod.live_scores(Fz, cp, fwd_by_h[g["horizon"]], g["horizon"], g["window"], cfg)

    selected, gate = select_config(grid)
    pinned = False
    pin = config.PINNED_CONFIG
    if pin:
        match = [g for g in grid if g["corr_window"] == pin["corr_window"] and g["horizon"] == pin["horizon"]
                and g["window"] == pin["window"]]
        if match:
            selected, pinned = match[0], True
            gate = True
    if selected is None:
        return None

    ls = live_for(selected)
    conf = confidence_label(selected["holdout"], gate)
    picks = base_picks = []
    if ls is not None:
        picks = make_picks(ls["score_m"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)
        base_picks = make_picks(ls["score_b"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)

    cw_picks = {}
    for cw in patterns:
        g, sgate = select_config(grid, cw)
        if g is None:
            continue
        lst = live_for(g)
        sconf = confidence_label(g["holdout"], sgate)
        cw_picks[str(cw)] = {
            "config": {"corr_window": cw, "horizon": g["horizon"], "window": g["window"]},
            "gate_passed": sgate,
            "holdout": g["holdout"],
            "confidence": sconf,
            "picks": make_picks(lst["score_m"], available, lst["dispersion"], g["horizon"],
                                sconf, config.TOP_N) if lst is not None else [],
        }

    wf, split = wf_store[(selected["corr_window"], selected["horizon"], selected["window"])]
    curves = {
        "dates": [dates[i] for i in wf["rows"]],
        "graphon_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_m"])],
        "baseline_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_b"])],
        "split_index": int(split),
    }

    top = {
        "corr_window": selected["corr_window"], "horizon": selected["horizon"], "window": selected["window"],
        "gate_passed": gate, "pinned": pinned, "confidence": conf,
        "selection": selected["selection"], "holdout": selected["holdout"], "full": selected["full"],
        "n_periods": selected["n_periods"],
    }
    feature_weights = None
    if ls is not None:
        feature_weights = {nm: round(float(w), 5) for nm, w in zip(gmod.FEATURE_NAMES, ls["beta_m"])}

    return {
        "tickers": available, "n_days": n, "last_date": dates[-1],
        "selected": top, "picks": picks, "baseline_picks": base_picks,
        "grid": grid, "cw_picks": cw_picks, "curves": curves,
        "patterns": {str(k): v for k, v in patterns.items()}, "feature_weights": feature_weights,
    }


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date, "algorithm": "GRAPHON-MEAN-FIELD",
        "top_picks": {}, "selected": {}, "grid": {}, "cw_picks": {}, "baseline_picks": {},
        "curves": {}, "patterns": {}, "feature_weights": {}, "universe_meta": {},
        "settings": {"trading_cost_bps": config.TRADING_COST_BPS, "top_n": config.TOP_N,
                    "selection_fraction": config.SELECTION_FRACTION, "lqr_horizon": config.LQR_HORIZON,
                    "lqr_q": config.LQR_Q, "lqr_terminal_cost": config.LQR_TERMINAL_COST},
    }

    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        results["top_picks"][name] = out["picks"]
        results["selected"][name] = out["selected"]
        results["grid"][name] = out["grid"]
        results["cw_picks"][name] = out["cw_picks"]
        results["baseline_picks"][name] = out["baseline_picks"]
        results["curves"][name] = out["curves"]
        results["patterns"][name] = out["patterns"]
        results["feature_weights"][name] = out["feature_weights"]
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"], "last_date": out["last_date"]}

        s = out["selected"]
        h = s["holdout"]
        logger.info(f"  ✅ Selected: corr_window={s['corr_window']} / {s['horizon']}d / window {s['window']} "
                    f"(selection gate {'passed' if s['gate_passed'] else 'NOT passed'}) | holdout n={h.get('n')} "
                    f"IC={h.get('ic_mean', 0):+.4f} (t={h.get('ic_t', 0):+.2f}) "
                    f"vs baseline IC={h.get('ic_base_mean', 0):+.4f} | net top-{config.TOP_N} excess "
                    f"{h.get('net_spread_mean', 0) * 1e4:+.1f}bps/period | confidence {s['confidence']}")
        for p in out["picks"]:
            logger.info(f"     {p['ticker']}: {p['expected_return']:+.3f}% excess over {p['horizon_days']}d ({p['confidence']})")

    output_path = f"graphon_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")

    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
