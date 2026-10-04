"""
config.py  —  Configuration for P2 GRAPHON-MEAN-FIELD
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-graphon-mean-field-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

TOP_N = 3

# ---------------------------------------------------------------------------
# Graphon mean-field model
#
# Each ETF i has a state x_i(t) = log_price_i(t) - EMA_trend_i(t) (deviation
# from its own slow trend; the same price-only proxy used across the sibling
# engines — no real fundamental data exists here). A graphon W_ij(t) (a
# rolling-correlation-based interaction kernel, symmetric, in [0,1]) gives
# each ETF a NETWORK-WEIGHTED PEER CONSENSUS:
#
#   xbar_i(t) = sum_j W_ij(t) x_j(t) / sum_j W_ij(t)
#
# and a CROWDING GAP  d_i(t) = x_i(t) - xbar_i(t)  — how far this ETF has
# drifted from what its correlated peers are doing. This is the discretized,
# finite-N analogue of a graphon mean-field interaction: the "mass" at each
# node is pulled toward a graphon-weighted average of its neighbors, which is
# exactly how a continuity equation dt(m) + div(m*v) = 0 with network
# interaction W(x,y) behaves once discretized onto a finite graph (the
# graphon-weighted Laplacian is the discrete generator of that flow) — see
# graphon_model.py's module docstring for the full derivation and honest
# scope notes.
#
# Each ETF then solves a finite-horizon LINEAR-QUADRATIC control problem,
# inspired by (not a reproduction of) the graphon-extended systemic-risk
# literature (Amini, Cao & Sulem 2022/2025, "Graphon mean-field backward
# stochastic differential equations... and associated dynamic risk
# measures"; Carmona-Fouque-Sun-style interbank/mutual-holding models with
# heterogeneous graphon interaction): choose a daily "liquidation" u_i(t) to
# partially close its own gap, trading off control effort against staying
# desynchronized from peers. This has a CLOSED-FORM solution (a backward
# Riccati recursion, independently derived and validated against brute-force
# dynamic programming — see repo notes), giving:
#   control        u*_i(t) = g(t) * d_i(t)     — the model's implied daily
#                                                 "crowding-closing flow"
#   risk_measure   P(t) * d_i(t)^2             — a dynamic risk measure (the
#                                                 LQR cost-to-go), the
#                                                 discrete-time analogue of
#                                                 the graphon-BSDE dynamic
#                                                 risk measure cited above
# ---------------------------------------------------------------------------
TREND_SPAN = 126
KAPPA_FIT_WINDOW = 126
MIN_KAPPA_FIT_DAYS = 60
GAP_CHANGE_LOOKBACK = 10

# Graphon kernel: W_ij = max(corr_ij, 0) ** GRAPHON_POWER (only positive
# correlation counts as a network tie; the power sharpens or flattens how
# much stronger ties dominate the weighted consensus).
GRAPHON_POWER = 2.0

# Grid axis: how fast the graphon/consensus responds to changing
# correlations (the rolling correlation window). Analogous to
# local_window / trend_span / short_corr_window in the sibling engines.
CORR_WINDOWS = [21, 42, 63]

# LQR control-problem parameters (fixed, not searched — these define the
# control problem itself, not a predictive hyperparameter):
#   horizon      : control problem horizon in trading days
#   q            : relative weight on staying desynchronized from peers vs
#                  control effort (higher q = closes the gap faster)
#   terminal_cost: terminal penalty weight on the remaining gap
LQR_HORIZON = 20
LQR_Q = 1.5
LQR_TERMINAL_COST = 3.0

# ---------------------------------------------------------------------------
# Prediction: pooled ridge regression of the cross-sectionally z-scored
# forward H-day return on cross-sectionally z-scored features, refit at each
# evaluation date on the trailing WINDOW days.
#   baseline : mom5, mom21, vol21, x (deviation from own trend) — no graphon.
#   graphon  : baseline + gap, gap_chg, kappa, control, risk_measure,
#              network_stress. Credited only for what it adds over baseline.
# ---------------------------------------------------------------------------
HORIZONS = [1, 5, 20]
WINDOWS = [252, 504, 756, 1008]
RIDGE_LAMBDA = 0.01
MIN_TRAIN_DAYS = 60

SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
TRADING_COST_BPS = 15
IC_T_HIGH = 1.65

# Selection eligibility requires a configuration's selection-segment period
# count to clear BOTH an absolute floor and a floor relative to the largest
# period count in the pool, and confidence is capped at Low whenever the
# selection gate didn't pass — both carried over directly from the sibling
# delay-agent-dynamics / hyperbolic-flows repos after real-data runs showed
# what goes wrong without them.
MIN_SELECTION_N = 50
MIN_SELECTION_N_FRACTION = 0.15

# Optional: pin the live configuration instead of re-selecting daily, e.g.
# {"corr_window": 63, "horizon": 5, "window": 756}. None = select.
PINNED_CONFIG = None

# Contagion lead diagnostic: does a spike in the GLOBAL systemic risk index
# (graphon-weighted sum of each ETF's own dynamic risk measure) tend to sit
# before a large aggregate (equal-weight universe) move? Both trigger
# definitions fixed here, not tuned. Chance level is an empirical
# circular-shift permutation (validated — see the sibling hyperbolic-flows
# repo's README for why an analytic uniform-placement formula is biased when
# onsets/events are deduplicated to a minimum spacing).
LEAD_PARAMS = {
    "z_window": 252,
    "risk_z": 1.5,
    "event_window": 10,
    "event_vol_k": 2.0,
    "dedup_days": 21,
    "lead_window": 20,
}

SERIES_STEP = 3
