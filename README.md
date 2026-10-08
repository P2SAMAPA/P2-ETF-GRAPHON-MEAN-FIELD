# P2-ETF-GRAPHON-MEAN-FIELD

A graphon mean-field model of ETF crowding, flow and contagion — not merely
a crowding measurement, but a model of what crowding implies happens next.
Distinct from the existing `MEAN-FIELD-CROWDING` and `MEAN-FIELD-GAME` repos.

## The idea

Treat each ETF as a particle with state `x_i(t) = log_price_i(t) -
EMA_trend_i(t)` (deviation from its own slow trend — the same price-only
proxy used across the sibling engines; no real fundamental data exists
here). A **graphon** — a symmetric interaction kernel `W(x,y)`, here a
rolling-correlation-based matrix `W_ij(t)` in [0,1] — couples the particles:
each ETF is pulled toward a **network-weighted peer consensus**

```
xbar_i(t) = sum_j W_ij(t) x_j(t) / sum_j W_ij(t)
```

and its **crowding gap** is `d_i(t) = x_i(t) - xbar_i(t)`. This is the
finite-N discretization of the continuity equation `∂_t m + ∇·(mv) = 0`
driven by network interaction through `W(x,y)`: the graphon-weighted
Laplacian `L_W = D_W - W` is exactly the discrete generator of flow toward
the network-weighted consensus.

**Crowding → flow**: each ETF solves a small finite-horizon **linear-
quadratic control problem** — choose a daily "liquidation" `u_t` to close its
own gap, trading control effort (an Almgren-Chriss-style quadratic trading
cost) against staying desynchronized from its network peers. This has a
**closed-form solution** (a backward Riccati recursion), giving:

* `control = g(t) * d_i(t)` — the model's implied daily crowding-closing flow
* `risk_measure = P(t) * d_i(t)²` — a **dynamic risk measure** (the LQR
  cost-to-go)

**Flow → contagion**: because `xbar_i(t)` is itself built from ALL peers'
states through the same graphon, a shock at one well-connected node shifts
the consensus its neighbors are pulled toward — the network channel through
which stress at one ETF can propagate to others.

This is independently derived (not a reproduction of any specific paper's
exact equations), **inspired by** active research on graphon mean-field
systems for systemic risk — specifically Amini, Cao & Sulem, *"Graphon
mean-field backward stochastic differential equations with jumps and
associated dynamic risk measures,"* Finance and Stochastics 29(4), 2025
(also SSRN 2022/2024), and the broader Carmona-Fouque-Sun-style interbank /
mutual-holding systemic-risk literature extended to heterogeneous graphon
interaction. Those papers solve a fully coupled continuous-time mean-field
equilibrium (HJB + Fokker-Planck, or a graphon-BSDE); this repo uses a
tractable discrete-time LQR simplification in the same spirit — a quadratic
tracking-vs-control tradeoff with a network-mediated target — made
implementable and testable on daily ETF data within a GitHub Actions
free-tier CPU budget.

## What's validated, concretely

* **The Riccati recursion is independently derived and checked against
  brute-force dynamic programming**: a 401-point state grid with explicit
  backward Bellman induction matched the closed-form optimal control to
  within grid resolution (max abs error 0.01) at every horizon step and
  every tested state value.
* **The mean-reversion-to-consensus fitting (`kappa`) recovers a known
  planted value**: a synthetic series built with `kappa=0.15` was recovered
  as a median estimate of 0.145 by the rolling-window fitting procedure.
* **The baseline ablation was checked with a clean target-only signal
  injection** (features computed from the untouched price series; only the
  forecast TARGET was made to depend on a graphon feature) after an earlier
  engine's injection design — which modified the return series itself and
  let the baseline pick up the same signal through the back door — gave a
  false negative. Here: dIC t≈3.8 (selection) and t≈2.5 (holdout),
  correctly attributed to the graphon machinery.
* **The contagion-lead diagnostic's chance baseline is an empirical
  circular-shift permutation, not an analytic formula** — carried over
  directly from the fix already made in the sibling hyperbolic-flows engine
  (an analytic "uniform random placement" formula was found there to be
  biased on fully independent data, because deduplicated onsets/events are
  quasi-periodic, not uniformly scattered).

## Asset-class-neutral ranking (added after the first real run)

The first real run's COMBINED universe (43 tickers spanning fixed income,
real estate, commodities and several equity groupings) produced a
"High"-confidence result that, on inspection, looked more like a single-theme
momentum call than genuine cross-sectional skill: the picks (SLV, GDX, URA)
were all precious-metals/mining ETFs and *identical* to the baseline
model's three picks, the baseline features' total weight magnitude was about
2x the graphon features' (0.243 vs 0.125), and only 36% of COMBINED's own
grid cells showed the graphon features beating baseline on holdout (the
selected cell's incremental advantage was t=0.80 even though the model's
overall IC was t=2.10 — most of the significance was the baseline's). The
hypothesis: a universe mixing fundamentally different asset classes lets a
model *look* like it's ranking ETFs while really just learning "commodities
were strong this period."

`graphon_model.cs_zscore_neutral` tests that directly: it de-means BOTH the
features and the forward-return target *within* each hand-labeled asset class
(`config.ASSET_CLASS_MAP`) before cross-sectional ranking, so the regression
can only see — and the top-3 can only reflect — skill at ranking *within* a
class, not at overweighting whichever class trended. It scales by the overall
post-demean std rather than a per-class std (a small class would make its own
std unstable). Both modes (`NEUTRALIZE_OPTIONS = [False, True]`) are run and
reported side by side in the same grid, so selection and the dashboard can
show directly whether an edge survives neutralization.

Validated on a constructed example: with one class given a constant +5
offset, plain `cs_zscore` left the class-level difference visible (class
means at +0.95 / -0.95), while `cs_zscore_neutral` removed it entirely (both
~0) and still preserved within-class ranking (0.77 correlation with the true
within-class signal).

**Cost, stated plainly**: this doubles the grid (36 → 72 configurations per
universe), so the multiple-comparisons risk the README already warns about
goes up with it — an additional reason to read the selected row's holdout
(and the baseline comparison) rather than any single cell in the grid.

**Also worth knowing**: the neutralization applies the SAME treatment to every
universe, not just COMBINED. FI_COMMODITIES and EQUITY_SECTORS each contain
more than one asset-class label too (fixed income + real estate +
commodities; broad equity + sector equity), so those results get tested for
the same confound as a side effect, not only COMBINED.

## What it does and does not claim

* `kappa_i(t)` and the graphon itself are fit from recent price correlations,
  not causal ownership or crowding data — this is a correlation-based proxy
  for "network interaction," not a measurement of actual shared positioning.
* The LQR control problem is a deliberate simplification for tractability,
  not a literal implementation of any cited paper's coupled system.
* The forecast is a small linear model on the gap/control/risk-measure
  readout, always compared against a baseline with **none of this**, so the
  graphon machinery is credited only for what it adds.

## How the result is kept honest

Same checks as every sibling engine, applied from the start:

1. **Baseline ablation.** Every configuration runs twice: the graphon model
   (`x, mom5, mom21, vol21` + `gap, gap_chg, kappa, control, risk_measure,
   network_stress`), and a baseline using only the first four. Credit is the
   *difference*.
2. **Cross-sectional target.** Features and forward returns are z-scored
   across ETFs each day, so drift/beta cancel.
3. **The product is what is tested**: mean forward return of the top-3 minus
   the equal-weight universe, after cost, plus cross-sectional rank-IC.
4. **Non-overlapping test points** spaced one horizon apart; Sharpe
   annualized by `sqrt(252/horizon)`.
5. **Selection/holdout split** with a sample-size-aware selection gate (an
   absolute AND relative period-count floor) and confidence **capped at Low
   whenever that gate didn't pass** — both carried over directly from the
   sibling delay-agent-dynamics / hyperbolic-flows engines after real-data
   runs there showed what goes wrong without them.

## The "contagion" claim is tested, not assumed

`graphon_model.contagion_lead_diagnostic` compares a **global systemic risk
onset** (the graphon-weighted mean of every ETF's own dynamic risk measure
crosses above a z-score of 1.5 — the crossing moment, not the whole elevated
stretch) against a **large aggregate move** (`|`10-day equal-weight universe
return`|` exceeds 2x 21-day realized vol, scaled), both de-duplicated to one
per 21 quiet days, with an empirical circular-shift permutation chance
baseline.

## Dashboard readouts (Tab 1)

* **The network**: ETFs on a circle, edges are the strongest correlation
  ties, color is each ETF's crowding gap, size is its dynamic risk measure.
* **Decoupling ETFs**: how many currently have `kappa < 0` — self-
  reinforcing away from the consensus rather than reverting to it.
* **Highest risk measure**: the ETFs with the largest current cost-to-go.

## Repo structure

```
config.py          universes, graphon/LQR parameters, CORR_WINDOWS/HORIZONS/
                   WINDOWS grid, LEAD_PARAMS
data_manager.py     master parquet from HF (same as sibling repos)
graphon_model.py     graphon construction, consensus/gap, kappa fitting, Riccati
                   recursion, features, O(1) window ridge fits, walk-forward
                   evaluation, contagion-lead diagnostic
trainer.py           per-universe orchestration, selection/holdout, live picks, JSON
push_results.py      HfApi.upload_file to the results dataset
us_calendar.py        trading-day helpers
streamlit_app.py      Tab 1 live picks + network map; Tab 2 backtest grid, curves,
                   contagion-lead diagnostic
.github/workflows/daily_run.yml   weekday cron after the US close
```

## Running

```bash
pip install -r requirements.txt
python trainer.py                 # writes graphon_results_YYYY-MM-DD.json, pushes to HF
streamlit run streamlit_app.py
```

Set `HF_TOKEN`; create the dataset repo `P2SAMAPA/p2-etf-graphon-mean-field-results`
or change `config.RESULTS_REPO`. To stop the live configuration re-selecting
daily, set `config.PINNED_CONFIG = {"corr_window": 63, "horizon": 5,
"window": 756}`.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.

* **Riccati recursion validated against brute-force DP** (see above).
* **Kappa fitting validated against a known planted value** (see above).
* **Baseline ablation validated with a clean target-only injection** (see
  above) — the earlier false-negative mistake is documented so it isn't
  repeated.
* **Contagion-lead diagnostic validated on pure noise**: hit rates tracked
  chance level within a few points in both directions across all 9
  universe/window cells (no systematic bias, unlike the bug already caught
  and fixed on the sibling hyperbolic-flows engine).
* **Full three-universe run** at realistic size (2,500 days, up to 43 ETFs,
  72 grid cells: 3 correlation windows × 3 horizons × 4 windows × 2
  neutralize modes) takes about 30 seconds — no iterative optimization anywhere in the live pipeline (the
  Riccati recursion is solved once per run, not per day; the graphon and
  ridge regression are both closed-form), so this is the fastest engine in
  the suite so far.

Nothing here says it works on real markets. The first real run is the
actual test; read the holdout numbers and baseline comparison before the
picks. Realistic rank-IC for a daily cross-sectional signal is small (order
0.01–0.05); with 36 configurations searched, one strong-looking cell is
expected by chance. This is not financial advice.
