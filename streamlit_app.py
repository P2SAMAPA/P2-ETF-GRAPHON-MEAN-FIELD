import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 Graphon-Mean-Field", page_icon="🕸️", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#7c2d12"
POSITIVE = "#16a34a"
NEUTRAL = "#d97706"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
CONF_COLORS = {"high": POSITIVE, "medium": NEUTRAL, "low": SUBTLE}

CSS = """
<style>
    .stApp { background-color: __PAGE_BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUBTLE__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .universe-heading { font-size: 1.15rem; font-weight: 700; color: __INK__; margin: 0 0 0.15rem 0; }
    .pick-card { background: #ffffff; border: 1px solid __BORDER__; border-left: 4px solid var(--accent);
        border-radius: 12px; padding: 1.0rem 1.2rem; margin: 0.35rem 0; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
    .pick-ticker { font-size: 1.05rem; font-weight: 700; color: __INK__; letter-spacing: 0.02em; }
    .pick-return { font-size: 1.8rem; font-weight: 800; color: __INK__; margin: 0.2rem 0 0.1rem 0; line-height: 1.1; }
    .pick-sub { font-size: 0.72rem; color: __SUBTLE__; margin-bottom: 0.4rem; }
    .pick-badge { display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 0.15rem 0.55rem; border-radius: 999px; color: white; background: var(--accent); }
    .kpi-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: __INK__; }
    .kpi-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }
    .banner { background: linear-gradient(90deg, #fff7ed 0%, #fffbeb 100%); border: 1px solid #fed7aa;
        border-left: 4px solid __PRIMARY__; border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.88rem;
        color: #431407; margin: 0.5rem 0 0.9rem 0; }
    .banner-warn { background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid __NEUTRAL__;
        border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.88rem; color: #78350f; margin: 0.5rem 0 0.9rem 0; }
    .note { font-size: 0.82rem; color: __SUBTLE__; margin: 0.4rem 0 0.9rem 0; }
    .readout { background: #ffffff; border: 1px solid __BORDER__; border-radius: 10px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; }
    .readout-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; }
    .readout-value { font-size: 1.15rem; font-weight: 800; color: __INK__; }
    .readout-sub { font-size: 0.78rem; color: __SUBTLE__; }
    div[data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 10px; overflow: hidden; }
</style>
"""
CSS = (CSS.replace("__PAGE_BG__", PAGE_BG).replace("__INK__", INK).replace("__SUBTLE__", SUBTLE)
       .replace("__BORDER__", CARD_BORDER).replace("__PRIMARY__", PRIMARY).replace("__NEUTRAL__", NEUTRAL))
st.markdown(CSS, unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-graphon-mean-field-results"


def _find_latest_hf_result(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("graphon_results_") and f.endswith(".json"))
        if not names:
            return None, "No graphon_results_*.json files found in " + repo_id + "."
        return names[-1], None
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key=""):
    notes = []
    local = glob.glob("graphon_results_*.json")
    if local:
        latest = sorted(local)[-1]
        try:
            with open(latest, "r") as f:
                return json.load(f), ["Loaded local file: " + latest]
        except Exception as e:
            notes.append("Found local file " + latest + " but couldn't parse it: " + str(e))
    name, err = _find_latest_hf_result(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            url = "https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(resp.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def conf_color(c):
    return CONF_COLORS.get((c or "low").lower(), SUBTLE)


def fmt_bps(x):
    return "n/a" if x is None else ("%+.1f" % (x * 1e4))


def render_pick_cards(picks):
    if not picks:
        st.info("No picks available for this selection.")
        return
    horizon = picks[0].get("horizon_days")
    cols = st.columns(min(len(picks), 3))
    for i, p in enumerate(picks):
        color = conf_color(p["confidence"])
        html = ('<div class="pick-card" style="--accent: ' + color + ';">'
                '<div class="pick-ticker">' + p["ticker"] + '</div>'
                '<div class="pick-return">' + ("%+.2f" % p["expected_return"]) + '%</div>'
                '<div class="pick-sub">expected ' + str(horizon) + '-day return vs universe average</div>'
                '<span class="pick-badge">' + p["confidence"] + ' confidence</span></div>')
        with cols[i % len(cols)]:
            st.markdown(html, unsafe_allow_html=True)


def kpi(label, value):
    return ('<div class="kpi-card"><div class="kpi-value">' + value + '</div>'
            '<div class="kpi-label">' + label + '</div></div>')


def readout(label, value, sub=""):
    return ('<div class="readout"><div class="readout-label">' + label + '</div>'
            '<div class="readout-value">' + value + '</div><div class="readout-sub">' + sub + '</div></div>')


def render_network(snapshot, edges, pick_tickers, key):
    if not snapshot:
        st.info("No snapshot available.")
        return
    df = pd.DataFrame(snapshot)
    df = df[np.isfinite(df["gap"]) & np.isfinite(df["risk_measure"])]
    if df.empty:
        st.info("No valid readout today.")
        return
    angle = {t: 2 * np.pi * i / len(df) for i, t in enumerate(df["ticker"])}
    xs = {t: np.cos(a) for t, a in angle.items()}
    ys = {t: np.sin(a) for t, a in angle.items()}
    fig = go.Figure()
    for e in (edges or []):
        if e["a"] in xs and e["b"] in xs:
            fig.add_trace(go.Scatter(x=[xs[e["a"]], xs[e["b"]]], y=[ys[e["a"]], ys[e["b"]]], mode="lines",
                                     line=dict(color="rgba(124,45,18,0.25)", width=max(1, e["weight"] * 4)),
                                     hoverinfo="skip", showlegend=False))
    is_pick = df["ticker"].isin(pick_tickers)
    risk_max = max(df["risk_measure"].max(), 1e-9)
    sizes = 9 + 20 * (df["risk_measure"] / risk_max)
    colors = df["gap"]
    fig.add_trace(go.Scatter(
        x=[xs[t] for t in df.loc[~is_pick, "ticker"]], y=[ys[t] for t in df.loc[~is_pick, "ticker"]],
        mode="markers+text", text=df.loc[~is_pick, "ticker"], textposition="top center",
        textfont=dict(size=9, color=SUBTLE),
        marker=dict(size=sizes[~is_pick], color=colors[~is_pick], colorscale="RdBu_r", cmid=0,
                   line=dict(width=1, color="#111827"), colorbar=dict(title="gap", thickness=12)),
        name="ETFs", hovertemplate="%{text}<extra></extra>"))
    sub = df[is_pick]
    if not sub.empty:
        fig.add_trace(go.Scatter(
            x=[xs[t] for t in sub["ticker"]], y=[ys[t] for t in sub["ticker"]], mode="markers+text",
            text=sub["ticker"], textposition="top center", textfont=dict(size=11, color=INK, family="Arial Black"),
            marker=dict(size=sizes[is_pick] + 4, color="#4ade80", symbol="diamond", line=dict(width=2, color="#111827")),
            name="top picks", hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(range=[-1.3, 1.3], showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(range=[-1.3, 1.3], showgrid=False, zeroline=False, showticklabels=False,
                                scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="net_" + key)
    st.caption("The graphon network today: lines are the strongest correlation-based ties. Dot color is each "
               "ETF's crowding gap (how far it has drifted from its network-weighted peer consensus — blue above, "
               "red below); dot size is its dynamic risk measure (the LQR cost-to-go). Layout is illustrative "
               "(evenly spaced), not a force-directed graph.")


def render_readouts(snapshot, n_decoupling, n_covered):
    if not snapshot:
        return
    st.markdown(readout("Decoupling ETFs", (str(n_decoupling) if n_decoupling is not None else "n/a"),
                        ("of " + str(n_covered) + " ETFs currently self-reinforcing away from their peer "
                         "consensus (kappa < 0) rather than mean-reverting back to it") if n_covered else ""),
               unsafe_allow_html=True)
    df = pd.DataFrame(snapshot)
    if "risk_measure" in df:
        top = df.sort_values("risk_measure", ascending=False).head(5)
        rows = [{"ETF": r["ticker"], "Risk measure": round(r["risk_measure"], 4), "Gap": round(r["gap"], 3),
                "Kappa": round(r["kappa"], 3) if r.get("kappa") is not None else None} for _, r in top.iterrows()]
        st.markdown('<div class="readout-label" style="margin-top:0.2rem;">Highest dynamic risk measure today</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_curves(curves, key):
    if not curves or not curves.get("dates"):
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["graphon_cum"], mode="lines", name="Graphon model top-N",
                             line=dict(color=PRIMARY, width=2.2)))
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["baseline_cum"], mode="lines", name="Baseline (no graphon features)",
                             line=dict(color="#94a3b8", width=2, dash="dot")))
    si = curves.get("split_index", 0)
    if 0 < si < len(curves["dates"]):
        fig.add_vline(x=curves["dates"][si], line_dash="dash", line_color="#334155")
        fig.add_annotation(x=curves["dates"][si], y=1, yref="paper", text="holdout starts", showarrow=False,
                           xanchor="left", font=dict(size=11, color="#334155"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Cumulative net excess return vs universe (%)", gridcolor="#eef2f7"),
                      xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="curves_" + key)


def render_series(pattern, key):
    s = (pattern or {}).get("series")
    if not s or not s.get("dates"):
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["dates"], y=s["global_risk_index"], mode="lines", name="Global systemic risk index",
                             line=dict(color=PRIMARY, width=1.8)))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Mean dynamic risk measure", gridcolor="#eef2f7"),
                      xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="series_" + key)


def grid_frame(grid, selected):
    rows = []
    for g in grid:
        s, h = g["selection"], g["holdout"]
        rows.append({
            "Corr window (d)": g["corr_window"], "Horizon (d)": g["horizon"], "Window (d)": g["window"],
            "Class-neutral": g.get("neutralize", False),
            "Sel. IC t": s.get("ic_t"), "Sel. net bps": (s.get("net_spread_mean") or 0) * 1e4,
            "Hold n": h.get("n"), "Hold IC": h.get("ic_mean"), "Hold IC t": h.get("ic_t"),
            "Hold IC vs baseline": h.get("d_ic_mean"),
            "Hold net bps": (h.get("net_spread_mean") or 0) * 1e4,
            "Hold net bps vs baseline": (h.get("d_spread_mean") or 0) * 1e4,
            "_sel": (g["corr_window"] == selected["corr_window"] and g["horizon"] == selected["horizon"]
                    and g["window"] == selected["window"] and g.get("neutralize", False) == selected.get("neutralize", False)),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Class-neutral", "Corr window (d)", "Horizon (d)", "Window (d)"]).reset_index(drop=True)


def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0
    head, refresh = st.columns([5, 1])
    with head:
        st.markdown('<div class="app-title">🕸️ P2 Graphon-Mean-Field</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">A graphon mean-field model of ETF crowding, flow and contagion. '
                    'Experimental: credited only for what it adds beyond a no-graphon baseline.</div>',
                    unsafe_allow_html=True)
    with refresh:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()

    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
            st.write("Run `python trainer.py` to create a local `graphon_results_*.json`, or check the HF results dataset and HF_TOKEN.")
        return

    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "**")
    if notes:
        st.caption("📡 " + notes[0])

    tab1, tab2 = st.tabs(["🔮 Live Signal", "🧪 Diagnostics & Backtest"])

    with tab1:
        st.markdown('<div class="note">Each ETF is pulled toward a correlation-network-weighted average of its '
                    'peers (the graphon consensus). The gap between an ETF and that consensus implies an optimal '
                    '"liquidation" flow and a dynamic risk measure, solved in closed form. A ridge model ranks '
                    'ETFs by forward return relative to the universe average, using that readout alongside plain '
                    'momentum/vol features.</div>', unsafe_allow_html=True)
        for uni, picks in data.get("top_picks", {}).items():
            sel = data["selected"][uni]
            h = sel["holdout"]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            cls = "banner" if sel["confidence"] != "Low" else "banner-warn"
            neut_tag = ('<b>asset-class-neutral</b>' if sel.get("neutralize")
                       else '<b>not</b> asset-class-neutral')
            txt = ('Selected: correlation window <b>' + str(sel["corr_window"]) + 'd</b>, <b>'
                   + str(sel["horizon"]) + '-day</b> horizon, <b>' + str(sel["window"]) + 'd</b> training window, '
                   + neut_tag
                   + (' (pinned)' if sel.get("pinned") else '')
                   + '<br/>Holdout (' + str(h.get("n", 0)) + ' non-overlapping periods, never used for selection): rank-IC <b>'
                   + ("%+.3f" % h.get("ic_mean", 0)) + '</b> (t=' + ("%+.2f" % h.get("ic_t", 0)) + ') vs no-graphon baseline <b>'
                   + ("%+.3f" % h.get("ic_base_mean", 0)) + '</b> &nbsp;|&nbsp; net top-N excess <b>'
                   + fmt_bps(h.get("net_spread_mean")) + ' bps</b>/period (baseline '
                   + fmt_bps(h.get("base_net_spread_mean")) + ')')
            if sel["confidence"] == "Low":
                txt += ('<br/><b>Not validated:</b> on the holdout the graphon features do not clearly add value '
                        'over the baseline, so treat these picks as unproven.')
            if not sel.get("gate_passed"):
                txt += '<br/>No configuration was profitable on the selection segment; this is the best of a weak set.'
            st.markdown('<div class="' + cls + '">' + txt + '</div>', unsafe_allow_html=True)

            render_pick_cards(picks)

            pattern = data["patterns"].get(uni, {}).get(str(sel["corr_window"]))
            colm, colr = st.columns([3, 2])
            with colm:
                render_network((pattern or {}).get("snapshot"), (pattern or {}).get("top_edges"),
                               [p["ticker"] for p in picks], key="live_" + uni)
            with colr:
                render_readouts((pattern or {}).get("snapshot"), (pattern or {}).get("n_decoupling"),
                               (pattern or {}).get("n_covered"))
            base = data.get("baseline_picks", {}).get(uni, [])
            if base:
                overlap = len(set(p["ticker"] for p in picks) & set(p["ticker"] for p in base))
                with st.expander("Baseline (no graphon features) picks for comparison"):
                    st.write("Baseline top picks: " + ", ".join(p["ticker"] for p in base)
                             + " — overlap with the graphon-model picks: " + str(overlap) + " of " + str(len(picks)))
                    fw = data.get("feature_weights", {}).get(uni)
                    if fw:
                        st.write("Fitted feature weights (standardized), graphon model:")
                        st.dataframe(pd.DataFrame([fw]).T.rename(columns={0: "weight"}), use_container_width=True)
            st.markdown("<hr style='margin: 1.4rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab2:
        for uni, grid in data.get("grid", {}).items():
            sel = data["selected"][uni]
            h = sel["holdout"]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            kp = [("Holdout rank-IC", "%+.3f" % h.get("ic_mean", 0)),
                  ("IC t-stat", "%+.2f" % h.get("ic_t", 0)),
                  ("Net excess bps/period", fmt_bps(h.get("net_spread_mean"))),
                  ("IC vs baseline", "%+.3f" % h.get("d_ic_mean", 0)),
                  ("Holdout periods", str(h.get("n", 0)))]
            for c, (lab, val) in zip(cols, kp):
                with c:
                    st.markdown(kpi(lab, val), unsafe_allow_html=True)
            st.markdown('<div class="note">Excess = mean forward return of the top-N picks minus the equal-weight '
                        'universe average, after a ' + str(data.get('settings', {}).get('trading_cost_bps', 15))
                        + ' bps round-trip cost on the fraction of picks replaced. Test points are spaced one '
                        'horizon apart, so periods never overlap. IC = cross-sectional rank correlation of forecast '
                        'and realized return. "vs baseline" is the difference to a model with no graphon features '
                        'on the same dates.</div>', unsafe_allow_html=True)

            st.markdown("###### Cumulative net excess return of the selected configuration")
            render_curves(data.get("curves", {}).get(uni), key=uni)

            df = grid_frame(grid, sel)
            sel_mask = df["_sel"].values
            view = df.drop(columns=["_sel"])
            st.markdown("###### All configurations (selection segment chose; holdout is the honest number)")
            st.caption("Testing " + str(len(view)) + " configurations means the best-looking one on any single "
                       "column can be luck. Only the selected row's holdout is used for the confidence label. "
                       "'Class-neutral' rows de-mean both the features and the forward-return target within each "
                       "hand-labeled asset class before ranking, so they can only reflect skill at picking within "
                       "a class, not at overweighting whichever class trended — added after a universe mixing "
                       "several asset classes produced a 'High' result whose picks and feature weights looked more "
                       "like a single-sector momentum call than genuine cross-sectional skill; see README.")
            styled = view.style.apply(lambda r: ["background-color: #ffedd5" if sel_mask[r.name] else "" for _ in r], axis=1)
            styled = styled.format({"Sel. IC t": "{:.2f}", "Sel. net bps": "{:+.1f}", "Hold IC": "{:+.3f}",
                                    "Hold IC t": "{:+.2f}", "Hold IC vs baseline": "{:+.3f}",
                                    "Hold net bps": "{:+.1f}", "Hold net bps vs baseline": "{:+.1f}"}, na_rep="n/a")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            pats = data["patterns"].get(uni, {})
            st.markdown("###### Global systemic risk index over time (selected window)")
            render_series(pats.get(str(sel["corr_window"])), key=uni)

            lead_rows = []
            for cw, pp in pats.items():
                rl = pp.get("contagion_lead") or {}
                if rl.get("available") and rl.get("n_large_move_events"):
                    lead_rows.append({
                        "Corr window": int(cw), "Risk onsets": rl["n_risk_onsets"],
                        "Large-move events": rl["n_large_move_events"],
                        "Onset in prior 20d": rl["hit_rate"], "Chance level": rl["chance_hit_rate"],
                        "Median lead (d)": rl["median_lead_days"], "False alarm rate": rl["false_alarm_rate"],
                        "Chance false alarm": rl["chance_false_alarm_rate"]})
            if lead_rows:
                st.markdown("###### Does a systemic risk spike lead a large aggregate move?")
                st.caption("Triggers fixed in advance: risk onset = global systemic risk index z-score crosses "
                           "above 1.5; large-move event = an equal-weight universe move of at least 2x 21-day "
                           "realized vol over 10 days. Chance level is an empirical circular-shift permutation "
                           "(not an analytic formula — validated to avoid overstating chance when onsets/events "
                           "are regularly spaced by deduplication; see README). A real early-warning effect needs "
                           "'onset in prior 20d' clearly above 'chance level' and a false-alarm rate below its "
                           "own chance level.")
                st.dataframe(pd.DataFrame(lead_rows).style.format(
                    {"Onset in prior 20d": "{:.0%}", "Chance level": "{:.0%}", "Median lead (d)": "{:.0f}",
                     "False alarm rate": "{:.0%}", "Chance false alarm": "{:.0%}"}, na_rep="n/a"),
                    use_container_width=True, hide_index=True)

            sp = data.get("cw_picks", {}).get(uni, {})
            if sp:
                st.markdown("###### Best configuration per correlation window")
                stabs = st.tabs([cw + "d" for cw in sp.keys()])
                for tb, (cw, info) in zip(stabs, sp.items()):
                    with tb:
                        c = info["config"]
                        neut = "class-neutral" if c.get("neutralize") else "not class-neutral"
                        st.caption(str(c["horizon"]) + "-day horizon, " + str(c["window"]) + "d window, " + neut
                                   + ", confidence " + info["confidence"] + ", holdout IC "
                                   + ("%+.3f" % info["holdout"].get("ic_mean", 0)))
                        render_pick_cards(info["picks"])
            st.markdown("<hr style='margin: 1.6rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
