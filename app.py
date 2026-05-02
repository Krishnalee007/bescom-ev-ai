
# ============================================================
# BESCOM EV GRID INTELLIGENCE — Streamlit App (app.py)
# ============================================================
# Save this file as:  app.py
# Save in a folder called:  bescom-ev-ai/
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="BESCOM EV Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .stMetric { background: #1E293B; border-radius: 12px; padding: 16px; border: 1px solid #334155; }
    .stMetric label { color: #94A3B8 !important; font-size: 12px !important; }
    .stMetric .metric-value { color: #F1F5F9 !important; }
    div[data-testid="metric-container"] {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 12px; padding: 16px;
    }
    .alert-red   { background:#3B1515; border-left:4px solid #EF4444; padding:12px 16px; border-radius:8px; margin:8px 0; }
    .alert-amber { background:#3B2A0A; border-left:4px solid #F59E0B; padding:12px 16px; border-radius:8px; margin:8px 0; }
    .alert-green { background:#0F2E1A; border-left:4px solid #22C55E; padding:12px 16px; border-radius:8px; margin:8px 0; }
    .card        { background:#1E293B; border:1px solid #334155; border-radius:12px; padding:20px; margin:8px 0; }
    .rank-badge  { background:#2563EB; color:white; border-radius:6px; padding:2px 10px; font-weight:700; font-size:13px; }
    h1,h2,h3 { color: #F1F5F9 !important; }
    .sidebar .sidebar-content { background: #020A14; }
    [data-testid="stSidebar"] { background: #020A14; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    zone_load     = pd.read_csv("zone_load.csv")
    ev_counts     = pd.read_csv("ev_counts.csv")
    transformers  = pd.read_csv("transformers.csv")
    stations      = pd.read_csv("charging_stations.csv")
    forecast      = pd.read_csv("forecast_output.csv")
    sched_compare = pd.read_csv("schedule_comparison.csv")
    sched_grid    = pd.read_csv("schedule_grid.csv")
    recommendations = pd.read_csv("charging_recommendations.csv")
    priority      = pd.read_csv("zone_priority_scores.csv")
    locations     = pd.read_csv("location_recommendations.csv")
    return (zone_load, ev_counts, transformers, stations,
            forecast, sched_compare, sched_grid,
            recommendations, priority, locations)

try:
    (zone_load, ev_counts, transformers, stations,
     forecast, sched_compare, sched_grid,
     recommendations, priority, locations) = load_data()
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"⚠️ Could not load data files: {e}")
    st.info("Make sure all CSV files from Steps 1–4 are in the same folder as app.py")
    st.stop()

ZONES = forecast["zone"].unique().tolist()
ZONE_COLORS = {
    "Koramangala":    "#EF4444",
    "HSR Layout":     "#F59E0B",
    "Whitefield":     "#3B82F6",
    "Electronic City":"#10B981",
    "Hebbal":         "#8B5CF6",
    "Indiranagar":    "#EC4899",
}

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ BESCOM")
    st.markdown("**EV Grid Intelligence**")
    st.caption("AI Decision-Support Layer · v1.0")
    st.divider()

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📈  Demand Forecast",
        "🕐  Charging Scheduler",
        "📍  Infrastructure Planner",
    ], label_visibility="collapsed")

    st.divider()
    st.caption("Data: Synthetic · Bengaluru 2025–26")
    st.caption("Model: Gradient Boosting · 6 zones")

    # Live clock substitute
    import datetime
    now = datetime.datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f"<small style='color:#475569'>{now} IST</small>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("⚡ BESCOM EV Grid Intelligence")
    st.caption("Real-time AI decision-support for EV charging demand and infrastructure planning")
    st.divider()

    # ── KPI Row ──
    total_evs     = ev_counts[ev_counts["year"]==2026]["ev_count"].sum()
    total_chargers= stations["chargers_count"].sum()
    critical_trs  = (transformers["status"]=="Critical").sum()
    avg_peak_util = transformers["utilization_pct"].mean()
    peak_reduction= sched_compare["peak_reduction_pct"].mean()
    zones_at_risk = (sched_compare["hours_over_limit_baseline"] > 0).sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total EVs (2026)",       f"{total_evs:,}",     "+31% YoY")
    c2.metric("Charging Stations",      f"{len(stations)}",   "Across 6 zones")
    c3.metric("Total Chargers",         f"{total_chargers}",  "Public + Private")
    c4.metric("Critical Transformers",  f"{critical_trs}",    delta="⚠️ Immediate action", delta_color="inverse")
    c5.metric("Avg Peak Utilisation",   f"{avg_peak_util:.1f}%", "Grid-wide")
    c6.metric("Scheduling Benefit",     f"{peak_reduction:.1f}%", "Peak load reduction")

    st.divider()
    col_l, col_r = st.columns([3, 2])

    # ── Active Alerts ──
    with col_l:
        st.subheader("🚨 Active Grid Alerts")
        crit = transformers[transformers["status"]=="Critical"]
        warn = transformers[transformers["status"]=="Warning"]
        for _, r in crit.iterrows():
            st.markdown(f"""<div class="alert-red">
                <b>🔴 CRITICAL — {r['transformer_id']} ({r['zone']})</b><br>
                Load at <b>{r['utilization_pct']}%</b> of rated capacity ({r['current_peak_load_kw']:.0f} / {r['rated_capacity_kw']} kW).
                Immediate scheduling intervention recommended.
            </div>""", unsafe_allow_html=True)
        for _, r in warn.iterrows():
            st.markdown(f"""<div class="alert-amber">
                <b>🟡 WARNING — {r['transformer_id']} ({r['zone']})</b><br>
                Load at <b>{r['utilization_pct']}%</b> of rated capacity.
                Monitor and pre-emptively schedule off-peak charging.
            </div>""", unsafe_allow_html=True)
        if len(crit) == 0 and len(warn) == 0:
            st.markdown('<div class="alert-green">✅ All transformers operating normally</div>',
                        unsafe_allow_html=True)

    # ── Zone Health Table ──
    with col_r:
        st.subheader("Zone Health Summary")
        for zone in ZONES:
            zt  = transformers[transformers["zone"]==zone]
            avg = zt["utilization_pct"].mean()
            evs = ev_counts[(ev_counts["zone"]==zone)&(ev_counts["year"]==2026)]["ev_count"].max()
            sc  = sched_compare[sched_compare["zone"]==zone]["peak_reduction_pct"].values[0]
            color = "#EF4444" if avg >= 85 else "#F59E0B" if avg >= 70 else "#22C55E"
            st.markdown(f"""
            <div style='display:flex;align-items:center;justify-content:space-between;
                         padding:8px 12px;background:#1E293B;border-radius:8px;margin:4px 0;
                         border-left:3px solid {color}'>
              <span style='color:#F1F5F9;font-weight:600'>{zone}</span>
              <span style='color:{color};font-weight:700'>{avg:.0f}%</span>
              <span style='color:#94A3B8;font-size:12px'>{evs:,} EVs</span>
              <span style='color:#22C55E;font-size:12px'>↓{sc:.0f}% if scheduled</span>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── EV Growth Chart ──
    st.subheader("📈 EV Adoption Growth — All Zones (2023–2026)")
    ev_plot = ev_counts.copy()
    ev_plot["date"] = pd.to_datetime(
        ev_plot["year"].astype(str) + "-" + ev_plot["month"].astype(str) + "-01")
    fig_ev = go.Figure()
    for zone in ZONES:
        zd = ev_plot[ev_plot["zone"]==zone]
        fig_ev.add_trace(go.Scatter(
            x=zd["date"], y=zd["ev_count"],
            name=zone, line=dict(color=ZONE_COLORS[zone], width=2),
            fill="tozeroy", fillcolor=ZONE_COLORS[zone].replace(")", ",0.05)").replace("rgb","rgba"),
        ))
    fig_ev.update_layout(
        plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
        font_color="#94A3B8", height=320,
        xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig_ev, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════
elif "Forecast" in page:
    st.title("📈 EV Charging Demand Forecast")
    st.caption("48-hour AI predictions per zone · Gradient Boosting model")
    st.divider()

    zone_sel = st.selectbox("Select Zone", ZONES)
    zf = forecast[forecast["zone"]==zone_sel].copy()
    zf["dt"] = pd.to_datetime(zf["forecast_datetime"])

    # Risk banner
    peak = zf["predicted_load_kw"].max()
    tr_cap = transformers[transformers["zone"]==zone_sel]["rated_capacity_kw"].max()
    util_pct = peak / tr_cap * 100
    if util_pct >= 90:
        st.markdown(f'<div class="alert-red">🔴 <b>CRITICAL</b> — Predicted peak of <b>{peak:.0f} kW</b> will hit {util_pct:.0f}% of transformer capacity. Immediate scheduling action needed.</div>', unsafe_allow_html=True)
    elif util_pct >= 80:
        st.markdown(f'<div class="alert-amber">🟡 <b>WARNING</b> — Predicted peak of <b>{peak:.0f} kW</b> approaches {util_pct:.0f}% capacity. Pre-emptive scheduling recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-green">✅ Load forecast within safe limits for {zone_sel}.</div>', unsafe_allow_html=True)

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Predicted Peak",   f"{peak:.0f} kW",      f"{util_pct:.0f}% of capacity")
    c2.metric("Transformer Cap",  f"{tr_cap:.0f} kW",    zone_sel)
    c3.metric("Peak Risk Hours",  f"{(zf['is_peak_risk']==1).sum()}h",  "in next 48h")
    c4.metric("Safe Hours",       f"{(zf['is_peak_risk']==0).sum()}h",  "of 48")

    st.divider()

    # ── Forecast Chart ──
    st.subheader(f"48-Hour Load Forecast — {zone_sel}")
    fig = go.Figure()
    color = ZONE_COLORS[zone_sel]

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([zf["dt"], zf["dt"][::-1]]),
        y=pd.concat([zf["upper_bound_kw"], zf["lower_bound_kw"][::-1]]),
        fill="toself", fillcolor=color.replace("#","rgba(").replace("EF","239,").replace("F5","245,")+"0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence band", showlegend=True,
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=zf["dt"], y=zf["predicted_load_kw"],
        name="Predicted load", line=dict(color=color, width=2.5, dash="dot"),
    ))
    # Capacity lines
    fig.add_hline(y=tr_cap*0.80, line_dash="dash", line_color="#F59E0B",
                  annotation_text="80% Warning", annotation_position="top right")
    fig.add_hline(y=tr_cap*0.90, line_dash="dash", line_color="#EF4444",
                  annotation_text="90% Critical", annotation_position="top right")
    # Evening shading
    for i in range(2):
        base = zf["dt"].iloc[0].replace(hour=0, minute=0) + pd.Timedelta(days=i)
        fig.add_vrect(x0=base+pd.Timedelta(hours=18), x1=base+pd.Timedelta(hours=22),
                      fillcolor="rgba(239,68,68,0.07)", line_width=0,
                      annotation_text="Evening peak" if i==0 else "",
                      annotation_position="top left")

    fig.update_layout(
        plot_bgcolor="#1E293B", paper_bgcolor="#0F172A", font_color="#94A3B8",
        height=380, xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155", title="Load (kW)"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── All Zones Side-by-side ──
    st.subheader("Peak Load — All Zones Compared")
    peak_by_zone = forecast.groupby("zone")["predicted_load_kw"].max().reset_index()
    cap_by_zone  = transformers.groupby("zone")["rated_capacity_kw"].max().reset_index()
    comp = peak_by_zone.merge(cap_by_zone, on="zone")
    comp["util_pct"] = (comp["predicted_load_kw"] / comp["rated_capacity_kw"] * 100).round(1)
    comp["color"]    = comp["zone"].map(ZONE_COLORS)
    comp = comp.sort_values("util_pct", ascending=False)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=comp["zone"], y=comp["rated_capacity_kw"],
        name="Rated capacity", marker_color="#1E3A5F", marker_line_width=0,
    ))
    fig2.add_trace(go.Bar(
        x=comp["zone"], y=comp["predicted_load_kw"],
        name="Predicted peak", marker_color=[ZONE_COLORS[z] for z in comp["zone"]],
        text=[f"{u:.0f}%" for u in comp["util_pct"]],
        textposition="outside", textfont=dict(color="#F1F5F9"),
    ))
    fig2.add_hline(y=comp["rated_capacity_kw"].mean()*0.80,
                   line_dash="dash", line_color="#F59E0B", annotation_text="Avg 80% limit")
    fig2.update_layout(
        barmode="overlay", plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
        font_color="#94A3B8", height=350,
        xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155", title="kW"),
        legend=dict(bgcolor="#1E293B"), margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Hourly heatmap
    st.subheader("Demand Heatmap — Hour × Zone")
    pivot = forecast.groupby(["zone","hour_of_day"])["predicted_load_kw"].mean().unstack()
    fig3  = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                      labels=dict(x="Hour", y="Zone", color="kW"),
                      aspect="auto")
    fig3.update_layout(
        plot_bgcolor="#1E293B", paper_bgcolor="#0F172A", font_color="#94A3B8",
        height=280, margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — CHARGING SCHEDULER
# ═══════════════════════════════════════════════════════════════
elif "Scheduler" in page:
    st.title("🕐 Smart Charging Scheduler")
    st.caption("AI-optimised load shifting · Grid-aware recommendations")
    st.divider()

    # Summary KPIs
    total_saved  = sched_compare["peak_reduction_kw"].sum()
    avg_reduc    = sched_compare["peak_reduction_pct"].mean()
    zones_fixed  = (sched_compare["hours_over_limit_optimised"]==0).sum()
    total_energy = sched_compare["total_ev_energy_kwh"].sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Peak Saved",     f"{total_saved:.0f} kW",     "Across all zones")
    c2.metric("Avg Peak Reduction",   f"{avg_reduc:.1f}%",         "vs unmanaged")
    c3.metric("Zones Fully Fixed",    f"{zones_fixed}/6",          "0 overload hours")
    c4.metric("Energy Preserved",     "100%",                       "Load shifted, not lost")

    st.divider()

    # Before / After bar chart
    st.subheader("Before vs After Scheduling — All Zones")
    fig_ba = go.Figure()
    sc = sched_compare.sort_values("peak_reduction_pct", ascending=False)
    fig_ba.add_trace(go.Bar(
        name="Before (unmanaged)", x=sc["zone"], y=sc["baseline_peak_load_kw"],
        marker_color="#EF4444", opacity=0.85,
        text=[f"{v:.0f} kW" for v in sc["baseline_peak_load_kw"]],
        textposition="outside", textfont=dict(color="#F87171"),
    ))
    fig_ba.add_trace(go.Bar(
        name="After (optimised)", x=sc["zone"], y=sc["optimised_peak_load_kw"],
        marker_color="#22C55E", opacity=0.85,
        text=[f"↓{v:.0f}%" for v in sc["peak_reduction_pct"]],
        textposition="outside", textfont=dict(color="#4ADE80"),
    ))
    fig_ba.update_layout(
        barmode="group", plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
        font_color="#94A3B8", height=350,
        yaxis=dict(title="Peak Load (kW)", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155"),
        margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig_ba, use_container_width=True)

    st.divider()

    # Per-zone recommendation cards
    st.subheader("📋 Zone Charging Recommendations")
    col1, col2 = st.columns(2)
    for i, zone in enumerate(ZONES):
        col = col1 if i % 2 == 0 else col2
        with col:
            sc_row  = sched_compare[sched_compare["zone"]==zone].iloc[0]
            recs    = recommendations[recommendations["zone"]==zone].sort_values("recommendation_rank")
            color   = ZONE_COLORS[zone]
            risk    = "🔴 HIGH RISK" if sc_row["hours_over_limit_baseline"]>3 else \
                      "🟡 MODERATE"  if sc_row["hours_over_limit_baseline"]>0 else "🟢 SAFE"

            best_rec = recs.iloc[0] if len(recs) > 0 else None

            st.markdown(f"""
            <div class="card" style="border-left: 4px solid {color}">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <b style="color:{color};font-size:15px">{zone}</b>
                <span style="font-size:12px">{risk}</span>
              </div>
              <div style="color:#94A3B8;font-size:12px;margin-bottom:8px">
                ❌ Avoid: <b style="color:#F87171">18:00 – 22:00</b>
                &nbsp;|&nbsp;
                Peak reduction: <b style="color:#4ADE80">{sc_row['peak_reduction_pct']:.0f}%</b>
              </div>
              {'<div style="color:#94A3B8;font-size:12px">Best window: <b style="color:#22C55E">' + best_rec["window_start"] + " – " + best_rec["window_end"] + '</b> · ' + str(best_rec["evs_can_charge_simultaneously"]) + ' EVs simultaneously</div>' if best_rec is not None else ''}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Hour-by-hour action grid for selected zone
    st.subheader("24-Hour Action Grid")
    zone_sel2 = st.selectbox("Select Zone for Detail", ZONES, key="sched_zone")
    zg = sched_grid[sched_grid["zone"]==zone_sel2].sort_values("hour")

    action_colors = {
        "DELAY":      ("#3B1515","#EF4444"),
        "CHARGE NOW": ("#0F2E1A","#22C55E"),
        "SHIFT HERE": ("#1A2A3B","#38BDF8"),
        "MONITOR":    ("#1E293B","#94A3B8"),
    }

    cols = st.columns(8)
    for i, (_, row) in enumerate(zg.iterrows()):
        bg, fg = action_colors.get(row["action"], ("#1E293B","#94A3B8"))
        cols[i % 8].markdown(f"""
        <div style="background:{bg};border:1px solid {fg};border-radius:8px;
                    padding:8px 4px;text-align:center;margin:2px">
          <div style="color:{fg};font-size:10px;font-weight:700">{row['time_label']}</div>
          <div style="color:{fg};font-size:9px;margin-top:3px">{row['action']}</div>
          <div style="color:#475569;font-size:9px">{row.get('grid_load_optimised_kw',0):.0f}kW</div>
        </div>""", unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:16px;margin-top:12px;flex-wrap:wrap">
      <span style="color:#EF4444">🔴 DELAY — Avoid charging (evening peak)</span>
      <span style="color:#22C55E">🟢 CHARGE NOW — Best window (off-peak)</span>
      <span style="color:#38BDF8">🔵 SHIFT HERE — Load moved here from peak</span>
      <span style="color:#94A3B8">⚪ MONITOR — Normal operation</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — INFRASTRUCTURE PLANNER
# ═══════════════════════════════════════════════════════════════
elif "Infrastructure" in page:
    st.title("📍 Infrastructure Location Planner")
    st.caption("AI-scored zone prioritisation · 3-factor weighted model")
    st.divider()

    # Priority ranking cards
    st.subheader("🏆 Zone Priority Ranking")
    cols = st.columns(3)
    for i, (_, row) in enumerate(priority.iterrows()):
        col = cols[i % 3]
        color = ZONE_COLORS.get(row["zone"],"#888")
        medal = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣"][i]
        with col:
            st.markdown(f"""
            <div class="card" style="border-top:3px solid {color}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:20px">{medal}</span>
                <span style="background:{color};color:white;border-radius:6px;
                             padding:2px 10px;font-size:12px;font-weight:700">
                  {row['composite_score']:.0f} pts
                </span>
              </div>
              <div style="color:#F1F5F9;font-weight:700;font-size:15px;margin:8px 0 4px">{row['zone']}</div>
              <div style="font-size:11px;color:#94A3B8">{row.get('priority_tier','—')}</div>
              <hr style="border-color:#334155;margin:10px 0">
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;font-size:11px;text-align:center">
                <div><div style="color:#3B82F6;font-weight:700">{row['demand_score']:.0f}</div><div style="color:#475569">Demand</div></div>
                <div><div style="color:#F59E0B;font-weight:700">{row['coverage_gap_score']:.0f}</div><div style="color:#475569">Coverage</div></div>
                <div><div style="color:#22C55E;font-weight:700">{row['grid_headroom_score']:.0f}</div><div style="color:#475569">Grid</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    col_l, col_r = st.columns([2,3])

    with col_l:
        # Scoring breakdown chart
        st.subheader("Score Breakdown")
        fig_score = go.Figure()
        p = priority.sort_values("composite_score")
        fig_score.add_trace(go.Bar(
            y=p["zone"], x=p["demand_score"]*0.40, name="Demand (40%)",
            orientation="h", marker_color="#3B82F6",
        ))
        fig_score.add_trace(go.Bar(
            y=p["zone"], x=p["coverage_gap_score"]*0.30, name="Coverage (30%)",
            orientation="h", marker_color="#F59E0B",
        ))
        fig_score.add_trace(go.Bar(
            y=p["zone"], x=p["grid_headroom_score"]*0.30, name="Grid (30%)",
            orientation="h", marker_color="#22C55E",
        ))
        fig_score.update_layout(
            barmode="stack", plot_bgcolor="#1E293B", paper_bgcolor="#0F172A",
            font_color="#94A3B8", height=320,
            xaxis=dict(title="Weighted Score", gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            legend=dict(bgcolor="#1E293B", orientation="h", y=-0.2),
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with col_r:
        # Scatter map
        st.subheader("Bengaluru Priority Map")
        all_lats = stations["lat"].tolist() + locations["lat"].tolist()
        all_lngs = stations["lng"].tolist() + locations["lng"].tolist()

        fig_map = go.Figure()

        # Existing stations
        fig_map.add_trace(go.Scattergeo(
            lat=stations["lat"], lon=stations["lng"],
            mode="markers",
            marker=dict(size=6, color="#22C55E", symbol="triangle-up"),
            name="Existing stations",
            hovertext=stations["station_id"],
        ))

        # Recommended locations
        fig_map.add_trace(go.Scattergeo(
            lat=locations["lat"], lon=locations["lng"],
            mode="markers+text",
            marker=dict(size=12, color="#FBBF24", symbol="star",
                        line=dict(color="#92400E", width=1)),
            name="Recommended new",
            text=locations["zone_priority_rank"].apply(lambda x: f"P{x}"),
            textfont=dict(size=8, color="white"),
            hovertext=locations["location_name"],
        ))

        # Zone bubbles
        zone_centers = {
            "Koramangala":    (12.935, 77.624), "HSR Layout":     (12.911, 77.641),
            "Whitefield":     (12.969, 77.749), "Electronic City": (12.839, 77.677),
            "Hebbal":         (13.035, 77.597), "Indiranagar":    (12.978, 77.641),
        }
        for zone, (lat, lng) in zone_centers.items():
            pr = priority[priority["zone"]==zone]["priority_rank"].values[0]
            sc = priority[priority["zone"]==zone]["composite_score"].values[0]
            fig_map.add_trace(go.Scattergeo(
                lat=[lat], lon=[lng], mode="markers",
                marker=dict(size=sc*0.6, color=ZONE_COLORS[zone], opacity=0.25),
                name=zone, showlegend=False,
                hovertext=f"{zone} — Score: {sc:.0f}",
            ))

        fig_map.update_layout(
            geo=dict(
                scope="asia",
                center=dict(lat=12.97, lon=77.67),
                projection_scale=150,
                showland=True, landcolor="#1E293B",
                showocean=True, oceancolor="#0F172A",
                showcountries=True, countrycolor="#334155",
                showlakes=False,
                bgcolor="#0F172A",
            ),
            paper_bgcolor="#0F172A", font_color="#94A3B8",
            height=340, margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(bgcolor="#1E293B", bordercolor="#334155", x=0, y=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    # Recommended locations table
    st.subheader("📋 Specific Location Recommendations")
    zone_filter = st.selectbox("Filter by Zone", ["All Zones"] + ZONES, key="infra_zone")
    show_locs = locations if zone_filter == "All Zones" else locations[locations["zone"]==zone_filter]

    for _, row in show_locs.iterrows():
        color = ZONE_COLORS.get(row["zone"],"#888")
        charger_badge = {
            "DC Fast": "🔵", "AC+DC": "🟡", "AC Slow": "🟢"
        }.get(row["recommended_charger_type"],"⚪")
        st.markdown(f"""
        <div class="card" style="border-left:4px solid {color}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <span style="color:{color};font-weight:700">📍 {row['location_name']}</span>
              <span style="background:#1E3A5F;color:#93C5FD;border-radius:4px;
                           padding:1px 8px;font-size:11px;margin-left:8px">{row['zone']}</span>
            </div>
            <span style="font-size:12px">{charger_badge} {row['recommended_charger_type']}</span>
          </div>
          <div style="color:#94A3B8;font-size:12px;margin-top:6px">
            {row['rationale']}
          </div>
          <div style="display:flex;gap:20px;margin-top:8px;font-size:11px;color:#475569">
            <span>🌐 {row['lat']:.4f}, {row['lng']:.4f}</span>
            <span>📊 Zone score: <b style="color:#F1F5F9">{row['composite_score']:.0f}</b></span>
            <span>⚡ Headroom: <b style="color:#22C55E">{row['grid_headroom_available_kw']:.0f} kW</b></span>
            <span>🔋 Est. {row['estimated_daily_sessions']} sessions/day</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("📐 Baseline Comparison")
    st.caption("AI scoring vs uniform placement (2 stations per zone regardless of need)")
    comp_data = {
        "Zone": ZONES,
        "Uniform Placement": [2]*6,
        "AI Recommended": priority.sort_values("zone").set_index("zone").loc[ZONES, "priority_rank"].apply(lambda r: 4-min(r,3)).tolist(),
        "Priority Score": priority.sort_values("zone").set_index("zone").loc[ZONES, "composite_score"].tolist(),
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df.style.background_gradient(subset=["Priority Score"], cmap="RdYlGn"),
                 use_container_width=True, hide_index=True)
