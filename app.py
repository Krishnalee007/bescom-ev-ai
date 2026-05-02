import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="BESCOM EV Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
div[data-testid="stMetricValue"] > div { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    plot_bgcolor="#1E293B", paper_bgcolor="#0F172A", font_color="#94A3B8",
    xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1),
    margin=dict(l=10, r=10, t=30, b=10),
)

ZONE_COLORS = {
    "Koramangala": "#EF4444", "HSR Layout": "#F59E0B",
    "Whitefield": "#3B82F6",  "Electronic City": "#10B981",
    "Hebbal": "#8B5CF6",      "Indiranagar": "#EC4899",
}
ZONES = ["Koramangala","HSR Layout","Whitefield","Electronic City","Hebbal","Indiranagar"]

@st.cache_data
def load():
    return (
        pd.read_csv("zone_load.csv"), pd.read_csv("ev_counts.csv"),
        pd.read_csv("transformers.csv"), pd.read_csv("charging_stations.csv"),
        pd.read_csv("forecast_output.csv"), pd.read_csv("schedule_comparison.csv"),
        pd.read_csv("schedule_grid.csv"), pd.read_csv("charging_recommendations.csv"),
        pd.read_csv("zone_priority_scores.csv"), pd.read_csv("location_recommendations.csv"),
    )

try:
    (zone_load, ev_counts, transformers, stations, forecast,
     sched_compare, sched_grid, recommendations, priority, locations) = load()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.info("Make sure all CSV files from Steps 1–4 are in the same folder as app.py")
    st.stop()

with st.sidebar:
    st.markdown("## ⚡ BESCOM")
    st.markdown("**EV Grid Intelligence**")
    st.caption("AI Decision-Support Layer · v1.0")
    st.divider()
    page = st.radio("", [
        "🏠  Overview", "📈  Demand Forecast",
        "🕐  Charging Scheduler", "📍  Infrastructure Planner",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Synthetic data · Bengaluru 2025–26")
    st.caption("Model: Gradient Boosting · 6 zones")
    import datetime
    st.caption(datetime.datetime.now().strftime("%d %b %Y · %H:%M IST"))


# ═══════════════════════════ PAGE 1 — OVERVIEW ═══════════════════════════
if "Overview" in page:
    st.title("⚡ BESCOM EV Grid Intelligence")
    st.caption("AI decision-support for EV charging demand and infrastructure planning · Bengaluru")
    st.divider()

    total_evs     = int(ev_counts[ev_counts["year"]==2026]["ev_count"].sum())
    critical_trs  = int((transformers["utilization_pct"] >= 90).sum())
    avg_util      = transformers["utilization_pct"].mean()
    avg_reduction = sched_compare["peak_reduction_pct"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total EVs (2026)",      f"{total_evs:,}",       "↑ 31% YoY")
    c2.metric("Charging Stations",     f"{len(stations)}",     "Across 6 zones")
    c3.metric("Total Chargers",        f"{stations['chargers_count'].sum()}", "Public + Private")
    c4.metric("Critical Transformers", f"{critical_trs}",      "≥ 90% utilisation", delta_color="inverse")
    c5.metric("Avg Grid Utilisation",  f"{avg_util:.1f}%",     "At evening peak")
    c6.metric("Scheduling Benefit",    f"{avg_reduction:.1f}%","Peak load reduction")

    st.divider()
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("🚨 Active Grid Alerts")
        crit = transformers[transformers["utilization_pct"] >= 90].sort_values("utilization_pct", ascending=False)
        warn = transformers[transformers["utilization_pct"].between(75, 90)]
        if len(crit)==0 and len(warn)==0:
            st.success("✅ All transformers operating within safe limits.")
        for _, r in crit.head(3).iterrows():
            st.error(f"🔴 **CRITICAL — {r['transformer_id']} ({r['zone']})**  \n"
                     f"Load at **{r['utilization_pct']:.0f}%** of {r['rated_capacity_kw']:.0f} kW. "
                     f"Immediate scheduling intervention recommended.")
        for _, r in warn.head(2).iterrows():
            st.warning(f"🟡 **WARNING — {r['transformer_id']} ({r['zone']})**  \n"
                       f"Load at **{r['utilization_pct']:.0f}%** capacity. "
                       f"Pre-emptive off-peak scheduling recommended.")

    with col_b:
        st.subheader("Zone Health")
        tr_z  = transformers.groupby("zone")["utilization_pct"].mean()
        ev_z  = ev_counts[ev_counts["year"]==2026].set_index("zone")["ev_count"]
        sc_z  = sched_compare.set_index("zone")["peak_reduction_pct"]
        for zone in ZONES:
            util  = tr_z.get(zone, 0)
            evs   = int(ev_z.get(zone, 0))
            reduc = sc_z.get(zone, 0)
            color = "#EF4444" if util>=85 else "#F59E0B" if util>=70 else "#22C55E"
            icon  = "🔴" if util>=85 else "🟡" if util>=70 else "🟢"
            st.markdown(
                f"{icon} **{zone}** — "
                f"<span style='color:{color};font-weight:700'>{util:.0f}%</span> · "
                f"{evs:,} EVs · "
                f"<span style='color:#22C55E'>↓{reduc:.0f}% if scheduled</span>",
                unsafe_allow_html=True)
            st.progress(min(int(util), 100))

    st.divider()
    st.subheader("📈 EV Adoption Growth — All Zones (2023–2026)")
    ev_plot = ev_counts.copy()
    ev_plot["date"] = pd.to_datetime(
        ev_plot["year"].astype(str) + "-" + ev_plot["month"].astype(str).str.zfill(2) + "-01")
    fig_ev = go.Figure()
    for zone in ZONES:
        zd = ev_plot[ev_plot["zone"]==zone].sort_values("date")
        fig_ev.add_trace(go.Scatter(
            x=zd["date"], y=zd["ev_count"], name=zone,
            line=dict(color=ZONE_COLORS[zone], width=2)))
    fig_ev.update_layout(height=320, **PLOT_LAYOUT)
    st.plotly_chart(fig_ev, use_container_width=True)


# ═══════════════════════ PAGE 2 — DEMAND FORECAST ═══════════════════════
elif "Forecast" in page:
    st.title("📈 EV Charging Demand Forecast")
    st.caption("48-hour AI predictions · Gradient Boosting model · Per zone")
    st.divider()

    zone_sel = st.selectbox("Select Zone", ZONES)
    zf = forecast[forecast["zone"]==zone_sel].copy()
    zf["dt"] = pd.to_datetime(zf["forecast_datetime"])
    tr_cap = float(transformers[transformers["zone"]==zone_sel]["rated_capacity_kw"].max())
    peak   = float(zf["predicted_load_kw"].max())
    util   = peak / tr_cap * 100

    if util >= 90:   st.error(f"🔴 **CRITICAL** — Peak {peak:.0f} kW = {util:.0f}% of capacity. Act now.")
    elif util >= 80: st.warning(f"🟡 **WARNING** — Peak {peak:.0f} kW = {util:.0f}% capacity. Schedule off-peak.")
    else:            st.success(f"✅ Load forecast safe for {zone_sel}.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Predicted Peak",  f"{peak:.0f} kW", f"{util:.0f}% of capacity")
    c2.metric("Transformer Cap", f"{tr_cap:.0f} kW", zone_sel)
    c3.metric("Peak Risk Hours", f"{int((zf['is_peak_risk']==1).sum())}h", "in next 48h")
    c4.metric("Safe Hours",      f"{int((zf['is_peak_risk']==0).sum())}h", "of 48")

    st.divider()
    st.subheader(f"48-Hour Load Forecast — {zone_sel}")
    color = ZONE_COLORS[zone_sel]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(zf["dt"])+list(zf["dt"][::-1]),
        y=list(zf["upper_bound_kw"])+list(zf["lower_bound_kw"][::-1]),
        fill="toself", fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence band"))
    fig.add_trace(go.Scatter(
        x=zf["dt"], y=zf["predicted_load_kw"], name="Predicted load",
        line=dict(color=color, width=2.5, dash="dot")))
    fig.add_hline(y=tr_cap*0.80, line_dash="dash", line_color="#F59E0B",
                  annotation_text="80% warning", annotation_font_color="#F59E0B")
    fig.add_hline(y=tr_cap*0.90, line_dash="dash", line_color="#EF4444",
                  annotation_text="90% critical", annotation_font_color="#EF4444")
    fig.update_layout(height=380, yaxis_title="Load (kW)", xaxis_title="Date / Hour", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("All Zones — Peak Utilisation")
        pz = forecast.groupby("zone")["predicted_load_kw"].max().reset_index()
        cz = transformers.groupby("zone")["rated_capacity_kw"].max().reset_index()
        comp = pz.merge(cz, on="zone")
        comp["util_pct"] = (comp["predicted_load_kw"]/comp["rated_capacity_kw"]*100).round(1)
        comp = comp.sort_values("util_pct", ascending=True)
        fig2 = go.Figure(go.Bar(
            y=comp["zone"], x=comp["util_pct"], orientation="h",
            marker_color=[ZONE_COLORS[z] for z in comp["zone"]],
            text=[f"{u:.0f}%" for u in comp["util_pct"]], textposition="outside"))
        fig2.add_vline(x=80, line_dash="dash", line_color="#F59E0B")
        fig2.add_vline(x=90, line_dash="dash", line_color="#EF4444")
        fig2.update_layout(height=300, xaxis_title="Peak utilisation %", xaxis_range=[0,115], **PLOT_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    with col_r:
        st.subheader("Demand Heatmap — Hour × Zone")
        pivot = forecast.groupby(["zone","hour_of_day"])["predicted_load_kw"].mean().unstack()
        pivot = pivot.reindex(ZONES)
        fig3 = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                         labels=dict(x="Hour", y="Zone", color="kW"), aspect="auto")
        fig3.update_layout(height=300, **PLOT_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════ PAGE 3 — CHARGING SCHEDULER ══════════════════════
elif "Scheduler" in page:
    st.title("🕐 Smart Charging Scheduler")
    st.caption("AI-optimised load shifting · Grid-aware · 100% energy preserved")
    st.divider()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Peak Saved",   f"{sched_compare['peak_reduction_kw'].sum():.0f} kW", "All zones")
    c2.metric("Avg Peak Reduction", f"{sched_compare['peak_reduction_pct'].mean():.1f}%", "vs unmanaged")
    c3.metric("Zones Fully Fixed",  f"{int((sched_compare['hours_over_limit_optimised']==0).sum())} / 6", "0 overload hrs")
    c4.metric("Energy Preserved",   "100%", "Shifted, not removed")

    st.divider()
    st.subheader("Before vs After — Peak Load All Zones")
    sc = sched_compare.sort_values("peak_reduction_pct", ascending=False)
    fig_ba = go.Figure()
    fig_ba.add_trace(go.Bar(name="❌ Unmanaged", x=sc["zone"], y=sc["baseline_peak_load_kw"],
        marker_color="#EF4444", opacity=0.85,
        text=[f"{v:.0f} kW" for v in sc["baseline_peak_load_kw"]], textposition="outside"))
    fig_ba.add_trace(go.Bar(name="✅ Optimised", x=sc["zone"], y=sc["optimised_peak_load_kw"],
        marker_color="#22C55E", opacity=0.85,
        text=[f"↓{v:.0f}%" for v in sc["peak_reduction_pct"]], textposition="outside"))
    fig_ba.update_layout(barmode="group", height=360, yaxis_title="Peak Load (kW)", **PLOT_LAYOUT)
    st.plotly_chart(fig_ba, use_container_width=True)

    st.divider()
    st.subheader("Zone Recommendations")
    col1, col2 = st.columns(2)
    for i, zone in enumerate(ZONES):
        col = col1 if i%2==0 else col2
        sc_row = sched_compare[sched_compare["zone"]==zone].iloc[0]
        recs   = recommendations[recommendations["zone"]==zone].sort_values("recommendation_rank")
        color  = ZONE_COLORS[zone]
        over   = int(sc_row["hours_over_limit_baseline"])
        risk   = "🔴 HIGH RISK" if over>3 else "🟡 MODERATE" if over>0 else "🟢 SAFE"
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<span style='color:{color};font-size:16px;font-weight:700'>{zone}</span>"
                    f"&nbsp;&nbsp;<span style='font-size:12px'>{risk}</span>",
                    unsafe_allow_html=True)
                st.caption(f"❌ Avoid: **18:00–22:00**  ·  Peak saved: **{sc_row['peak_reduction_pct']:.0f}%**")
                if len(recs) > 0:
                    r = recs.iloc[0]
                    st.success(f"✅ Best window: **{r['window_start']} – {r['window_end']}**"
                               f"  ({r['evs_can_charge_simultaneously']} EVs simultaneously)")

    st.divider()
    st.subheader("24-Hour Action Grid")
    zone_sel2 = st.selectbox("Select Zone", ZONES, key="sched_z")
    zg = sched_grid[sched_grid["zone"]==zone_sel2].sort_values("hour")
    ACTION = {"DELAY":("🔴","#EF4444"),"CHARGE NOW":("🟢","#22C55E"),
              "SHIFT HERE":("🔵","#38BDF8"),"MONITOR":("⚪","#64748B")}
    for row_start in [0,6,12,18]:
        cols = st.columns(6)
        for j,(_,hr) in enumerate(zg[zg["hour"].between(row_start,row_start+5)].iterrows()):
            icon, clr = ACTION.get(hr["action"],("⚪","#64748B"))
            cols[j].markdown(
                f"<div style='text-align:center;background:#1E293B;border:1px solid {clr};"
                f"border-radius:8px;padding:8px 2px;margin:2px'>"
                f"<div style='color:{clr};font-size:10px;font-weight:700'>{hr['time_label']}</div>"
                f"<div style='font-size:14px'>{icon}</div>"
                f"<div style='color:{clr};font-size:8px'>{hr['action']}</div>"
                f"<div style='color:#475569;font-size:8px'>{hr.get('grid_load_optimised_kw',0):.0f}kW</div></div>",
                unsafe_allow_html=True)
    st.caption("🔴 DELAY  ·  🟢 CHARGE NOW  ·  🔵 SHIFT HERE  ·  ⚪ MONITOR")


# ════════════════════ PAGE 4 — INFRASTRUCTURE PLANNER ════════════════════
elif "Infrastructure" in page:
    st.title("📍 Infrastructure Location Planner")
    st.caption("3-factor AI scoring · Demand 40% · Coverage Gap 30% · Grid Headroom 30%")
    st.divider()

    st.subheader("🏆 Zone Priority Ranking")
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣"]
    cols = st.columns(3)
    for i, (_, row) in enumerate(priority.iterrows()):
        color = ZONE_COLORS.get(row["zone"], "#888")
        with cols[i%3]:
            with st.container(border=True):
                st.markdown(
                    f"{medals[i]}&nbsp;"
                    f"<span style='color:{color};font-size:15px;font-weight:700'>{row['zone']}</span>"
                    f"&nbsp;<span style='background:{color};color:white;border-radius:5px;"
                    f"padding:2px 8px;font-size:12px'>{row['composite_score']:.0f} pts</span>",
                    unsafe_allow_html=True)
                st.caption(row.get("priority_tier",""))
                a,b,c = st.columns(3)
                a.metric("Demand",   f"{row['demand_score']:.0f}")
                b.metric("Coverage", f"{row['coverage_gap_score']:.0f}")
                c.metric("Grid",     f"{row['grid_headroom_score']:.0f}")

    st.divider()
    col_ch, col_map = st.columns([2,3])

    with col_ch:
        st.subheader("Score Breakdown")
        p = priority.sort_values("composite_score")
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(y=p["zone"], x=(p["demand_score"]*0.40).round(1),
            name="Demand (40%)", orientation="h", marker_color="#3B82F6"))
        fig_sc.add_trace(go.Bar(y=p["zone"], x=(p["coverage_gap_score"]*0.30).round(1),
            name="Coverage (30%)", orientation="h", marker_color="#F59E0B"))
        fig_sc.add_trace(go.Bar(y=p["zone"], x=(p["grid_headroom_score"]*0.30).round(1),
            name="Grid (30%)", orientation="h", marker_color="#22C55E"))
        fig_sc.update_layout(barmode="stack", height=320, xaxis_title="Weighted Score",
            legend=dict(orientation="h", y=-0.25, bgcolor="#0F172A"), **PLOT_LAYOUT)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_map:
        st.subheader("Bengaluru Priority Map")
        ZONE_CENTERS = {
            "Koramangala":(12.935,77.624), "HSR Layout":(12.911,77.641),
            "Whitefield":(12.969,77.749),  "Electronic City":(12.839,77.677),
            "Hebbal":(13.035,77.597),      "Indiranagar":(12.978,77.641),
        }
        fig_map = go.Figure()
        for zone,(lat,lng) in ZONE_CENTERS.items():
            pr_row = priority[priority["zone"]==zone]
            sc   = float(pr_row["composite_score"].values[0]) if len(pr_row) else 50
            rank = int(pr_row["priority_rank"].values[0])     if len(pr_row) else 6
            fig_map.add_trace(go.Scattermapbox(
                lat=[lat], lon=[lng], mode="markers+text",
                marker=dict(size=sc*0.55, color=ZONE_COLORS[zone], opacity=0.7),
                text=[f"#{rank}"], textfont=dict(size=11, color="white"),
                name=zone, hovertext=f"{zone} — {sc:.0f} pts", hoverinfo="text"))
        fig_map.add_trace(go.Scattermapbox(
            lat=stations["lat"], lon=stations["lng"], mode="markers",
            marker=dict(size=7, color="#22C55E"), name="Existing stations",
            hovertext=stations["station_id"], hoverinfo="text"))
        fig_map.add_trace(go.Scattermapbox(
            lat=locations["lat"], lon=locations["lng"], mode="markers",
            marker=dict(size=13, color="#FBBF24", symbol="star"),
            name="⭐ Recommended", hovertext=locations["location_name"], hoverinfo="text"))
        fig_map.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=12.97, lon=77.67), zoom=10.5),
            height=330, paper_bgcolor="#0F172A", font_color="#94A3B8",
            legend=dict(bgcolor="#1E293B", bordercolor="#334155", x=0, y=1, font=dict(size=10)),
            margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.subheader("📋 Specific Location Recommendations")
    zone_filter = st.selectbox("Filter by Zone", ["All Zones"]+ZONES)
    show_locs = locations if zone_filter=="All Zones" else locations[locations["zone"]==zone_filter]
    for _, row in show_locs.sort_values("zone_priority_rank").iterrows():
        color = ZONE_COLORS.get(row["zone"],"#888")
        icons = {"DC Fast":"🔵","AC+DC":"🟡","AC Slow":"🟢"}
        ctype = str(row.get("recommended_charger_type",""))
        with st.container(border=True):
            cl, cr = st.columns([4,1])
            with cl:
                st.markdown(f"<span style='color:{color};font-weight:700'>📍 {row['location_name']}</span>",
                            unsafe_allow_html=True)
                st.caption(row.get("rationale",""))
            with cr:
                st.markdown(f"{icons.get(ctype,'⚪')} **{ctype}**  \n"
                            f"Score: **{row['composite_score']:.0f}**")
            st.caption(f"📌 {row['lat']:.4f}, {row['lng']:.4f}  ·  "
                       f"⚡ {row['grid_headroom_available_kw']:.0f} kW headroom  ·  "
                       f"~{row['estimated_daily_sessions']} sessions/day")

    st.divider()
    st.subheader("📐 AI vs Uniform Placement Baseline")
    comp_rows = []
    for zone in ZONES:
        pr = priority[priority["zone"]==zone]
        sc = float(pr["composite_score"].values[0]) if len(pr) else 50
        ai = 3 if sc>=65 else 2 if sc>=45 else 1
        comp_rows.append({"Zone":zone,"Uniform (2 per zone)":2,"AI Recommended":ai,
                          "Priority Score":f"{sc:.0f}","Difference":f"{ai-2:+d}"})
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
