"""
map_component.py
Enhanced interactive map for BESCOM EV Grid Intelligence.
Uses Folium + streamlit-folium for a fully interactive,
realistic map with transformer nodes, zone rings, and
connecting lines between infrastructure.

Used by app.py on the Infrastructure Planner page.
"""

import folium
from folium import plugins
import pandas as pd

# ── Status colours ──────────────────────────────────────────
STATUS_COLOR   = {"Critical": "#EF4444", "Warning": "#F59E0B", "Normal": "#22C55E"}
STATUS_PULSE   = {"Critical": True,      "Warning": True,      "Normal": False}
ZONE_COLORS    = {
    "Koramangala":     "#EF4444",
    "HSR Layout":      "#F59E0B",
    "Whitefield":      "#3B82F6",
    "Electronic City": "#10B981",
    "Hebbal":          "#8B5CF6",
    "Indiranagar":     "#EC4899",
}
ZONE_CENTERS = {
    "Koramangala":     (12.935, 77.624),
    "HSR Layout":      (12.911, 77.641),
    "Whitefield":      (12.969, 77.749),
    "Electronic City": (12.839, 77.677),
    "Hebbal":          (13.035, 77.597),
    "Indiranagar":     (12.978, 77.641),
}

# Approximate transformer coordinates
# (offset slightly from zone centre so they don't stack)
TRANSFORMER_COORDS = {
    "KR-TR1":  (12.9380, 77.6290),
    "KR-TR2":  (12.9310, 77.6180),
    "HSR-TR1": (12.9150, 77.6460),
    "HSR-TR2": (12.9060, 77.6390),
    "WF-TR1":  (12.9740, 77.7540),
    "WF-TR2":  (12.9620, 77.7420),
    "WF-TR3":  (12.9800, 77.7630),
    "EC-TR1":  (12.8460, 77.6810),
    "EC-TR2":  (12.8320, 77.6710),
    "HB-TR1":  (13.0380, 77.5990),
    "HB-TR2":  (13.0290, 77.6060),
    "IND-TR1": (12.9800, 77.6440),
    "IND-TR2": (12.9740, 77.6380),
}


def _transformer_icon_html(tr_id: str, util_pct: float, status: str) -> str:
    """Custom HTML marker for a transformer node — inspired by aviation node style."""
    color  = STATUS_COLOR.get(status, "#888")
    pulse  = STATUS_PULSE.get(status, False)
    pulse_css = f"""
        @keyframes pulse {{
            0%   {{ box-shadow: 0 0 0 0 {color}88; }}
            70%  {{ box-shadow: 0 0 0 10px {color}00; }}
            100% {{ box-shadow: 0 0 0 0 {color}00; }}
        }}
        animation: pulse 1.6s infinite;
    """ if pulse else ""

    return f"""
    <div style="
        position: relative; width: 44px; height: 44px;
        display: flex; align-items: center; justify-content: center;
    ">
      <!-- outer ring -->
      <div style="
        position: absolute; width: 44px; height: 44px;
        border-radius: 50%; border: 2px solid {color};
        background: {color}22; {pulse_css}
      "></div>
      <!-- inner node -->
      <div style="
        width: 26px; height: 26px; border-radius: 50%;
        background: #0F172A; border: 2px solid {color};
        display: flex; align-items: center; justify-content: center;
        z-index: 10;
      ">
        <!-- lightning bolt icon -->
        <svg width="12" height="14" viewBox="0 0 12 14" fill="none">
          <path d="M7 1L1 8h5l-1 5 6-7H6l1-5z"
                fill="{color}" stroke="{color}" stroke-width="0.5"/>
        </svg>
      </div>
      <!-- util badge -->
      <div style="
        position: absolute; bottom: -4px; left: 50%;
        transform: translateX(-50%);
        background: {color}; color: #0F172A;
        font-size: 8px; font-weight: 800;
        border-radius: 4px; padding: 1px 4px;
        white-space: nowrap; z-index: 11;
      ">{util_pct:.0f}%</div>
    </div>
    """


def _zone_icon_html(zone: str, rank: int, score: float) -> str:
    """Zone centre marker — large coloured ring with rank badge."""
    color = ZONE_COLORS.get(zone, "#888")
    return f"""
    <div style="position:relative;width:54px;height:54px;
                display:flex;align-items:center;justify-content:center;">
      <div style="position:absolute;width:54px;height:54px;border-radius:50%;
                  border:2px dashed {color}88;background:{color}11;"></div>
      <div style="width:34px;height:34px;border-radius:50%;
                  background:{color}33;border:2px solid {color};
                  display:flex;align-items:center;justify-content:center;
                  font-size:13px;font-weight:800;color:{color};">
        #{rank}
      </div>
      <div style="position:absolute;top:-6px;right:-2px;
                  background:{color};color:#0F172A;
                  font-size:8px;font-weight:800;
                  border-radius:4px;padding:1px 4px;">
        {score:.0f}
      </div>
    </div>
    """


def _station_icon_html(stype: str) -> str:
    color = "#22C55E" if stype == "public" else "#64748B"
    sym   = "▲" if stype == "public" else "■"
    return f"""
    <div style="width:16px;height:16px;border-radius:3px;
                background:{color}33;border:1.5px solid {color};
                display:flex;align-items:center;justify-content:center;
                font-size:9px;color:{color}">{sym}</div>
    """


def _recommended_icon_html(priority_rank: int) -> str:
    medals = {1:"🥇",2:"🥈",3:"🥉"}
    m = medals.get(priority_rank, "⭐")
    return f"""
    <div style="font-size:20px;
                filter:drop-shadow(0 0 4px #FBBF24);">{m}</div>
    """


def build_map(
    transformers: pd.DataFrame,
    stations: pd.DataFrame,
    locations: pd.DataFrame,
    priority: pd.DataFrame,
    height: int = 520,
) -> folium.Map:
    """
    Build and return a Folium map with:
    - Realistic OpenStreetMap tiles + CartoDB dark tiles (toggle)
    - Zone boundary rings (coloured, clickable)
    - Transformer nodes (pulsing if critical/warning)
    - Connecting lines: transformer → zone centre
    - Existing charging stations
    - Recommended new locations (star markers)
    - Full-screen button, mini-map, layer control
    """

    m = folium.Map(
        location=[12.960, 77.660],
        zoom_start=11,
        tiles=None,          # We add tiles manually for toggle
        prefer_canvas=True,
    )

    # ── Tile layers ──────────────────────────────────────────
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="🌑 Dark (default)",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="🗺️ Street Map",
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="🛰️ Satellite",
        max_zoom=19,
    ).add_to(m)

    # ── Feature groups (toggleable layers) ──────────────────
    fg_zones    = folium.FeatureGroup(name="🔵 Zones",               show=True)
    fg_trs      = folium.FeatureGroup(name="⚡ Transformers",         show=True)
    fg_lines    = folium.FeatureGroup(name="〰️ Grid Connections",     show=True)
    fg_stations = folium.FeatureGroup(name="🔌 Existing Stations",    show=True)
    fg_recs     = folium.FeatureGroup(name="⭐ Recommended Locations", show=True)

    # ── Zone boundary rings + centre markers ─────────────────
    for zone, (lat, lng) in ZONE_CENTERS.items():
        color    = ZONE_COLORS.get(zone, "#888")
        pr_row   = priority[priority["zone"] == zone]
        rank     = int(pr_row["priority_rank"].values[0]) if len(pr_row) else 6
        score    = float(pr_row["composite_score"].values[0]) if len(pr_row) else 0
        tier     = str(pr_row["priority_tier"].values[0]) if len(pr_row) else ""
        ds       = float(pr_row["demand_score"].values[0]) if len(pr_row) else 0
        cs       = float(pr_row["coverage_gap_score"].values[0]) if len(pr_row) else 0
        gs       = float(pr_row["grid_headroom_score"].values[0]) if len(pr_row) else 0

        # Outer translucent circle (zone boundary)
        folium.Circle(
            location=[lat, lng],
            radius=1400,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.06,
            dash_array="8 4",
            tooltip=zone,
        ).add_to(fg_zones)

        # Zone centre marker
        folium.Marker(
            location=[lat, lng],
            icon=folium.DivIcon(
                html=_zone_icon_html(zone, rank, score),
                icon_size=(54, 54),
                icon_anchor=(27, 27),
            ),
            popup=folium.Popup(f"""
            <div style='font-family:sans-serif;min-width:200px;background:#1E293B;
                        color:#F1F5F9;padding:12px;border-radius:8px;
                        border-left:4px solid {color}'>
              <b style='color:{color};font-size:14px'>{zone}</b><br>
              <span style='color:#94A3B8;font-size:11px'>{tier}</span><br><br>
              <b>Priority Score: {score:.0f}</b><br>
              Demand growth:  {ds:.0f}/100<br>
              Coverage gap:   {cs:.0f}/100<br>
              Grid headroom:  {gs:.0f}/100<br>
              <br><span style='color:{color}'>Rank #{rank} of 6 zones</span>
            </div>""", max_width=240),
            tooltip=f"{zone} — Score {score:.0f}",
        ).add_to(fg_zones)

    fg_zones.add_to(m)

    # ── Transformer nodes ─────────────────────────────────────
    zone_to_trs = {}   # used later for connection lines

    for _, tr in transformers.iterrows():
        tr_id  = tr["transformer_id"]
        zone   = tr["zone"]
        util   = float(tr["utilization_pct"])
        status = str(tr["status"])
        rated  = float(tr["rated_capacity_kw"])
        peak   = float(tr["current_peak_load_kw"])
        head   = float(tr["headroom_kw"])
        color  = STATUS_COLOR.get(status, "#888")

        coords = TRANSFORMER_COORDS.get(tr_id)
        if not coords:
            continue

        zone_to_trs.setdefault(zone, []).append(coords)

        folium.Marker(
            location=list(coords),
            icon=folium.DivIcon(
                html=_transformer_icon_html(tr_id, util, status),
                icon_size=(44, 50),
                icon_anchor=(22, 38),
            ),
            popup=folium.Popup(f"""
            <div style='font-family:sans-serif;min-width:210px;background:#1E293B;
                        color:#F1F5F9;padding:12px;border-radius:8px;
                        border-left:4px solid {color}'>
              <b style='color:{color};font-size:14px'>⚡ {tr_id}</b><br>
              <span style='color:#94A3B8;font-size:11px'>{zone}</span><br><br>
              <b style='color:{color}'>Status: {status}</b><br>
              Rated capacity:  {rated:.0f} kW<br>
              Current peak:    {peak:.0f} kW<br>
              Utilisation:     <b style='color:{color}'>{util:.1f}%</b><br>
              Headroom left:   {head:.0f} kW<br>
              <br>
              <div style='background:{color}22;border-radius:4px;padding:6px;
                           font-size:11px;color:{color}'>
                {"🔴 Immediate action — delay EV charging" if status=="Critical"
                  else "🟡 Monitor and schedule off-peak" if status=="Warning"
                  else "✅ Operating normally"}
              </div>
            </div>""", max_width=250),
            tooltip=f"{tr_id} ({zone}) — {util:.0f}% {status}",
        ).add_to(fg_trs)

    fg_trs.add_to(m)

    # ── Grid connection lines: transformer → zone centre ─────
    for zone, tr_coords_list in zone_to_trs.items():
        zone_center = ZONE_CENTERS.get(zone)
        if not zone_center:
            continue
        color = ZONE_COLORS.get(zone, "#888")
        for tr_coord in tr_coords_list:
            folium.PolyLine(
                locations=[zone_center, tr_coord],
                color=color,
                weight=1.5,
                opacity=0.45,
                dash_array="6 4",
                tooltip=f"{zone} grid connection",
            ).add_to(fg_lines)

    fg_lines.add_to(m)

    # ── Existing charging stations ────────────────────────────
    for _, s in stations.iterrows():
        stype   = str(s.get("type","public"))
        sid     = str(s["station_id"])
        zone    = str(s["zone"])
        chargers= int(s["chargers_count"])
        sessions= int(s["avg_daily_sessions"])
        util_s  = float(s.get("avg_utilization_pct", 0))
        ctype   = str(s.get("charger_type","AC"))
        color   = "#22C55E" if stype == "public" else "#64748B"

        folium.Marker(
            location=[float(s["lat"]), float(s["lng"])],
            icon=folium.DivIcon(
                html=_station_icon_html(stype),
                icon_size=(16, 16),
                icon_anchor=(8, 8),
            ),
            popup=folium.Popup(f"""
            <div style='font-family:sans-serif;min-width:190px;background:#1E293B;
                        color:#F1F5F9;padding:10px;border-radius:8px;
                        border-left:4px solid {color}'>
              <b style='color:{color}'>🔌 {sid}</b><br>
              <span style='color:#94A3B8;font-size:11px'>{zone}</span><br><br>
              Type:           {stype.capitalize()}<br>
              Charger type:   {ctype}<br>
              Chargers:       {chargers}<br>
              Daily sessions: {sessions}<br>
              Utilisation:    <b>{util_s:.0f}%</b>
            </div>""", max_width=220),
            tooltip=f"{sid} — {chargers} chargers ({stype})",
        ).add_to(fg_stations)

    fg_stations.add_to(m)

    # ── Recommended new locations ─────────────────────────────
    for _, loc in locations.iterrows():
        zone     = str(loc["zone"])
        name     = str(loc["location_name"])
        rationale= str(loc.get("rationale",""))
        ctype    = str(loc.get("recommended_charger_type",""))
        score    = float(loc.get("composite_score",0))
        headroom = float(loc.get("grid_headroom_available_kw",0))
        sessions = int(loc.get("estimated_daily_sessions",0))
        pr_rank  = int(loc.get("zone_priority_rank",6))
        color    = ZONE_COLORS.get(zone,"#FBBF24")

        folium.Marker(
            location=[float(loc["lat"]), float(loc["lng"])],
            icon=folium.DivIcon(
                html=_recommended_icon_html(pr_rank),
                icon_size=(24, 24),
                icon_anchor=(12, 20),
            ),
            popup=folium.Popup(f"""
            <div style='font-family:sans-serif;min-width:220px;background:#1E293B;
                        color:#F1F5F9;padding:12px;border-radius:8px;
                        border-left:4px solid {color}'>
              <b style='color:{color};font-size:13px'>⭐ {name}</b><br>
              <span style='color:#94A3B8;font-size:11px'>{zone}</span><br><br>
              <i style='color:#94A3B8'>{rationale}</i><br><br>
              Charger type:   <b>{ctype}</b><br>
              Zone score:     <b>{score:.0f}</b><br>
              Grid headroom:  {headroom:.0f} kW<br>
              Est. sessions:  {sessions}/day<br>
              <br>
              <div style='background:#FBBF2422;border-radius:4px;padding:5px;
                           font-size:11px;color:#FBBF24'>
                Zone Priority #{pr_rank} — Recommended build
              </div>
            </div>""", max_width=260),
            tooltip=f"🆕 {name}",
        ).add_to(fg_recs)

    fg_recs.add_to(m)

    # ── Plugins ───────────────────────────────────────────────
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MiniMap(
        tile_layer="CartoDB dark_all",
        position="bottomright",
        width=120, height=100,
        zoom_level_offset=-6,
    ).add_to(m)
    plugins.MousePosition(position="bottomleft").add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # ── Custom legend overlay ─────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 12px; z-index: 9999;
        background: #0F172Aee; border: 1px solid #334155;
        border-radius: 10px; padding: 12px 14px;
        font-family: sans-serif; font-size: 11px; color: #94A3B8;
        min-width: 170px;
    ">
      <b style="color:#F1F5F9;font-size:12px">Map Legend</b><br><br>
      <span style="color:#EF4444">⚡</span> Transformer — Critical<br>
      <span style="color:#F59E0B">⚡</span> Transformer — Warning<br>
      <span style="color:#22C55E">⚡</span> Transformer — Normal<br>
      <span style="color:#22C55E">▲</span> Existing public station<br>
      <span style="color:#64748B">■</span> Existing private station<br>
      ⭐ Recommended new location<br>
      <span style="color:#94A3B8">- - -</span> Zone boundary<br>
      <span style="color:#94A3B8">— —</span> Grid connection<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m
