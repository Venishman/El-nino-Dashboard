"""
AI-Powered El Niño Early Warning & Global Climate Impact Dashboard
===================================================================
REAL MODE ONLY: every number comes from live NOAA / Open-Meteo APIs.
No demo data, no mock data. If an API fails, an error is shown instead.

Install :  pip install "streamlit>=1.37" pandas numpy requests plotly scikit-learn pyserial
Run     :  streamlit run el_nino_dashboard.py        (use a TERMINAL, not IDLE)

What updates every minute?
  - LIVE PANEL: re-downloads sea-surface temperature (SST) for a grid of points
    inside the Niño 3.4 box and CALCULATES the box-average SST + anomaly.
What updates daily / monthly?
  - NOAA ONI + Niño indices (NOAA publishes monthly), local climate (ERA5).
"""

import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Must be the first Streamlit command
st.set_page_config(page_title="El Niño Early Warning Dashboard", page_icon="🌊", layout="wide")

# ============================================================================
# CONFIG
# ============================================================================
REFRESH_SECONDS = 60

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Niño 3.4 region = 5°N-5°S, 170°W-120°W. We sample a 5x5 grid inside it.
BOX = dict(lat_min=-5, lat_max=5, lon_min=-170, lon_max=-120)
GRID_LATS = [-4, -2, 0, 2, 4]
GRID_LONS = [-165, -155, -145, -135, -125]

CLIM_START, CLIM_END = 1991, 2020  # climatology base period

PLACES = {
    "Chennai / Tambaram, India": (12.92, 80.12),
    "Mumbai, India": (19.07, 72.88),
    "Delhi, India": (28.61, 77.21),
    "Jakarta, Indonesia": (-6.20, 106.85),
    "Lima, Peru": (-12.05, -77.04),
    "Sydney, Australia": (-33.87, 151.21),
    "Nairobi, Kenya": (-1.29, 36.82),
    "Custom lat/lon": None,
}

SEASONS = {
    "All year": list(range(1, 13)),
    "Jun-Sep (SW monsoon)": [6, 7, 8, 9],
    "Oct-Dec (NE monsoon)": [10, 11, 12],
    "Dec-Feb": [12, 1, 2],
    "Mar-May": [3, 4, 5],
}

PHASE_COLOR = {"El Niño": "#e5484d", "La Niña": "#3b82f6", "Neutral": "#8b8b8b"}

FEATS = [
    "oni_l0", "oni_l1", "oni_l2", "oni_l3", "oni_l6", "oni_l12",
    "n34_l0", "n34_l1", "n34_l2", "m_sin", "m_cos",
]

# ============================================================================
# SMALL HELPERS
# ============================================================================


def phase(v: float) -> str:
    """NOAA definition: ONI >= +0.5 El Niño, <= -0.5 La Niña."""
    if v >= 0.5:
        return "El Niño"
    if v <= -0.5:
        return "La Niña"
    return "Neutral"


def strength(v: float) -> str:
    a = abs(v)
    if a < 0.5:
        return "-"
    if a < 1.0:
        return "Weak"
    if a < 1.5:
        return "Moderate"
    if a < 2.0:
        return "Strong"
    return "Very strong"


def clim_at(ts: pd.Timestamp, clim: pd.Series) -> float:
    """Daily-interpolated climatology from monthly means (month value = mid-month)."""
    x = ts.month + (ts.day - 0.5) / ts.days_in_month - 0.5
    lo = int(np.floor(x))
    w = x - lo
    lo_m = (lo - 1) % 12 + 1
    hi_m = lo % 12 + 1
    return float(clim[lo_m] * (1 - w) + clim[hi_m] * w)


# ============================================================================
# DATA: NOAA (monthly, cached)
# ============================================================================


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading NOAA ONI...")
def load_oni() -> pd.DataFrame:
    df = pd.read_csv(ONI_URL, sep=r"\s+")  # SEAS YR TOTAL ANOM
    seasons = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
               "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    month_of = {s: i + 1 for i, s in enumerate(seasons)}  # season centred on middle month
    df["month"] = df["SEAS"].map(month_of)
    df["date"] = pd.to_datetime(dict(year=df["YR"], month=df["month"], day=1))
    return df.rename(columns={"ANOM": "oni"})[["date", "oni"]].sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading NOAA Niño indices...")
def load_nino() -> pd.DataFrame:
    raw = pd.read_csv(
        NINO_URL, sep=r"\s+", skiprows=1, header=None,
        names=["yr", "mon", "n12", "n12a", "n3", "n3a", "n4", "n4a", "n34", "n34a"],
    )
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna()
    raw["date"] = pd.to_datetime(dict(year=raw["yr"].astype(int), month=raw["mon"].astype(int), day=1))
    return raw[["date", "n34", "n34a"]].reset_index(drop=True)


@st.cache_data(ttl=6 * 3600)
def nino34_climatology() -> pd.Series:
    """Mean Niño 3.4 SST for each calendar month over 1991-2020 (from NOAA data)."""
    n = load_nino()
    base = n[(n.date.dt.year >= CLIM_START) & (n.date.dt.year <= CLIM_END)]
    return base.groupby(base.date.dt.month)["n34"].mean()


# ============================================================================
# DATA: LIVE SST (refetched every minute)
# ============================================================================


@st.cache_data(ttl=REFRESH_SECONDS - 5, show_spinner=False)
def fetch_live_sst():
    """Hourly SST for 25 grid points inside Niño 3.4 (past 2 days). Real API call."""
    pts = [(la, lo) for la in GRID_LATS for lo in GRID_LONS]
    r = requests.get(
        MARINE_URL,
        params={
            "latitude": ",".join(str(p[0]) for p in pts),
            "longitude": ",".join(str(p[1]) for p in pts),
            "hourly": "sea_surface_temperature",
            "past_days": 2,
            "forecast_days": 1,
            "timezone": "UTC",
        },
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    if isinstance(js, dict):
        js = [js]
    times = pd.to_datetime(js[0]["hourly"]["time"])
    arr = np.array(
        [[np.nan if v is None else v for v in j["hourly"]["sea_surface_temperature"]] for j in js],
        dtype=float,
    )  # shape: (points, hours)
    return times, arr, pts, dt.datetime.utcnow()


# ============================================================================
# DATA: LOCAL CLIMATE (daily -> monthly, cached)
# ============================================================================


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading local climate history (ERA5)...")
def load_local(lat: float, lon: float) -> pd.DataFrame:
    end = (dt.date.today() - dt.timedelta(days=7)).isoformat()
    r = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": "1950-01-01", "end_date": end,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto",
        },
        timeout=90,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["daily"])
    df["time"] = pd.to_datetime(df["time"])
    mon = df.set_index("time").resample("MS").agg(
        rain=("precipitation_sum", "sum"),
        temp=("temperature_2m_mean", "mean"),
        n=("temperature_2m_mean", "count"),
    )
    mon = mon[mon.n == mon.index.days_in_month]  # complete months only
    base = mon[(mon.index.year >= CLIM_START) & (mon.index.year <= CLIM_END)]
    clim = base.groupby(base.index.month)[["rain", "temp"]].mean()
    mon["rain_anom"] = mon["rain"] - clim["rain"].reindex(mon.index.month).values
    mon["temp_anom"] = mon["temp"] - clim["temp"].reindex(mon.index.month).values
    mon.index.name = "date"
    return mon.reset_index()[["date", "rain", "temp", "rain_anom", "temp_anom"]]


# ============================================================================
# MACHINE LEARNING
# ============================================================================


def build_features() -> pd.DataFrame:
    d = load_oni().merge(load_nino()[["date", "n34a"]], on="date", how="inner")
    d = d.sort_values("date").reset_index(drop=True)
    d["oni_l0"] = d["oni"]
    for lag in (1, 2, 3, 6, 12):
        d[f"oni_l{lag}"] = d["oni"].shift(lag)
    d["n34_l0"] = d["n34a"]
    for lag in (1, 2):
        d[f"n34_l{lag}"] = d["n34a"].shift(lag)
    d["m_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
    d["m_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
    return d


@st.cache_resource(ttl=6 * 3600, show_spinner="Training ML models (1-6 month horizons)...")
def train_all():
    d = build_features()
    models, metrics, tests = {}, {}, {}
    for h in range(1, 7):
        t = d.copy()
        t["target"] = t["oni"].shift(-h)
        t = t.dropna(subset=FEATS + ["target"])
        split = int(len(t) * 0.85)  # chronological split, NO shuffling
        tr, te = t.iloc[:split], t.iloc[split:]

        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf.fit(tr[FEATS].values, tr["target"].values)
        pred = rf.predict(te[FEATS].values)
        rmse = float(np.sqrt(np.mean((te["target"].values - pred) ** 2)))
        base = float(np.sqrt(np.mean((te["target"].values - te["oni"].values) ** 2)))  # persistence

        final = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
        final.fit(t[FEATS].values, t["target"].values)  # refit on all data for the live forecast

        models[h] = final
        metrics[h] = (rmse, base)
        tests[h] = pd.DataFrame({
            "date": te["date"] + pd.DateOffset(months=h),
            "actual": te["target"].values,
            "pred": pred,
        })
    return d, models, metrics, tests


# ============================================================================
# LIVE PANEL (auto-refreshes every REFRESH_SECONDS)
# ============================================================================


@st.fragment(run_every=REFRESH_SECONDS)
def live_panel():
    st.session_state["n_fetch"] = st.session_state.get("n_fetch", 0) + 1
    try:
        clim = nino34_climatology()
        times, arr, pts, fetched = fetch_live_sst()
    except Exception as e:  # real mode: show the error, never fake data
        st.error(f"Live SST fetch failed: {e}")
        return

    now = pd.Timestamp.now("UTC").tz_localize(None)
    idx = np.where(times <= now)[0]
    if len(idx) == 0:
        st.error("No SST data available for the current time.")
        return
    i = idx[-1]
    cur = arr[:, i]
    if np.all(np.isnan(cur)):
        st.error("API returned no valid ocean points for the current hour.")
        return

    # ---------- THE CALCULATION ----------
    sst = float(np.nanmean(cur))                       # 1) box-average SST of valid grid points
    clim_now = clim_at(times[i], clim)                 # 2) NOAA 1991-2020 climatology for this date
    anom = sst - clim_now                              # 3) anomaly = SST - climatology
    live_phase = phase(anom)                           # 4) classify (|anomaly| >= 0.5 °C)
    # -------------------------------------

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Niño 3.4 SST (live)", f"{sst:.2f} °C")
    h2.metric("Climatology (1991-2020)", f"{clim_now:.2f} °C")
    h3.metric("Live anomaly", f"{anom:+.2f} °C")
    h4.markdown(
        f"<div style='font-size:0.85rem;opacity:0.7'>Instant phase</div>"
        f"<div style='font-size:1.9rem;font-weight:700;color:{PHASE_COLOR[live_phase]}'>{live_phase}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])

    with left:
        mask = times <= now
        series = pd.DataFrame(arr[:, mask].T).mean(axis=1).values  # mean over grid, skips NaN
        t_used = times[mask]
        clim_line = [clim_at(t, clim) for t in t_used]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_used, y=series, name="Box-average SST",
                                 line=dict(width=3, color="#e5484d")))
        fig.add_trace(go.Scatter(x=t_used, y=clim_line, name="Climatology",
                                 line=dict(dash="dash", color="#8b8b8b")))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                          title="Last 48 hours (UTC)", yaxis_title="°C",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        lats = [p[0] for p in pts]
        lons = [p[1] for p in pts]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode="markers",
            marker=dict(size=26, symbol="square", color=cur, colorscale="RdYlBu_r",
                        colorbar=dict(title="°C"), cmin=np.nanmin(cur), cmax=np.nanmax(cur)),
            text=[f"{v:.2f} °C" if not np.isnan(v) else "no data" for v in cur],
            hovertemplate="lat %{y}, lon %{x}<br>%{text}<extra></extra>",
        ))
        fig.add_shape(type="rect", x0=BOX["lon_min"], x1=BOX["lon_max"],
                      y0=BOX["lat_min"], y1=BOX["lat_max"], line=dict(color="#888", dash="dot"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                          title="Niño 3.4 sampling grid (25 points)",
                          xaxis_title="Longitude", yaxis_title="Latitude")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Fetched {fetched:%H:%M:%S} UTC · refresh #{st.session_state['n_fetch']} · "
        f"auto-refresh every {REFRESH_SECONDS}s · {int(np.sum(~np.isnan(cur)))}/25 ocean points valid · "
        "Source: Open-Meteo Marine API (hourly SST analysis, so values change hourly, "
        "but the app re-checks every minute)."
    )


# ============================================================================
# UI HELPERS
# ============================================================================

def section(title: str, sub: str = ""):
    st.markdown(f"<div class='sec'><h3>{title}</h3><span>{sub}</span></div>", unsafe_allow_html=True)


# ============================================================================
# CURRENT WEATHER (Open-Meteo, refreshes every 5 min)
# ============================================================================

WMO = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Rime fog", "🌫️"), 51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"), 56: ("Freezing drizzle", "🌧️"), 57: ("Freezing drizzle", "🌧️"),
    61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "🌨️"), 75: ("Heavy snow", "❄️"), 77: ("Snow grains", "❄️"),
    80: ("Rain showers", "🌦️"), 81: ("Rain showers", "🌧️"), 82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "🌨️"), 86: ("Snow showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm, hail", "⛈️"), 99: ("Thunderstorm, hail", "⛈️"),
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> dict:
    r = requests.get(WEATHER_URL, params=dict(
        latitude=lat, longitude=lon, timezone="auto",
        current="temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,"
                "weather_code,cloud_cover,surface_pressure,wind_speed_10m,wind_direction_10m",
    ), timeout=10)
    r.raise_for_status()
    return r.json()["current"]


def compass(deg: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) % 360 // 45)]


@st.fragment(run_every=300)
def weather_panel(place_name: str, lat: float, lon: float):
    section("Current weather", place_name)
    try:
        w = fetch_weather(lat, lon)
    except Exception as e:
        st.error(f"Weather fetch failed: {e}")
        return
    desc, icon = WMO.get(int(w["weather_code"]), ("Unknown", "🌡️"))
    st.markdown(f"""
<div class='card wx'>
  <div class='wx-top'>
    <div class='wx-icon'>{icon}</div>
    <div>
      <div class='wx-temp'>{w['temperature_2m']:.1f}°C</div>
      <div class='wx-desc'>{desc} · feels like {w['apparent_temperature']:.1f}°C</div>
    </div>
  </div>
  <div class='grid'>
    <div><span>Humidity</span><b>{w['relative_humidity_2m']:.0f}%</b></div>
    <div><span>Wind</span><b>{w['wind_speed_10m']:.0f} km/h {compass(w['wind_direction_10m'])}</b></div>
    <div><span>Pressure</span><b>{w['surface_pressure']:.0f} hPa</b></div>
    <div><span>Cloud cover</span><b>{w['cloud_cover']:.0f}%</b></div>
    <div><span>Rain now</span><b>{w['precipitation']:.1f} mm</b></div>
    <div><span>Updated</span><b>{str(w['time'])[-5:]} local</b></div>
  </div>
</div>""", unsafe_allow_html=True)
    st.caption("Source: Open-Meteo current-conditions API (updates every 5 min).")


# ============================================================================
# PHYSICAL WATER TEMPERATURE SENSOR (Arduino + DS18B20 over USB serial)
# ============================================================================

AUTO = "Auto-detect (recommended)"


class NoArduino(Exception):
    pass


def list_serial_ports():
    try:
        from serial.tools import list_ports
        return list(list_ports.comports())
    except Exception:
        return []


def _rank(p) -> int:
    t = f"{p.description} {p.manufacturer or ''}".lower()
    if "bluetooth" in t:
        return 9
    return 0 if any(k in t for k in ("arduino", "ch340", "ch341", "wch", "ftdi", "usb serial", "mega")) else 1


def detect_sensor_port():
    """Try every COM port and return the one that streams our DS18B20 JSON."""
    import serial
    for p in sorted(list_serial_ports(), key=_rank):
        if _rank(p) >= 9:
            continue
        try:
            with serial.Serial(p.device, 9600, timeout=2) as s:
                time.sleep(2.2)              # board resets when the port opens
                s.reset_input_buffer()
                for _ in range(3):
                    line = s.readline().decode(errors="ignore").strip()
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and ("temperature_c" in obj or "error" in obj):
                        return p.device
        except Exception:
            continue
    return None


@st.cache_resource
def get_serial(port: str, baud: int = 9600):
    import serial  # pip install pyserial
    ser = serial.Serial(port, baud, timeout=3)
    time.sleep(2)
    return ser


def read_uno_json(port: str) -> dict:
    ser = get_serial(port)
    last = None
    while ser.in_waiting:              # drain buffer, keep the newest line
        raw = ser.readline()
        if raw.strip():
            last = raw
    if last is None:
        last = ser.readline()
    return json.loads(last.decode(errors="ignore").strip())


@st.fragment(run_every=5)
def physical_sensor_panel(choice: str):
    section("Water probe (DS18B20)", "Arduino · USB serial")
    auto = choice == AUTO
    port = st.session_state.get("auto_port") if auto else choice
    try:
        if not port:
            with st.spinner("Scanning USB ports for the Arduino..."):
                port = detect_sensor_port()
            if not port:
                raise NoArduino()
            st.session_state["auto_port"] = port
        payload = read_uno_json(port)
        if "error" in payload:
            raise ValueError("Sensor not found")
        temp_c = float(payload["temperature_c"])
        if not np.isfinite(temp_c) or temp_c < -55 or temp_c > 125:
            raise ValueError("Temperature is outside the DS18B20 range.")
        st.markdown(f"""
<div class='card probe'>
  <div class='probe-label'>Probe water temperature</div>
  <div class='probe-temp'>{temp_c:.2f}<small>°C</small></div>
  <div class='probe-f'>{temp_c * 9 / 5 + 32:.2f} °F</div>
  <div class='probe-status'><span class='dot'></span> Connected on <b>{port}</b>{' (auto)' if auto else ''} · {dt.datetime.now():%H:%M:%S}</div>
</div>""", unsafe_allow_html=True)
    except NoArduino:
        st.warning("No Arduino found. Plug in the USB cable, close Arduino Serial Monitor, "
                   "then press **Rescan** in the sidebar.")
    except ImportError:
        st.error("pyserial missing. Run:  pip install pyserial")
    except OSError as e:   # includes serial.SerialException
        get_serial.clear()
        if auto:
            st.session_state.pop("auto_port", None)
        st.error(f"Cannot use {port}. Close Arduino Serial Monitor / any other program using it. ({e})")
    except json.JSONDecodeError:
        st.info("Waiting for the next reading...")
    except ValueError:
        st.error("Sensor not found. Check DS18B20 wiring and the 4.7k pull-up resistor.")
    except KeyError:
        st.error("Unexpected data from Arduino. Upload ds18b20_uno.ino again.")


# ============================================================================
# PAGE
# ============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
footer, #MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.topbar { background: linear-gradient(120deg,#0a1b33 0%,#0f3a66 55%,#117a8b 100%); color:#fff;
  border-radius:16px; padding:22px 28px; margin-bottom:18px; display:flex; justify-content:space-between;
  align-items:center; flex-wrap:wrap; gap:12px; box-shadow:0 6px 24px rgba(10,27,51,.25); }
.topbar h1 { font-size:1.75rem; font-weight:800; margin:0; letter-spacing:-.4px; color:#fff; }
.topbar p { margin:4px 0 0; opacity:.8; font-size:.92rem; }
.live { background:rgba(255,255,255,.14); border-radius:999px; padding:6px 14px; font-size:.8rem; font-weight:600; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#22c55e; margin-right:6px;
  animation:pulse 1.6s infinite; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(34,197,94,.7)} 70%{box-shadow:0 0 0 9px rgba(34,197,94,0)} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0)} }

.sec { display:flex; align-items:baseline; gap:10px; margin:22px 0 10px; border-left:4px solid #117a8b; padding-left:10px; }
.sec h3 { margin:0; font-size:1.15rem; font-weight:700; }
.sec span { opacity:.6; font-size:.85rem; }

.card { border:1px solid rgba(128,128,128,.22); border-radius:16px; padding:20px 22px; background:rgba(128,128,128,.07); }
.hero { color:#fff; border:none; background:linear-gradient(135deg,#0b1f3a,#15467a); box-shadow:0 6px 24px rgba(10,27,51,.22); }
.hero .lbl { font-size:.78rem; letter-spacing:1.6px; opacity:.7; font-weight:600; }
.hero .ph { font-size:2.7rem; font-weight:800; line-height:1.1; margin:4px 0; }
.hero .meta { opacity:.85; font-size:.95rem; margin-bottom:10px; }
.hero .adv { opacity:.8; font-size:.88rem; line-height:1.5; margin-bottom:14px; }
.chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip { background:rgba(255,255,255,.12); border-radius:10px; padding:8px 12px; font-size:.8rem; }
.chip b { display:block; font-size:.95rem; }

.wx { color:#fff; border:none; background:linear-gradient(135deg,#0f3d4c,#1b7f8c); }
.wx-top { display:flex; align-items:center; gap:16px; margin-bottom:14px; }
.wx-icon { font-size:3.2rem; }
.wx-temp { font-size:2.6rem; font-weight:800; line-height:1; }
.wx-desc { opacity:.85; font-size:.92rem; margin-top:4px; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.grid div { background:rgba(255,255,255,.12); border-radius:10px; padding:8px 10px; }
.grid span { display:block; font-size:.7rem; opacity:.75; text-transform:uppercase; letter-spacing:.8px; }
.grid b { font-size:.95rem; }

.probe { text-align:center; }
.probe-label { font-size:.78rem; letter-spacing:1.4px; text-transform:uppercase; opacity:.65; font-weight:600; }
.probe-temp { font-size:3.4rem; font-weight:800; line-height:1.1; color:#117a8b; }
.probe-temp small { font-size:1.4rem; font-weight:600; margin-left:2px; }
.probe-f { opacity:.7; margin-bottom:10px; }
.probe-status { font-size:.82rem; opacity:.8; }

[data-testid="stMetric"] { border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:12px 16px;
  background:rgba(128,128,128,.07); }
[data-testid="stMetricLabel"] { opacity:.7; }
.stTabs [data-baseweb="tab"] { font-weight:600; padding:10px 18px; }
section[data-testid="stSidebar"] { border-right:1px solid rgba(128,128,128,.2); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(f"""
<div class='topbar'>
  <div>
    <h1>🌊 El Niño Early Warning &amp; Climate Intelligence</h1>
    <p>Live NOAA &amp; Open-Meteo data · Machine-learning ENSO forecast · Local sensor monitoring</p>
  </div>
  <div class='live'><span class='dot'></span>LIVE · {dt.datetime.now():%d %b %Y, %H:%M}</div>
</div>
""", unsafe_allow_html=True)

hero = st.container()   # filled in once NOAA data is loaded (below)

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    port_names = [p.device for p in list_serial_ports()]
    port_choice = st.selectbox(
        "Arduino COM port", [AUTO] + port_names,
        help="Auto-detect tries every USB port and picks the one sending DS18B20 data."
    )
    if st.button("🔌 Rescan USB ports"):
        st.session_state.pop("auto_port", None)
        get_serial.clear()
        st.rerun()
    place = st.selectbox("Location (weather & local impact)", list(PLACES.keys()))
    if PLACES[place] is None:
        lat = st.number_input("Latitude", -90.0, 90.0, 12.92, 0.01)
        lon = st.number_input("Longitude", -180.0, 180.0, 80.12, 0.01)
    else:
        lat, lon = PLACES[place]
        st.caption(f"lat {lat}, lon {lon}")
    horizon = st.slider("Forecast horizon (months)", 1, 6, 3)
    season = st.selectbox("Season for impact analysis", list(SEASONS.keys()))
    lag = st.slider("Response lag (months)", 0, 3, 0,
                    help="Climate reacts to the Pacific with a delay in some regions.")
    st.divider()
    if st.button("🔄 Force reload all data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ---------------- local conditions ----------------
section("Local conditions", "your location + your sensor")
c_w, c_s = st.columns([3, 2])
with c_w:
    weather_panel(place if PLACES[place] else f"{lat:.2f}, {lon:.2f}", lat, lon)
with c_s:
    physical_sensor_panel(port_choice)

# ---------------- live Pacific ----------------
section("Pacific Ocean · live Niño 3.4", "box-average SST vs 1991-2020 climatology")
live_panel()

st.divider()

# ---------------- official data + ML ----------------
try:
    d, models, metrics, tests = train_all()
except Exception as e:
    st.error(f"Could not load NOAA data / train models: {e}")
    st.stop()

last_date = d["date"].iloc[-1]
last_oni = float(d["oni"].iloc[-1])
x_last = d.iloc[[-1]][FEATS].values

rows = []
for h in range(1, 7):
    p = float(models[h].predict(x_last)[0])
    rows.append((last_date + pd.DateOffset(months=h), p, metrics[h][0]))
fc = pd.DataFrame(rows, columns=["date", "pred", "rmse"])
sel = fc.iloc[horizon - 1]

# ---------------- hero: current El Niño level ----------------
_SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
_ADV = {
    "El Niño": "Central Pacific is warmer than normal. Historically this tends to bring a weaker Indian summer "
               "monsoon, drier Indonesia/Australia and wetter Peru/Ecuador.",
    "La Niña": "Central Pacific is cooler than normal. Historically this tends to bring a stronger Indian summer "
               "monsoon, wetter Indonesia/Australia and drier Peru/Ecuador.",
    "Neutral": "Pacific temperatures are near normal. No strong El Niño or La Niña signal right now.",
}
cur_phase = phase(last_oni)
cur_col = PHASE_COLOR[cur_phase]
chips = "".join(
    f"<div class='chip'>{r.date:%b %Y}<b style='color:{PHASE_COLOR[phase(r.pred)]}'>{phase(r.pred)}</b>{r.pred:+.2f}</div>"
    for r in fc.head(3).itertuples()
)
with hero:
    hl, hr = st.columns([3, 2])
    with hl:
        st.markdown(f"""
<div class='card hero'>
  <div class='lbl'>CURRENT ENSO STATUS (NOAA ONI)</div>
  <div class='ph' style='color:{cur_col if cur_phase != "Neutral" else "#e2e8f0"}'>{cur_phase}</div>
  <div class='meta'>ONI <b>{last_oni:+.2f} °C</b> · Strength: <b>{strength(last_oni)}</b> · Season {_SEAS[last_date.month - 1]} {last_date.year}</div>
  <div class='adv'>{_ADV[cur_phase]}</div>
  <div class='lbl' style='margin-bottom:6px'>ML OUTLOOK · NEXT 3 MONTHS</div>
  <div class='chips'>{chips}</div>
</div>""", unsafe_allow_html=True)
    with hr:
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=last_oni,
            number=dict(suffix=" °C", valueformat="+.2f"),
            title=dict(text="ONI level"),
            gauge=dict(
                axis=dict(range=[-2.5, 2.5], tickvals=[-2, -1, -0.5, 0, 0.5, 1, 2]),
                bar=dict(color=cur_col, thickness=0.28),
                steps=[dict(range=[-2.5, -0.5], color="rgba(59,130,246,.25)"),
                       dict(range=[-0.5, 0.5], color="rgba(139,139,139,.22)"),
                       dict(range=[0.5, 2.5], color="rgba(229,72,77,.25)")],
            )))
        g.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g, use_container_width=True)

st.subheader("🛰️ Official status & AI forecast")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Latest ONI ({last_date:%b %Y})", f"{last_oni:+.2f}")
c2.metric("Current phase", f"{phase(last_oni)}", strength(last_oni))
c3.metric(f"Forecast ONI ({sel.date:%b %Y})", f"{sel.pred:+.2f}", f"± {sel.rmse:.2f} typical error",
          delta_color="off")
c4.metric("Forecast phase", phase(sel.pred), strength(sel.pred))

tab_fc, tab_imp, tab_q, tab_how = st.tabs(
    ["🔮 ENSO forecast", "🌧️ Local impact", "🧪 Model quality", "📘 How it's calculated"]
)

# ---------------- tab: forecast ----------------
with tab_fc:
    hist = d.tail(120)
    fx = [last_date] + list(fc["date"])
    fy = [last_oni] + list(fc["pred"])
    fe = [0.0] + list(fc["rmse"])
    upper = [y + e for y, e in zip(fy, fe)]
    lower = [y - e for y, e in zip(fy, fe)]

    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=3.5, fillcolor="red", opacity=0.07, line_width=0)
    fig.add_hrect(y0=-3.5, y1=-0.5, fillcolor="blue", opacity=0.07, line_width=0)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#e5484d")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="#3b82f6")
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["oni"], name="Observed ONI (NOAA)",
                             line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=fx + fx[::-1], y=upper + lower[::-1], fill="toself",
                             fillcolor="rgba(255,140,0,0.2)", line=dict(width=0),
                             name="± typical error (test RMSE)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=fy, name="ML forecast", mode="lines+markers",
                             line=dict(dash="dash", width=3, color="orange")))
    fig.update_layout(height=430, yaxis=dict(range=[-3, 3], title="ONI (°C anomaly)"),
                      margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    table = fc.copy()
    table["Month"] = table["date"].dt.strftime("%b %Y")
    table["Forecast ONI"] = table["pred"].round(2)
    table["± error"] = table["rmse"].round(2)
    table["Phase"] = table["pred"].apply(phase)
    table["Strength"] = table["pred"].apply(strength)
    st.dataframe(table[["Month", "Forecast ONI", "± error", "Phase", "Strength"]],
                 hide_index=True, use_container_width=True)

# ---------------- tab: impact ----------------
with tab_imp:
    st.markdown(f"**Location:** {place} ({lat:.2f}, {lon:.2f}) · **Season:** {season} · **Lag:** {lag} month(s)")
    try:
        local = load_local(lat, lon)
    except Exception as e:
        st.error(f"Local climate download failed: {e}")
        local = None

    if local is not None:
        o = load_oni().copy()
        o["date"] = o["date"] + pd.DateOffset(months=lag)  # ONI at t-lag -> climate at t
        imp = local.merge(o, on="date", how="inner")
        imp = imp[imp["date"].dt.month.isin(SEASONS[season])].copy()
        imp["phase"] = imp["oni"].apply(phase)

        if len(imp) < 24:
            st.warning("Not enough overlapping months for a reliable analysis.")
        else:
            grp = imp.groupby("phase").agg(
                rain=("rain_anom", "mean"), temp=("temp_anom", "mean"), n=("rain_anom", "count")
            ).reindex(["El Niño", "Neutral", "La Niña"])
            r_rain = imp["oni"].corr(imp["rain_anom"])
            r_temp = imp["oni"].corr(imp["temp_anom"])

            ph = phase(sel.pred)
            st.subheader(f"Expected impact if {ph} develops by {sel.date:%b %Y}")
            if ph in grp.index and not np.isnan(grp.loc[ph, "rain"]):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rain vs normal", f"{grp.loc[ph, 'rain']:+.1f} mm/month")
                m2.metric("Temperature vs normal", f"{grp.loc[ph, 'temp']:+.2f} °C")
                m3.metric("Rain correlation (r)", f"{r_rain:+.2f}")
                m4.metric("Temp correlation (r)", f"{r_temp:+.2f}")
                st.caption(f"Based on {int(grp.loc[ph, 'n'])} historical {ph} months since 1950. "
                           "|r| below ~0.2 means ENSO has only a weak, noisy link here. "
                           "Treat this as a tendency, not a guarantee.")
            else:
                st.info("No historical months for this phase in the selected season.")

            b1, b2 = st.columns(2)
            colors = [PHASE_COLOR[p] for p in grp.index]
            with b1:
                f = go.Figure(go.Bar(x=grp.index, y=grp["rain"], marker_color=colors,
                                     text=grp["n"].apply(lambda v: f"n={int(v)}" if not np.isnan(v) else ""),
                                     textposition="outside"))
                f.update_layout(title="Avg rainfall anomaly by phase (mm/month)", height=340,
                                margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(f, use_container_width=True)
            with b2:
                f = go.Figure(go.Bar(x=grp.index, y=grp["temp"], marker_color=colors))
                f.update_layout(title="Avg temperature anomaly by phase (°C)", height=340,
                                margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(f, use_container_width=True)

            sc = go.Figure()
            sc.add_trace(go.Scatter(x=imp["oni"], y=imp["rain_anom"], mode="markers",
                                    marker=dict(size=6, opacity=0.5), name="Months"))
            k, b = np.polyfit(imp["oni"], imp["rain_anom"], 1)
            xs = np.array([imp["oni"].min(), imp["oni"].max()])
            sc.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="Trend",
                                    line=dict(color="orange", width=3)))
            sc.update_layout(title="ONI vs local rainfall anomaly", height=360,
                             xaxis_title="ONI (°C)", yaxis_title="Rain anomaly (mm/month)",
                             margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(sc, use_container_width=True)

# ---------------- tab: model quality ----------------
with tab_q:
    rmse, base = metrics[horizon]
    skill = 1 - rmse / base
    q1, q2, q3 = st.columns(3)
    q1.metric(f"ML RMSE ({horizon}-month ahead)", f"{rmse:.3f}")
    q2.metric("Persistence baseline RMSE", f"{base:.3f}", help="Baseline: 'future = today'")
    q3.metric("Skill vs baseline", f"{skill * 100:+.0f}%")
    if skill <= 0:
        st.warning("At this horizon the model does NOT beat the simple persistence baseline.")

    te = tests[horizon]
    f = go.Figure()
    f.add_trace(go.Scatter(x=te["date"], y=te["actual"], name="Actual ONI", line=dict(width=2.5)))
    f.add_trace(go.Scatter(x=te["date"], y=te["pred"], name="Predicted", line=dict(dash="dash", color="orange")))
    f.update_layout(title="Held-out test period (last 15% of history, never seen in training)",
                    height=380, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(f, use_container_width=True)

    imp_s = pd.Series(models[horizon].feature_importances_, index=FEATS).sort_values()
    f = go.Figure(go.Bar(x=imp_s.values, y=imp_s.index, orientation="h"))
    f.update_layout(title="Feature importance", height=360, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(f, use_container_width=True)
    st.caption("ENSO forecasts are hard across the spring predictability barrier (Mar-May). "
               "Skill drops as the horizon grows. That is expected.")

# ---------------- tab: how ----------------
with tab_how:
    st.markdown(f"""
### How the live ocean temperature is calculated
1. **Sample**: a 5x5 grid of points inside the Niño 3.4 box (5°N-5°S, 170°W-120°W), hourly SST from the Open-Meteo Marine API.
2. **Average**: mean of all valid ocean points = `Niño 3.4 SST`.
3. **Climatology**: NOAA's monthly Niño 3.4 SST averaged over {CLIM_START}-{CLIM_END}, interpolated to the current day.
4. **Anomaly** = `SST - climatology`.
5. **Phase**: anomaly ≥ +0.5 °C → El Niño, ≤ -0.5 °C → La Niña, otherwise Neutral.

> The live value is an *instantaneous* indicator. NOAA's official **ONI** is a 3-month running mean of the Niño 3.4 anomaly,
> and the data source (Open-Meteo analysis) can differ slightly from NOAA's ERSSTv5, so expect an offset of a few tenths of a degree.

### ML forecast
- Features: ONI lags (0, 1, 2, 3, 6, 12 months), Niño 3.4 anomaly lags, month-of-year (sin/cos).
- Model: Random Forest, one model per horizon (1-6 months), chronological train/test split (no leakage).
- Uncertainty band = the model's RMSE on unseen test data.

### Local impact
- Monthly rain / temperature anomalies from ERA5 (via Open-Meteo), relative to {CLIM_START}-{CLIM_END}.
- Months grouped by ENSO phase (with optional lag) → average anomaly per phase + correlation.
""")
