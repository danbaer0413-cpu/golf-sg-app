import base64
import io
import math
import re
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="Golf Strokes Gained", page_icon="⛳", layout="centered")

# ---------------------------------------------------------------------------
# Baseline expected strokes (approximate PGA Tour-style tables).
# Green = feet. Everything else = yards to the hole.
# ---------------------------------------------------------------------------
BASE = {
    "Green": ([3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 90],
              [1.04, 1.13, 1.23, 1.34, 1.42, 1.50, 1.56, 1.61, 1.78, 1.87, 1.98, 2.06, 2.14, 2.21, 2.40]),
    "Tee": ([100, 200, 300, 400, 450, 500, 550, 600],
            [2.92, 3.45, 3.71, 3.99, 4.17, 4.41, 4.75, 5.14]),
    "Fairway": ([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
                [2.40, 2.60, 2.70, 2.75, 2.80, 2.85, 2.91, 2.98, 3.08, 3.19, 3.32, 3.45, 3.58, 3.69, 3.78]),
    "Rough": ([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
              [2.59, 2.78, 2.91, 2.96, 3.02, 3.08, 3.15, 3.23, 3.31, 3.42, 3.53, 3.64, 3.74, 3.83, 3.90]),
    "Sand": ([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
             [2.53, 2.82, 3.15, 3.24, 3.23, 3.21, 3.22, 3.28, 3.40, 3.55, 3.70, 3.84, 3.93, 4.00, 4.04]),
    "Recovery": ([100, 200, 300], [3.80, 3.99, 4.25]),
}
LIES = list(BASE.keys())
CLUBS = ["Driver", "3W", "7W", "9W", "5i", "6i", "7i", "8i", "9i",
         "46°", "50°", "54°", "60°", "Putter"]
WEDGE_LOFTS = {46: "46°", 50: "50°", 54: "54°", 60: "60°"}


def expected(lie, dist):
    xs, ys = BASE[lie]
    return float(np.interp(dist, xs, ys))


def haversine_yds(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a)) * 1.09361


def to_yds(dist, unit):
    return dist if unit == "yd" else dist / 3.0


# Generic starting distances, used for any club with fewer than 3 shots of your own.
DEFAULT_CLUB_YDS = {"Driver": 275, "3W": 255, "7W": 233, "9W": 215, "5i": 208, "6i": 196,
                    "7i": 177, "8i": 165, "9i": 153, "46°": 140, "50°": 125, "54°": 110, "60°": 90}
# Typical roll after landing (yds), used only to turn FlightScope CARRY into an estimated TOTAL.
ROLL_DEFAULT = {"Driver": 20, "3W": 15, "7W": 12, "9W": 10, "5i": 6, "6i": 5, "7i": 4,
                "8i": 3, "9i": 3, "46°": 2, "50°": 2, "54°": 1, "60°": 1}
WIND_OPTS = ["Calm", "Into", "Helping", "Cross"]


def plays_factor(temp_f, wind_mph, wind_rel):
    """>1 means the shot plays LONGER than the number (headwind, cold air).
    Rules of thumb: +1% per mph into the wind, -0.5% per mph helping,
    +0.12% per degree F below 70."""
    f = 1.0
    if not pd.isna(wind_mph) and wind_mph:
        if wind_rel == "Into":
            f *= 1 + 0.01 * wind_mph
        elif wind_rel == "Helping":
            f *= 1 - 0.005 * wind_mph
    if not pd.isna(temp_f):
        f *= 1 + (float(st.session_state.get("temp_coef", 0.12)) / 100) * (70 - temp_f)
    return f


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat, lon):
    """Current conditions from Open-Meteo (free, no API key)."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=dict(
            latitude=lat, longitude=lon,
            current="temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            temperature_unit="fahrenheit", wind_speed_unit="mph"), timeout=10)
        r.raise_for_status()
        c = r.json()["current"]
        return dict(temp_f=c["temperature_2m"], humidity=c["relative_humidity_2m"],
                    wind_mph=c["wind_speed_10m"], wind_deg=c["wind_direction_10m"])
    except Exception:
        return None


def normalize_club(name):
    """Map FlightScope club names to the bag in this app (extra text like 'tour ad' is ignored)."""
    s = str(name).strip().lower().replace("-", " ")
    if s in ("driver", "dr", "1 wood", "1w"):
        return "Driver"
    m = re.match(r"^(\d)\s*(wood|w)\b", s)
    if m:
        c = f"{m.group(1)}W"
        return c if c in CLUBS else None
    m = re.match(r"^(\d)\s*(iron|i)\b", s)
    if m:
        c = f"{m.group(1)}i"
        return c if c in CLUBS else None
    m = re.match(r"^(\d{2})\s*(?:°|deg\w*)?\s*(?:wedge|wd|w)?$", s)
    if m:
        loft = int(m.group(1))
        near = min(WEDGE_LOFTS, key=lambda k: abs(k - loft))
        return WEDGE_LOFTS[near] if abs(near - loft) <= 1 else None
    named = {"pw": 46, "pitching wedge": 46, "gw": 50, "aw": 50, "gap wedge": 50, "approach wedge": 50,
             "sw": 54, "sand wedge": 54, "lw": 60, "lob wedge": 60}
    return WEDGE_LOFTS.get(named.get(s))


def build_baseline(raw, club_col, carry_col, total_col, meters):
    k = 1.09361 if meters else 1.0
    d = pd.DataFrame({"club": raw[club_col].map(normalize_club),
                      "carry": pd.to_numeric(raw[carry_col], errors="coerce") * k})
    d["total"] = (pd.to_numeric(raw[total_col], errors="coerce") * k
                  if total_col != "(none)" else np.nan)
    unmapped = sorted(raw.loc[d["club"].isna(), club_col].astype(str).unique())
    d = d.dropna(subset=["club"])
    g = d.groupby("club").agg(carry=("carry", "median"), total=("total", "median"),
                              shots=("carry", "count")).reset_index()
    g[["carry", "total"]] = g[["carry", "total"]].round(1)
    return g, unmapped


def club_table(df, base=None, mode="total"):
    """Typical distance per club. Priority: your on-course shots (3+, weather-normalised),
    then your FlightScope numbers, then generic defaults."""
    own = {}
    if df is not None and not df.empty and {"hit_yds", "lie", "distance", "club", "penalty"} <= set(df.columns):
        d = df[df["lie"].isin(["Tee", "Fairway", "Rough"]) & df["hit_yds"].notna()
               & (pd.to_numeric(df["distance"], errors="coerce") >= 40)
               & (pd.to_numeric(df["penalty"], errors="coerce").fillna(0) == 0)].copy()
        if len(d):
            for c in ("temp_f", "wind_mph", "wind_rel"):
                if c not in d.columns:
                    d[c] = np.nan
            f = [plays_factor(t, w, r) for t, w, r in zip(d["temp_f"], d["wind_mph"], d["wind_rel"])]
            d["norm"] = d["hit_yds"].astype(float) * np.array(f)
            for club, g in d.groupby("club"):
                own[club] = (float(g["norm"].median()), len(g))
    fs = {}
    fsc = {}
    if base is not None and len(base):
        # Normalise the range session to 70F, the same as on-course shots.
        tf = plays_factor(float(st.session_state.get("base_temp", 70.0)), 0, "Calm")
        rs = float(st.session_state.get("roll_scale", 1.0))
        for _, r in base.iterrows():
            carry, total = r.get("carry"), r.get("total")
            if not pd.isna(carry):
                fsc[r["club"]] = float(carry) * tf
            v = None
            if mode == "total":
                if not pd.isna(total):
                    v = total
                elif not pd.isna(carry):
                    v = carry + ROLL_DEFAULT.get(r["club"], 0) * rs
            else:
                v = carry if not pd.isna(carry) else total
            if v is not None and not pd.isna(v):
                fs[r["club"]] = float(v) * tf
    rows = []
    for club, dflt in DEFAULT_CLUB_YDS.items():
        avg, n = own.get(club, (None, 0))
        fsv = fs.get(club)
        if avg is not None and n >= 3:
            val, src = round(avg), "your data"
        elif fsv is not None:
            val, src = round(fsv), "FlightScope"
        else:
            val, src = dflt, "default"
        rows.append(dict(club=club, avg_yds=val, shots=n, source=src,
                         flightscope=round(fsv) if fsv is not None else np.nan,
                         fs_carry=round(fsc[club]) if club in fsc else np.nan,
                         on_course=round(avg) if avg is not None else np.nan))
    return pd.DataFrame(rows)


def compare_view(tbl):
    return pd.DataFrame({"club": tbl["club"], "FS carry": tbl["fs_carry"],
                         "On course": tbl["on_course"],
                         "Implied roll": tbl["on_course"] - tbl["fs_carry"],
                         "FS est. total": tbl["flightscope"],
                         "shots": tbl["shots"], "using": tbl["source"]})


def recommend(tbl, plays_like):
    t = tbl.assign(diff=(tbl["avg_yds"] - plays_like).abs()).sort_values("diff")
    best = t.iloc[0]
    alt = t[t["avg_yds"] < plays_like] if best["avg_yds"] >= plays_like else t[t["avg_yds"] > plays_like]
    return best, (alt.iloc[0] if len(alt) else None)


# ---------------------------------------------------------------------------
# GitHub as the backend: the whole dataset lives in one CSV in a private repo.
# Secrets (Streamlit Cloud > Settings > Secrets):
#   [github]
#   token  = "github_pat_..."
#   repo   = "yourname/golf-data"
#   path   = "data/golf_shots.csv"   # optional
#   branch = "main"                  # optional
# ---------------------------------------------------------------------------
def gh_cfg():
    try:
        g = st.secrets["github"]
        return dict(token=g["token"], repo=g["repo"],
                    path=g.get("path", "data/golf_shots.csv"),
                    branch=g.get("branch", "main"))
    except Exception:
        return None


def _gh_headers(cfg):
    return {"Authorization": f"Bearer {cfg['token']}",
            "Accept": "application/vnd.github+json"}


def _gh_url(cfg):
    return f"https://api.github.com/repos/{cfg['repo']}/contents/{cfg['path']}"


def gh_load(cfg):
    r = requests.get(_gh_url(cfg), headers=_gh_headers(cfg),
                     params={"ref": cfg["branch"]}, timeout=20)
    if r.status_code == 404:
        return pd.DataFrame(), None
    r.raise_for_status()
    j = r.json()
    text = base64.b64decode(j["content"]).decode("utf-8")
    return pd.read_csv(io.StringIO(text)), j["sha"]


def gh_save(cfg, df, message):
    sha = None
    r = requests.get(_gh_url(cfg), headers=_gh_headers(cfg),
                     params={"ref": cfg["branch"]}, timeout=20)
    if r.status_code == 200:
        sha = r.json()["sha"]
    body = {"message": message,
            "content": base64.b64encode(df.to_csv(index=False).encode()).decode(),
            "branch": cfg["branch"]}
    if sha:
        body["sha"] = sha
    r = requests.put(_gh_url(cfg), headers=_gh_headers(cfg), json=body, timeout=30)
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("shots", [])
ss.setdefault("pin", None)
ss.setdefault("hole", 1)
ss.setdefault("data", pd.DataFrame())
ss.setdefault("loaded", False)
ss.setdefault("baseline", None)

CFG = gh_cfg()
if CFG and not ss.get("gh_loaded"):
    try:
        ss.data, _ = gh_load(CFG)
        ss.gh_loaded = True
    except Exception as e:
        ss.flash = f"⚠️ Couldn't load from GitHub: {e}"
        ss.gh_loaded = True
    try:
        b, _ = gh_load(dict(CFG, path="data/flightscope_clubs.csv"))
        if not b.empty:
            ss.baseline = b
    except Exception:
        pass

with st.sidebar:
    st.header("Round")
    course = st.text_input("Course", value="Mercer Oaks East", key="course")
    round_date = st.date_input("Date", value=date.today(), key="round_date")
    ss.hole = st.number_input("Hole", 1, 18, int(ss.hole))
    st.slider("Cold-air correction (% per °F below 70)", 0.05, 0.30, 0.12, 0.01, key="temp_coef",
              help="Plays-like distance rises by this much for every degree below 70°F. "
                   "Raise it for winter if cold-day shots keep coming up short.")
    st.divider()
    if CFG:
        st.caption(f"☁️ GitHub sync on · {CFG['repo']}")
        if st.button("Sync now", use_container_width=True) and not ss.data.empty:
            try:
                gh_save(CFG, ss.data, "Manual sync")
                st.success("Synced")
            except Exception as e:
                st.error(f"Sync failed: {e}")
    else:
        st.caption("GitHub sync off (add secrets to enable)")
    up = st.file_uploader("Load previous data (CSV)", type="csv")
    if up is not None and not ss.loaded:
        ss.data = pd.concat([ss.data, pd.read_csv(up)], ignore_index=True)
        ss.loaded = True
        st.success("Loaded")
    if not ss.data.empty:
        st.download_button("⬇️ Download all data", ss.data.to_csv(index=False),
                           "golf_shots.csv", "text/csv")

st.title("⛳ Strokes Gained")
W = None
tab_log, tab_caddie, tab_stats = st.tabs(["Log hole", "Caddie", "Stats"])
if ss.get("flash"):
    st.info(ss.pop("flash"))

# ---------------------------------------------------------------------------
# LOG TAB
# ---------------------------------------------------------------------------
with tab_log:
    loc = streamlit_geolocation()
    lat, lon, acc = loc.get("latitude"), loc.get("longitude"), loc.get("accuracy")
    if lat is not None:
        st.success(f"GPS fix ±{acc:.0f} m" if acc else "GPS fix")
    else:
        st.warning("No GPS fix. Tap the button above and allow location.")
    W = fetch_weather(round(lat, 2), round(lon, 2)) if lat is not None else None
    if W:
        st.caption(f"🌤 {W['temp_f']:.0f}°F · wind {W['wind_mph']:.0f} mph (saved with each shot)")

    par = st.radio("Par", [3, 4, 5], index=1, horizontal=True, key=f"par{ss.hole}")

    k = len(ss.shots)
    st.subheader(f"Hole {ss.hole} · shot {k + 1}")

    lie = st.radio("Lie", LIES, index=0 if k == 0 else 1, horizontal=True, key=f"lie{k}")

    feet, man_yds = 0.0, 0.0
    if lie == "Green":
        feet = st.number_input("Putt distance (feet)", 1.0, 150.0, 15.0, step=1.0, key=f"ft{k}")
        club = "Putter"
    else:
        man_yds = st.number_input("Yards in (to pin)", 0, 700, 0, step=1, key=f"my{k}",
                                  help="Rangefinder, watch or yardage book. Leave 0 to use GPS "
                                       "(needs the pin set when you finish the hole).")
        club_opts = [c for c in CLUBS if c != "Putter"]
        club = st.selectbox("Club", club_opts,
                            index=0 if k == 0 else club_opts.index("7i"), key=f"club{k}")
    pen = st.number_input("Penalty strokes", 0, 3, 0, key=f"pen{k}")
    wind_rel = st.radio("Wind on this shot", WIND_OPTS, horizontal=True, key="wind_rel_sel")
    finish = None
    if lie == "Green" and ss.shots and ss.shots[-1]["lie"] != "Green":
        finish = st.radio("Approach finished", ["Not sure", "Short", "Pin high", "Long"],
                          horizontal=True, key=f"fin{k}",
                          help="Short/long vs the pin. Your putt distance is used as the miss distance.")

    if st.button("➕ Add shot", use_container_width=True, type="primary"):
        if lie != "Green" and man_yds == 0 and lat is None:
            st.error("Enter yards in, or get a GPS fix.")
        else:
            if finish in ("Short", "Pin high", "Long"):
                ss.shots[-1]["finish"] = finish
            ss.shots.append(dict(lie=lie, club=club, lat=lat, lon=lon, feet=feet,
                                 man_yds=float(man_yds), pen=int(pen), wind_rel=wind_rel,
                                 temp_f=W["temp_f"] if W else None,
                                 wind_mph=W["wind_mph"] if W else None,
                                 wind_deg=W["wind_deg"] if W else None,
                                 humidity=W["humidity"] if W else None))
            st.rerun()

    if ss.shots:
        show = pd.DataFrame([{
            "#": i + 1, "lie": s["lie"], "club": s["club"],
            "dist": f"{s['feet']:.0f} ft" if s["lie"] == "Green"
            else (f"{s['man_yds']:.0f} yd" if s["man_yds"] else "GPS"),
            "pen": s["pen"]} for i, s in enumerate(ss.shots)])
        st.dataframe(show, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        if c1.button("↩️ Undo last", use_container_width=True):
            ss.shots.pop()
            st.rerun()
        if c2.button("📍 Set pin = my GPS", use_container_width=True):
            if lat is None:
                st.error("No GPS fix.")
            else:
                ss.pin = (lat, lon)
        if ss.pin:
            st.info("Pin saved (used for any shot with no yards entered).")

        if st.button("✅ Finish hole (ball in cup)", use_container_width=True):
            rows, ok = [], True
            for i, s in enumerate(ss.shots):
                if s["lie"] == "Green":
                    dist, unit = s["feet"], "ft"
                elif s["man_yds"] > 0:
                    dist, unit = s["man_yds"], "yd"
                elif s["lat"] is not None and ss.pin:
                    dist, unit = haversine_yds(s["lat"], s["lon"], ss.pin[0], ss.pin[1]), "yd"
                else:
                    st.error(f"Shot {i + 1}: no distance. Set the pin with GPS or enter yards.")
                    ok = False
                    break
                rows.append(dict(s, dist=dist, unit=unit))

            if ok:
                n = len(rows)
                score = n + sum(r["pen"] for r in rows)
                putts = sum(1 for r in rows if r["lie"] == "Green")
                first_green = next((i for i, r in enumerate(rows) if r["lie"] == "Green"), None)
                gir = 0
                if first_green is not None:
                    before = first_green + sum(rows[j]["pen"] for j in range(first_green))
                    gir = int(before <= par - 2)

                out = []
                for i, r in enumerate(rows):
                    nxt = rows[i + 1] if i + 1 < n else None
                    start = expected(r["lie"], r["dist"])
                    end = expected(nxt["lie"], nxt["dist"]) if nxt else 0.0
                    sg = start - end - 1 - r["pen"]

                    if r["lie"] == "Green":
                        cat = "Putting"
                    elif i == 0 and par > 3:
                        cat = "Off the Tee"
                    elif r["dist"] <= 30:
                        cat = "Around the Green"
                    else:
                        cat = "Approach"

                    hit = np.nan
                    if r["lie"] != "Green" and r["pen"] == 0:
                        yin = to_yds(r["dist"], r["unit"])
                        fin = r.get("finish")
                        if nxt is None:
                            hit = yin
                        elif nxt["lie"] == "Green" and fin in ("Short", "Pin high", "Long"):
                            miss = to_yds(nxt["dist"], nxt["unit"])
                            hit = yin - miss if fin == "Short" else (yin + miss if fin == "Long" else yin)
                        elif nxt["lie"] != "Green" and r["lat"] is not None and nxt["lat"] is not None:
                            hit = haversine_yds(r["lat"], r["lon"], nxt["lat"], nxt["lon"])
                        else:
                            hit = max(to_yds(r["dist"], r["unit"]) - to_yds(nxt["dist"], nxt["unit"]), 0)

                    out.append(dict(
                        date=str(round_date), course=course, hole=ss.hole, par=par, shot=i + 1,
                        lie=r["lie"], club=r["club"], distance=round(r["dist"], 1), unit=r["unit"],
                        ends_in=nxt["lie"] if nxt else "Holed", hit_yds=round(hit, 1) if not np.isnan(hit) else np.nan,
                        penalty=r["pen"], category=cat, sg=round(sg, 3),
                        finish=r.get("finish"), temp_f=r.get("temp_f"), wind_mph=r.get("wind_mph"),
                        wind_deg=r.get("wind_deg"), humidity=r.get("humidity"), wind_rel=r.get("wind_rel"),
                        lat=r["lat"], lon=r["lon"], gir=gir, putts=putts, score=score))

                ss.data = pd.concat([ss.data, pd.DataFrame(out)], ignore_index=True)
                msg = f"Hole {ss.hole}: {score} strokes, {putts} putts, SG {sum(o['sg'] for o in out):+.2f}"
                if CFG:
                    try:
                        gh_save(CFG, ss.data, f"{course} {round_date} hole {ss.hole}")
                        msg += " · ☁️ saved"
                    except Exception as e:
                        msg += f" · ⚠️ GitHub save failed ({e}). Use Download in the sidebar."
                ss.flash = msg
                ss.shots, ss.pin = [], None
                ss.hole = min(18, ss.hole + 1)
                st.rerun()

# ---------------------------------------------------------------------------
# CADDIE TAB
# ---------------------------------------------------------------------------
with tab_caddie:
    tbl = club_table(ss.data, ss.get("baseline"), ss.get("base_mode", "total"))
    n_own = int((tbl["source"] == "your data").sum())
    n_fs = int((tbl["source"] == "FlightScope").sum())
    st.caption(f"Distances: {n_own} club(s) from your rounds, {n_fs} from FlightScope, "
               f"{len(tbl) - n_own - n_fs} generic defaults. A club switches to your round data after 3 shots.")

    c1, c2, c3 = st.columns(3)
    front = c1.number_input("Front", 0, 600, 0, key="cad_f")
    mid = c2.number_input("Middle", 0, 600, 0, key="cad_m")
    back = c3.number_input("Back", 0, 600, 0, key="cad_b")
    pin_pos = st.select_slider("Pin position", ["Front", "Middle", "Back"], value="Middle", key="cad_pin")
    play_to = st.radio("Play to", ["Pin", "Middle"], horizontal=True, key="cad_play")

    have_fb = front > 0 and back > 0
    if mid == 0 and have_fb:
        mid = (front + back) / 2
    pct = {"Front": 0.25, "Middle": 0.5, "Back": 0.75}[pin_pos]
    pin_num = front + pct * (back - front) if have_fb else mid
    target = pin_num if play_to == "Pin" else mid

    tdef = float(W["temp_f"]) if W else 70.0
    wdef = float(W["wind_mph"]) if W else 0.0
    t1, t2 = st.columns(2)
    temp = t1.number_input("Temp °F", -10.0, 120.0, tdef, step=1.0, key=f"cad_t_{int(tdef)}")
    wind = t2.number_input("Wind mph", 0.0, 60.0, wdef, step=1.0, key=f"cad_w_{int(wdef)}")
    rel = st.radio("Wind direction vs shot", WIND_OPTS, horizontal=True, key="cad_rel")

    if target <= 0:
        st.info("Enter at least the middle yardage.")
    else:
        f_wind = plays_factor(70, wind, rel)
        f_temp = plays_factor(temp, 0, "Calm")
        plays = target * f_wind * f_temp
        st.metric("Plays like", f"{plays:.0f} yds",
                  f"{plays - target:+.0f} vs {target:.0f}", delta_color="off")
        bits = []
        if f_wind != 1:
            bits.append(f"wind {100 * (f_wind - 1):+.0f}%")
        if f_temp != 1:
            bits.append(f"temp {100 * (f_temp - 1):+.0f}%")
        if bits:
            st.caption(" · ".join(bits))
        if rel == "Cross":
            st.caption("Crosswind: no distance change applied. Aim for it.")

        best, second = recommend(tbl, plays)
        st.success(f"**{best['club']}** · {best['avg_yds']} yds "
                   f"({best['source']}, {int(best['shots'])} shots)")
        if second is not None and abs(second["avg_yds"] - plays) <= 15:
            st.caption(f"Between clubs: also consider {second['club']} ({second['avg_yds']} yds).")
        if plays > tbl["avg_yds"].max() + 10:
            st.warning("Beyond your longest club.")
        if plays < tbl["avg_yds"].min() - 10:
            st.warning("Inside full-wedge range. Think partial wedge.")

    with st.expander("My club distances vs FlightScope"):
        st.dataframe(compare_view(tbl), use_container_width=True, hide_index=True)
        st.caption("On course = GPS total distance hit, weather-adjusted to 70°F. Implied roll = that minus your "
                   "FlightScope carry. Each GPS point is only good to about 3-5 m, so one shot can be off by ~10 yds; "
                   "trust the roll after 3+ shots per club. Negative roll usually means a bad lie, wind or a mishit.")

    with st.expander("FlightScope baseline"):
        ss.base_mode = st.radio("Use FlightScope", ["total", "carry"], horizontal=True,
                                index=0 if ss.get("base_mode", "total") == "total" else 1,
                                help="Your on-course distances include roll, so 'total' is the fairer comparison.")
        st.number_input("Temperature on the range day (°F)", -10.0, 120.0, 70.0, step=1.0, key="base_temp",
                        help="Cold range sessions fly shorter. This brings them back to a 70°F equivalent.")
        st.slider("Roll allowance (x default)", 0.0, 2.0, 1.0, 0.1, key="roll_scale",
                  help="FlightScope gives carry only, so total is estimated as carry plus a typical roll "
                       "(about 4 yds for mid irons, 20 for driver). 0 turns it off; raise it for firm summer ground.")
        how = st.radio("Add distances", ["Upload export", "Type them in"], horizontal=True)

        def _save_baseline():
            if CFG and ss.get("baseline") is not None:
                try:
                    gh_save(dict(CFG, path="data/flightscope_clubs.csv"), ss.baseline,
                            "Update FlightScope baseline")
                    st.success("Saved to GitHub")
                except Exception as e:
                    st.warning(f"Kept for this session, but GitHub save failed: {e}")

        if how == "Upload export":
            f = st.file_uploader("FlightScope export (CSV or Excel)", type=["csv", "xlsx"], key="fs_up")
            if f is not None:
                raw = pd.read_excel(f) if f.name.lower().endswith("xlsx") else pd.read_csv(f)
                cols = list(raw.columns)

                def _guess(word, opts):
                    for i, c in enumerate(opts):
                        if word in str(c).lower():
                            return i
                    return 0

                club_col = st.selectbox("Club column", cols, index=_guess("club", cols))
                carry_col = st.selectbox("Carry column", cols, index=_guess("carry", cols))
                tot_opts = ["(none)"] + cols
                total_col = st.selectbox("Total column", tot_opts, index=_guess("total", tot_opts))
                meters = st.checkbox("Values are in meters")
                if st.button("Use this data", use_container_width=True):
                    g, unmapped = build_baseline(raw, club_col, carry_col, total_col, meters)
                    ss.baseline = g
                    st.success(f"Loaded {len(g)} clubs")
                    if unmapped:
                        st.caption("Skipped (club name not recognised): " + ", ".join(unmapped))
                    _save_baseline()
        else:
            full = pd.DataFrame({"club": list(DEFAULT_CLUB_YDS)})
            cur = ss.baseline
            if cur is not None and len(cur):
                full = full.merge(cur[["club", "carry", "total"]], on="club", how="left")
            else:
                full["carry"], full["total"] = np.nan, np.nan
            edited = st.data_editor(full, hide_index=True, disabled=["club"],
                                    use_container_width=True, key="fs_edit")
            if st.button("Save typed distances", use_container_width=True):
                ss.baseline = edited.dropna(subset=["carry", "total"], how="all").reset_index(drop=True)
                st.success("Saved")
                _save_baseline()
        if ss.baseline is not None and len(ss.baseline):
            st.download_button("⬇️ Download baseline", ss.baseline.to_csv(index=False),
                               "flightscope_clubs.csv", "text/csv")

# ---------------------------------------------------------------------------
# STATS TAB
# ---------------------------------------------------------------------------
with tab_stats:
    df = ss.data.copy()
    if df.empty:
        st.info("Finish a hole to see stats.")
    else:
        for c in ["ends_in", "hit_yds", "gir", "putts", "score"]:
            if c not in df.columns:
                df[c] = np.nan
        rounds = sorted(df["date"].astype(str).unique())
        pick = st.multiselect("Rounds", rounds, default=rounds)
        d = df[df["date"].astype(str).isin(pick)]
        holes = d.drop_duplicates(["date", "hole"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total SG", f"{d['sg'].sum():+.2f}")
        c2.metric("Holes", len(holes))
        c3.metric("Putts / hole", f"{pd.to_numeric(holes['putts'], errors='coerce').mean():.2f}")

        c4, c5, c6 = st.columns(3)
        tee = d[d["category"] == "Off the Tee"]
        c4.metric("Fairways", f"{(tee['ends_in'] == 'Fairway').mean() * 100:.0f}%" if len(tee) else "–")
        c5.metric("GIR", f"{pd.to_numeric(holes['gir'], errors='coerce').mean() * 100:.0f}%")
        c6.metric("3-putts", int((pd.to_numeric(holes["putts"], errors="coerce") >= 3).sum()))

        st.subheader("Strokes gained")
        cat = d.groupby("category")["sg"].sum().reindex(
            ["Off the Tee", "Approach", "Around the Green", "Putting"]).fillna(0)
        st.bar_chart(cat)

        st.subheader("Putting by distance")
        pt = d[d["lie"] == "Green"].copy()
        if len(pt):
            pt["range"] = pd.cut(pt["distance"], [0, 5, 10, 20, 200],
                                 labels=["<5 ft", "5-10 ft", "10-20 ft", "20+ ft"])
            tbl = pt.groupby("range", observed=True).agg(
                putts=("sg", "count"),
                make_pct=("ends_in", lambda s: round((s == "Holed").mean() * 100)),
                avg_sg=("sg", lambda s: round(s.mean(), 2)))
            st.dataframe(tbl, use_container_width=True)

        st.subheader("Club distances (weather-adjusted)")
        st.dataframe(compare_view(club_table(d, ss.get("baseline"), ss.get("base_mode", "total"))),
                     use_container_width=True, hide_index=True)

        st.subheader("By hole")
        ph = d.groupby(["date", "hole"]).agg(par=("par", "first"), score=("score", "first"),
                                             putts=("putts", "first"), sg=("sg", "sum")).reset_index()
        ph["sg"] = ph["sg"].round(2)
        st.dataframe(ph, use_container_width=True, hide_index=True)
        with st.expander("All shots"):
            st.dataframe(d, use_container_width=True, hide_index=True)
