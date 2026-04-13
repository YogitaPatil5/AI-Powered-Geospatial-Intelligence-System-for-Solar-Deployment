import math
import time
import random
import requests


def safe_get(url, params):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if attempt < 2:
                time.sleep(3)
    raise ConnectionError(f"Failed: {url}")


def safe_post(url, query_text):
    for attempt in range(2):
        try:
            r = requests.post(url, data={"data": query_text}, timeout=25)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if attempt < 1:
                time.sleep(8 + random.uniform(1, 3))
    return None


def fetch_nasa_power(lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,T2M,CLOUD_AMT,ALLSKY_KT",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }

    res = safe_get(url, params).json()["properties"]["parameter"]

    def avg(k):
        d = res[k]
        if "ANN" in d and d["ANN"] not in (-999, -999.0):
            return round(float(d["ANN"]), 4)
        vals = [float(v) for m, v in d.items() if m != "ANN" and v not in (-999, -999.0, None)]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "ghi": avg("ALLSKY_SFC_SW_DWN"),
        "dni": avg("ALLSKY_SFC_SW_DNI"),
        "temperature": avg("T2M"),
        "cloud_pct": avg("CLOUD_AMT"),
        "clearness": avg("ALLSKY_KT"),
    }


def fetch_openmeteo(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": "precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean",
        "timezone": "auto",
    }

    d = safe_get(url, params).json().get("daily", {})

    def avg(k):
        vals = [v for v in d.get(k, []) if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "wind_speed": avg("windspeed_10m_max"),
        "precipitation": avg("precipitation_sum"),
        "humidity": avg("relative_humidity_2m_mean"),
    }


def fetch_elevation(lat, lon):
    offset = 0.001
    pts = [(lat, lon), (lat+offset, lon), (lat-offset, lon),
           (lat, lon+offset), (lat, lon-offset)]

    locs = "|".join(f"{a},{b}" for a, b in pts)
    url = "https://api.opentopodata.org/v1/srtm90m"

    res = safe_get(url, {"locations": locs}).json().get("results", [])

    if len(res) < 5:
        return {"elevation": 200.0, "slope": 5.0, "aspect": 180.0}

    c, n, s, e, w = [r.get("elevation") or 0.0 for r in res]

    d = offset * 111000
    dz_dx = (e - w) / (2 * d)
    dz_dy = (n - s) / (2 * d)

    slope = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = (math.degrees(math.atan2(dz_dx, dz_dy)) + 360) % 360

    return {
        "elevation": round(c, 2),
        "slope": round(slope, 2),
        "aspect": round(aspect, 2),
    }


LAND_SCORE_MAP = {
    "barren": 1.0, "scrub": 0.85, "grassland": 0.8, "farmland": 0.65,
    "meadow": 0.65, "industrial": 0.4, "residential": 0.15,
    "commercial": 0.15, "forest": 0.1, "water": 0.0, "unknown": 0.5,
}

NDVI_MAP = {
    "barren": 0.05, "scrub": 0.25, "grassland": 0.35, "farmland": 0.45,
    "forest": 0.75, "residential": 0.2, "commercial": 0.1,
    "industrial": 0.1, "water": 0.0, "unknown": 0.3,
}


def dist_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def fetch_osm(lat, lon, radius=5000):
    url = "https://overpass-api.de/api/interpreter"

    land = "unknown"
    try:
        q = f"""
        [out:json];
        (way["landuse"](around:{radius},{lat},{lon});
         way["natural"](around:{radius},{lat},{lon}););
        out center 5;
        """
        r = safe_post(url, q)
        if r:
            el = r.json().get("elements", [])
            if el:
                t = el[0].get("tags", {})
                land = (t.get("landuse") or t.get("natural") or "unknown").lower()
    except:
        pass

    time.sleep(2)

    road_km = 5.0
    try:
        q = f'[out:json];way["highway"](around:{radius},{lat},{lon});out center 5;'
        r = safe_post(url, q)
        if r:
            d = []
            for e in r.json().get("elements", []):
                c = e.get("center", {})
                if c:
                    d.append(dist_km(lat, lon, c["lat"], c["lon"]))
            if d:
                road_km = round(min(d), 3)
    except:
        pass

    time.sleep(2)

    grid_km = 10.0
    try:
        q = f"""
        [out:json];
        (way["power"="line"](around:{radius},{lat},{lon});
         node["power"="substation"](around:{radius},{lat},{lon}););
        out center 5;
        """
        r = safe_post(url, q)
        if r:
            d = []
            for e in r.json().get("elements", []):
                c = e.get("center", {})
                la = e.get("lat") or c.get("lat")
                lo = e.get("lon") or c.get("lon")
                if la and lo:
                    d.append(dist_km(lat, lon, la, lo))
            if d:
                grid_km = round(min(d), 3)
    except:
        pass

    return {
        "land_use": land,
        "land_score": LAND_SCORE_MAP.get(land, 0.5),
        "ndvi": NDVI_MAP.get(land, 0.3),
        "road_km": road_km,
        "grid_km": grid_km,
    }


def collect_all(lat, lon, name=""):
    res = {"lat": lat, "lon": lon, "name": name}

    try:
        res.update(fetch_nasa_power(lat, lon))
    except:
        res.update({"ghi": 4.5, "dni": 4.0, "temperature": 27.0, "cloud_pct": 30.0, "clearness": 0.55})

    try:
        res.update(fetch_openmeteo(lat, lon))
    except:
        res.update({"wind_speed": 10.0, "precipitation": 2.0, "humidity": 40.0})

    try:
        res.update(fetch_elevation(lat, lon))
    except:
        res.update({"elevation": 300.0, "slope": 5.0, "aspect": 180.0})

    try:
        res.update(fetch_osm(lat, lon))
    except:
        res.update({"land_use": "unknown", "land_score": 0.5, "ndvi": 0.3, "road_km": 5.0, "grid_km": 10.0})

    return res


if __name__ == "__main__":
    lat, lon = 26.9124, 70.9090
    data = collect_all(lat, lon, "Jaisalmer")
    for k, v in data.items():
        print(k, ":", v)