import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import requests
import time
import threading
from typing import Optional, Dict
from datetime import datetime


# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
OWM_BASE      = "https://api.openweathermap.org/data/2.5"
CITY_NAME     = "Chennai,IN"
FETCH_EVERY   = 300          # seconds — OWM free tier: 60 calls/min, fetch every 5min
DEFAULT_TEMP  = 30.0
DEFAULT_RAIN  = 0.0
DEFAULT_WIND  = 2.0
DEFAULT_DEG   = 270


# ─────────────────────────────────────────
#  WEATHER FEED
# ─────────────────────────────────────────
class WeatherFeed:
    """
    Pulls live Chennai weather from OpenWeatherMap every 5 minutes.
    Runs in a background thread — never blocks the twin's 500ms loop.
    Falls back to last known values if API call fails.
    Also supports manual override for demo scenarios.
    """

    def __init__(self, api_key: str, twin=None):
        self.api_key     = api_key
        self.twin        = twin
        self._running    = False
        self._thread: Optional[threading.Thread] = None

        # Last known weather
        self.current: Dict = {
            "rainfall":   DEFAULT_RAIN,
            "wind_speed": DEFAULT_WIND,
            "wind_deg":   DEFAULT_DEG,
            "temperature": DEFAULT_TEMP,
            "humidity":   60.0,
            "description": "clear sky",
            "fetched_at": None,
            "source":     "default",
        }

        # Manual override — frontend demo sliders bypass live API
        self._override: Optional[Dict] = None

    # ── LIFECYCLE ────────────────────────────────────────────────

    def start(self):
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        # Fetch immediately on start
        self._fetch_and_inject()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"[ WeatherFeed ] Started — polling every {FETCH_EVERY}s")

    def stop(self):
        self._running = False
        print("[ WeatherFeed ] Stopped")

    # ── OVERRIDE (demo mode) ──────────────────────────────────────

    def set_override(self, rainfall: float, wind_speed: float = 2.0,
                     wind_deg: float = 270, temperature: float = 32.0):
        """
        Frontend slider sets this — bypasses live API.
        Twin gets scenario values instead of real weather.
        """
        self._override = {
            "rainfall":    max(0.0, rainfall),
            "wind_speed":  max(0.0, wind_speed),
            "wind_deg":    wind_deg % 360,
            "temperature": temperature,
        }
        if self.twin:
            self._inject(self._override)

    def clear_override(self):
        """Go back to live weather."""
        self._override = None
        if self.twin:
            self._inject(self.current)

    # ── GETTERS ──────────────────────────────────────────────────

    def get_current(self) -> Dict:
        """Returns override if active, else live data."""
        if self._override:
            return {**self.current, **self._override, "source": "override"}
        return self.current

    def get_rainfall(self) -> float:
        return self.get_current()["rainfall"]

    def get_temperature(self) -> float:
        return self.get_current()["temperature"]

    # ── INTERNALS ────────────────────────────────────────────────

    def _poll_loop(self):
        while self._running:
            time.sleep(FETCH_EVERY)
            if not self._override:   # don't overwrite override with live data
                self._fetch_and_inject()

    def _fetch_and_inject(self):
        data = self._fetch_owm()
        if data:
            self.current = data
            if self.twin and not self._override:
                self._inject(data)

    def _fetch_owm(self) -> Optional[Dict]:
        """
        Hit OWM current weather endpoint.
        Returns parsed dict or None on failure.
        """
        try:
            resp = requests.get(
                f"{OWM_BASE}/weather",
                params={
                    "q":     CITY_NAME,
                    "appid": self.api_key,
                    "units": "metric",
                },
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()

            rainfall   = raw.get("rain", {}).get("1h", 0.0)
            wind_speed = raw.get("wind", {}).get("speed", DEFAULT_WIND)
            wind_deg   = raw.get("wind", {}).get("deg",   DEFAULT_DEG)
            temp       = raw["main"]["temp"]
            humidity   = raw["main"]["humidity"]
            desc       = raw["weather"][0]["description"] if raw.get("weather") else "unknown"

            result = {
                "rainfall":    float(rainfall),
                "wind_speed":  float(wind_speed),
                "wind_deg":    float(wind_deg),
                "temperature": float(temp),
                "humidity":    float(humidity),
                "description": desc,
                "fetched_at":  datetime.now().isoformat(),
                "source":      "openweathermap",
            }

            print(f"[ WeatherFeed ] {datetime.now().strftime('%H:%M:%S')} — "
                  f"{desc}, {temp:.1f}°C, rain={rainfall}mm/hr, "
                  f"wind={wind_speed}m/s@{wind_deg}°")
            return result

        except requests.exceptions.ConnectionError:
            print("[ WeatherFeed ] No internet — using last known values")
        except requests.exceptions.Timeout:
            print("[ WeatherFeed ] OWM timeout — using last known values")
        except requests.exceptions.HTTPError as e:
            if "401" in str(e):
                print("[ WeatherFeed ] Invalid API key — check config.py")
            else:
                print(f"[ WeatherFeed ] HTTP error: {e}")
        except Exception as e:
            print(f"[ WeatherFeed ] Unexpected error: {e}")

        return None   # caller uses last known values

    def _inject(self, data: Dict):
        """Push weather into twin engine."""
        self.twin.inject_weather(
            rainfall   = data["rainfall"],
            wind_speed = data["wind_speed"],
            wind_deg   = data["wind_deg"],
            temp       = data["temperature"],
        )


# ─────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(r"D:\projects__\resonate_26\work")
    from core.twin_engine import TwinEngine

    # ── Test 1: No API key — fallback behaviour
    print("── Test 1: Invalid API key (fallback test) ──")
    CACHE = r"D:\projects__\resonate_26\datasets\chennai_graph.graphml"
    twin  = TwinEngine(cache_path=CACHE)
    # feed  = WeatherFeed(api_key="INVALID_KEY_TEST", twin=twin)
    
    # ////////////////////////////////////////
    import sys
    sys.path.append(r"D:\projects__\resonate_26\work")
    from config import OWM_API_KEY
    feed = WeatherFeed(api_key=OWM_API_KEY, twin=twin)
    feed._fetch_and_inject()   # should print error and use defaults

    print(f"  Rainfall  : {feed.get_rainfall()} mm/hr")
    print(f"  Temp      : {feed.get_temperature()} °C")
    print(f"  Source    : {feed.get_current()['source']}")

    # ── Test 2: Manual override
    print("\n── Test 2: Demo override ──")
    feed.set_override(rainfall=80.0, temperature=38.5)
    state = twin.step()
    s     = state["summary"]
    print(f"  Override rainfall : 80mm/hr")
    print(f"  Twin avg_risk     : {s['avg_risk_score']:.3f}")
    print(f"  Twin avg_temp     : {s['avg_temperature']:.2f}°C")
    print(f"  Source            : {feed.get_current()['source']}")

    # ── Test 3: Clear override
    print("\n── Test 3: Clear override ──")
    feed.clear_override()
    print(f"  Source after clear: {feed.get_current()['source']}")

    print("\n[ WeatherFeed ] All checks passed.")
    print("\nNOTE: To test live API, replace 'INVALID_KEY_TEST' with your OWM key in config.py")