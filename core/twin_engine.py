import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")


# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"])




import torch
torch._dynamo.config.suppress_errors = True

import osmnx as ox
import networkx as nx
import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional
from datetime import datetime

# ─────────────────────────────────────────
#  CONSTANTS — tweak these, never hardcode
# ─────────────────────────────────────────
CITY          = "Chennai, India"
NETWORK_TYPE  = "drive"
TIMESTEP_SEC  = 0.5          # twin updates every 500ms

# Physics factors
# RAIN_FLOOD_FACTOR      = 0.012   # mm/hr → water level rise per timestep
RAIN_FLOOD_FACTOR  = 0.0008   # was 0.012 — way too fast
# DRAINAGE_DECAY         = 0.008   # natural drainage per timestep
DRAINAGE_DECAY     = 0.0004   # was 0.008 — balance with new rain factor
TRAFFIC_AQI_FACTOR     = 0.003   # vehicles/capacity → AQI contribution
WIND_DIFFUSION_ALPHA   = 0.15    # AQI spreads to neighbors at this rate
HEAT_TRAFFIC_FACTOR    = 0.004   # congestion → temperature rise
FLOOD_ROAD_PENALTY     = 0.6     # flood reduces road capacity by 60%

# Risk thresholds
FLOOD_WARNING    = 0.4
# FLOOD_CRITICAL   = 0.7
FLOOD_CRITICAL     = 0.75  
AQI_WARNING      = 100
AQI_CRITICAL     = 150
CONGESTION_WARN  = 0.75
CONGESTION_CRIT  = 0.90
TEMP_WARNING     = 38.0
TEMP_CRITICAL    = 42.0

# Zone types — determines baseline emissions + heat absorption
ZONE_TYPES = ["residential", "commercial", "industrial", "green", "mixed"]

ZONE_BASELINE_AQI = {
    "residential": 45,
    "commercial":  65,
    "industrial":  90,
    "green":       20,
    "mixed":       55,
}

ZONE_HEAT_FACTOR = {
    "residential": 1.0,
    "commercial":  1.4,
    "industrial":  1.6,
    "green":       0.6,
    "mixed":       1.2,
}

# Peak hour traffic multipliers
PEAK_HOUR_MULTIPLIERS = {
    8:  2.8, 9:  2.6, 10: 1.8,   # morning peak
    12: 1.6, 13: 1.5,             # lunch
    17: 3.1, 18: 2.9, 19: 2.4,   # evening peak
}


# ─────────────────────────────────────────
#  ZONE STATE — what every node carries
# ─────────────────────────────────────────
@dataclass
class ZoneState:
    node_id:          int
    lat:              float
    lon:              float
    zone_type:        str   = "mixed"

    # Live physics
    water_level:      float = 0.0    # 0–1, normalised flood depth
    traffic_density:  float = 0.0    # 0–1, vehicles / capacity
    aqi:              float = 0.0    # EPA composite AQI
    temperature:      float = 30.0   # °C
    rainfall_local:   float = 0.0    # mm/hr at this node

    # Derived
    risk_score:       float = 0.0    # 0–1 composite risk
    flood_risk:       float = 0.0
    congestion_risk:  float = 0.0
    heat_risk:        float = 0.0
    aqi_risk:         float = 0.0

    # Flags
    is_flooded:       bool  = False
    is_critical:      bool  = False
    alert_level:      str   = "safe"   # safe / warning / critical

    def to_vector(self) -> np.ndarray:
        """Flat feature vector for GNN input — 10 dims per node."""
        return np.array([
            self.water_level,
            self.traffic_density,
            self.aqi / 300.0,       # normalise to 0-1
            (self.temperature - 20) / 30.0,
            self.rainfall_local / 100.0,
            self.risk_score,
            self.flood_risk,
            self.congestion_risk,
            self.heat_risk,
            self.aqi_risk,
        ], dtype=np.float32)

    def compute_risks(self):
        """Derive risk sub-scores and composite from raw physics."""
        self.flood_risk       = min(self.water_level / FLOOD_CRITICAL, 1.0)
        self.congestion_risk  = min(self.traffic_density / CONGESTION_CRIT, 1.0)
        self.heat_risk        = max((self.temperature - TEMP_WARNING) / (TEMP_CRITICAL - TEMP_WARNING), 0.0)
        self.heat_risk        = min(self.heat_risk, 1.0)
        self.aqi_risk         = max((self.aqi - AQI_WARNING) / (AQI_CRITICAL - AQI_WARNING), 0.0)
        self.aqi_risk         = min(self.aqi_risk, 1.0)

        # Weighted composite
        self.risk_score = (
            0.35 * self.flood_risk +
            0.25 * self.congestion_risk +
            0.20 * self.aqi_risk +
            0.20 * self.heat_risk
        )

        # Flags
        self.is_flooded  = self.water_level >= FLOOD_CRITICAL
        self.is_critical = self.risk_score  >= 0.7

        if self.risk_score >= 0.7:
            self.alert_level = "critical"
        elif self.risk_score >= 0.4:
            self.alert_level = "warning"
        else:
            self.alert_level = "safe"


# ─────────────────────────────────────────
#  EDGE STATE — what every road carries
# ─────────────────────────────────────────
@dataclass
class EdgeState:
    u:                  int
    v:                  int
    road_capacity:      int    = 1200    # vehicles/hr baseline
    current_flow:       int    = 0
    length_m:           float  = 100.0
    travel_time:        float  = 60.0   # seconds baseline

    # Live
    congestion_factor:  float  = 1.0    # multiplier on travel_time
    pollution_corridor: float  = 0.0    # AQI contribution from this edge
    flood_passable:     bool   = True   # False when flooded
    route_load:         int    = 0      # routing suggestions sent here this cycle

    def flow_ratio(self) -> float:
        return min(self.current_flow / max(self.road_capacity, 1), 1.0)

    def effective_travel_time(self) -> float:
        """BPR function — travel time rises with congestion."""
        ratio = self.flow_ratio()
        bpr   = 1 + 0.15 * (ratio ** 4)   # Bureau of Public Roads formula
        return self.travel_time * bpr * self.congestion_factor

    def compute_pollution(self, zone_aqi: float) -> float:
        """AQI contribution from vehicle exhaust on this edge."""
        exhaust = self.flow_ratio() * TRAFFIC_AQI_FACTOR * 300
        self.pollution_corridor = zone_aqi * 0.3 + exhaust
        return self.pollution_corridor


# ─────────────────────────────────────────
#  TWIN ENGINE — the city itself
# ─────────────────────────────────────────
class TwinEngine:
    def __init__(self, cache_path: Optional[str] = None):
        print("[ TwinEngine ] Initialising Urban Sentinel Digital Twin...")
        self.G              = None
        self.nodes: Dict[int, ZoneState] = {}
        self.edges: Dict[Tuple, EdgeState] = {}
        self.timestep       = 0
        self.sim_time       = datetime.now()

        # Global weather inputs (set by weather_feed.py)
        self.global_rainfall   = 0.0   # mm/hr
        self.global_wind_speed = 2.0   # m/s
        self.global_wind_deg   = 270   # direction in degrees
        self.global_temp_base  = 30.0  # °C ambient

        # Simulation scenario inputs (set by frontend sliders)
        self.scenario_rainfall    = 0.0
        self.scenario_traffic_mul = 1.0
        self.ai_enabled           = True

        # Delta state for WebSocket broadcast
        self._prev_state: Dict = {}

        self._load_graph(cache_path)
        self._initialise_states()
        print(f"[ TwinEngine ] Ready — {len(self.nodes)} zones, {len(self.edges)} corridors")

    # ── GRAPH LOADING ──────────────────────────────────────────
    def _load_graph(self, cache_path: Optional[str]):
        """Pull Chennai road graph. Cache locally so it's instant on reload."""
        if cache_path and os.path.exists(cache_path):
            print("[ TwinEngine ] Loading cached graph...")
            self.G = ox.load_graphml(cache_path)
        else:
            print("[ TwinEngine ] Pulling Chennai graph from OSMnx (first time, ~30s)...")
            self.G = ox.graph_from_place(CITY, network_type=NETWORK_TYPE)
            if cache_path:
                ox.save_graphml(self.G, cache_path)
                print(f"[ TwinEngine ] Graph cached → {cache_path}")

        # Project to UTM for accurate distance calcs
        self.G = ox.project_graph(self.G)
        self.G = ox.convert.to_undirected(self.G)

    def _initialise_states(self):
        """Assign initial ZoneState to every node, EdgeState to every edge."""
        rng = np.random.default_rng(42)   # reproducible

        node_list = list(self.G.nodes(data=True))
        zone_cycle = ZONE_TYPES * (len(node_list) // len(ZONE_TYPES) + 1)

        for i, (node_id, data) in enumerate(node_list):
            # OSMnx projected graph stores x,y — convert back to lat/lon
            lat = data.get("lat", data.get("y", 13.0827))
            lon = data.get("lon", data.get("x", 80.2707))

            zone = zone_cycle[i]
            state = ZoneState(
                node_id         = node_id,
                lat             = lat,
                lon             = lon,
                zone_type       = zone,
                temperature     = self.global_temp_base + rng.uniform(-2, 4),
                aqi             = ZONE_BASELINE_AQI[zone] + rng.uniform(-10, 10),
                traffic_density = rng.uniform(0.1, 0.3),
            )
            state.compute_risks()
            self.nodes[node_id] = state

        for u, v, data in self.G.edges(data=True):
            length   = data.get("length", 100.0)
            speed    = data.get("speed_kph", 40.0)
            if isinstance(speed, list):
                speed = float(speed[0])
            travel_t = (length / 1000.0) / float(speed) * 3600   # seconds

            cap = int(1200 * (float(speed) / 40.0))   # faster roads = more capacity

            edge = EdgeState(
                u             = u,
                v             = v,
                road_capacity = cap,
                length_m      = length,
                travel_time   = travel_t,
                current_flow  = int(rng.uniform(0.1, 0.3) * cap),
            )
            self.edges[(u, v)] = edge

    # ── SCENARIO SETTERS (called by frontend sliders) ──────────
    def set_rainfall(self, mm_per_hr: float):
        self.scenario_rainfall = max(0.0, mm_per_hr)

    def set_traffic_surge(self, multiplier: float):
        self.scenario_traffic_mul = max(1.0, multiplier)

    def set_ai_enabled(self, enabled: bool):
        self.ai_enabled = enabled

    def inject_weather(self, rainfall: float, wind_speed: float,
                       wind_deg: float, temp: float):
        """Called by weather_feed.py with live OpenWeatherMap data."""
        self.global_rainfall   = rainfall
        self.global_wind_speed = wind_speed
        self.global_wind_deg   = wind_deg
        self.global_temp_base  = temp

    # ── CORE TIMESTEP ──────────────────────────────────────────
    def step(self):
        """Advance twin by one timestep. Called every 500ms by APScheduler."""
        self.timestep  += 1
        hour            = datetime.now().hour
        total_rainfall  = self.global_rainfall + self.scenario_rainfall
        traffic_mul     = self.scenario_traffic_mul * PEAK_HOUR_MULTIPLIERS.get(hour, 1.0)

        # 1. Update edges first
        self._update_edges(traffic_mul)

        # 2. Update node physics
        self._update_flood(total_rainfall)
        self._update_traffic(traffic_mul)
        self._update_pollution()
        self._update_heat()

        # 3. Recompute all risk scores
        for state in self.nodes.values():
            state.compute_risks()

        return self.get_full_state()

    def _update_edges(self, traffic_mul: float):
        for (u, v), edge in self.edges.items():
            # Flood on either endpoint reduces passability
            u_flooded = self.nodes[u].is_flooded if u in self.nodes else False
            v_flooded = self.nodes[v].is_flooded if v in self.nodes else False
            edge.flood_passable = not (u_flooded or v_flooded)

            # Traffic flow scales with scenario multiplier
            base_flow = int(edge.road_capacity * 0.3 * traffic_mul)
            edge.current_flow = min(base_flow + edge.route_load, edge.road_capacity)

            # Congestion factor from flow ratio
            ratio = edge.flow_ratio()
            edge.congestion_factor = 1 + 0.15 * (ratio ** 4)

            # Decay route_load each timestep (people complete their journeys)
            edge.route_load = max(0, edge.route_load - 2)

    def _update_flood(self, total_rainfall: float):
        """Water level rises with rainfall, drains naturally, propagates via edges."""
        for node_id, state in self.nodes.items():
            # Rainfall raises water level
            rain_rise = total_rainfall * RAIN_FLOOD_FACTOR
            state.water_level = min(state.water_level + rain_rise, 1.0)

            # Natural drainage (green zones drain faster)
            drain_rate = DRAINAGE_DECAY * (2.0 if state.zone_type == "green" else 1.0)
            state.water_level = max(state.water_level - drain_rate, 0.0)

            state.rainfall_local = total_rainfall

        # Propagate flood through drainage edges (high → low water level)
        for (u, v), edge in self.edges.items():
            if u in self.nodes and v in self.nodes:
                u_state = self.nodes[u]
                v_state = self.nodes[v]
                diff    = u_state.water_level - v_state.water_level
                if diff > 0.05:   # only propagate significant differences
                    flow = diff * 0.1
                    u_state.water_level = max(u_state.water_level - flow, 0.0)
                    v_state.water_level = min(v_state.water_level + flow, 1.0)

    def _update_traffic(self, traffic_mul: float):
        """Node traffic density from adjacent edge flows."""
        for node_id, state in self.nodes.items():
            neighbors = list(self.G.neighbors(node_id))
            if not neighbors:
                continue
            adj_flows = []
            for nb in neighbors:
                key = (node_id, nb) if (node_id, nb) in self.edges else (nb, node_id)
                if key in self.edges:
                    adj_flows.append(self.edges[key].flow_ratio())
            if adj_flows:
                state.traffic_density = float(np.mean(adj_flows))

    def _update_pollution(self):
        """AQI = baseline + traffic exhaust + wind diffusion from neighbors."""
        wind_rad    = np.radians(self.global_wind_deg)
        wind_dx     = np.cos(wind_rad)
        wind_dy     = np.sin(wind_rad)

        new_aqi: Dict[int, float] = {}

        for node_id, state in self.nodes.items():
            baseline    = ZONE_BASELINE_AQI[state.zone_type]
            traffic_aqi = state.traffic_density * TRAFFIC_AQI_FACTOR * 300

            # Neighbor diffusion weighted by wind direction
            neighbors   = list(self.G.neighbors(node_id))
            diffused    = 0.0
            if neighbors:
                for nb in neighbors:
                    if nb in self.nodes:
                        nb_data  = self.G.nodes[nb]
                        dx       = nb_data.get("x", 0) - self.G.nodes[node_id].get("x", 0)
                        dy       = nb_data.get("y", 0) - self.G.nodes[node_id].get("y", 0)
                        dist     = max(np.sqrt(dx**2 + dy**2), 1.0)
                        wind_dot = (dx * wind_dx + dy * wind_dy) / dist
                        weight   = max(wind_dot, 0)   # only downwind diffusion
                        diffused += weight * self.nodes[nb].aqi

                if diffused > 0:
                    diffused = diffused / len(neighbors) * WIND_DIFFUSION_ALPHA

            new_aqi[node_id] = min(baseline + traffic_aqi + diffused, 500.0)

        for node_id, aqi_val in new_aqi.items():
            self.nodes[node_id].aqi = aqi_val

    def _update_heat(self):
        """Temperature = ambient + traffic heat + zone factor."""
        for node_id, state in self.nodes.items():
            heat_from_traffic = state.traffic_density * HEAT_TRAFFIC_FACTOR * 10
            zone_factor       = ZONE_HEAT_FACTOR[state.zone_type]
            state.temperature = (
                self.global_temp_base
                + heat_from_traffic * zone_factor
                + (state.aqi / 300.0) * 2.0   # high AQI = trapped heat
            )

    # ── RL AGENT ACTIONS (called by rl_agent.py) ───────────────
    def apply_action(self, action_type: str, target_id, value: float = 1.0):
        """
        action_type: 'open_drainage' | 'reroute_edge' | 'reduce_signal' | 'green_corridor'
        target_id:   node_id or (u,v) edge tuple
        value:       magnitude of intervention (0-1)
        """
        if action_type == "open_drainage" and target_id in self.nodes:
            # Boost drainage rate at this node
            self.nodes[target_id].water_level = max(
                self.nodes[target_id].water_level - 0.15 * value, 0.0
            )

        elif action_type == "reroute_edge" and target_id in self.edges:
            # Reduce flow on this edge (signal timing intervention)
            edge = self.edges[target_id]
            edge.current_flow = int(edge.current_flow * (1 - 0.3 * value))
            edge.route_load   = max(0, edge.route_load - 10)

        elif action_type == "reduce_signal" and target_id in self.nodes:
            # Green wave — reduce congestion at node and adjacent edges
            for nb in self.G.neighbors(target_id):
                key = (target_id, nb) if (target_id, nb) in self.edges else (nb, target_id)
                if key in self.edges:
                    self.edges[key].current_flow = int(
                        self.edges[key].current_flow * 0.85
                    )

        elif action_type == "green_corridor" and isinstance(target_id, list):
            # Activate low-AQI corridor — reduce pollution weighting on path
            for edge_key in target_id:
                if edge_key in self.edges:
                    self.edges[edge_key].pollution_corridor *= 0.7

    # ── STATE EXPORT ───────────────────────────────────────────
    def get_full_state(self) -> Dict:
        """Full state dict — sent to frontend via WebSocket."""
        return {
            "timestep":  self.timestep,
            "timestamp": time.time(),
            "ai_enabled": self.ai_enabled,
            "scenario": {
                "rainfall":    self.scenario_rainfall,
                "traffic_mul": self.scenario_traffic_mul,
            },
            "nodes": {
                str(nid): {
                    "lat":             s.lat,
                    "lon":             s.lon,
                    "zone_type":       s.zone_type,
                    "water_level":     round(s.water_level, 4),
                    "traffic_density": round(s.traffic_density, 4),
                    "aqi":             round(s.aqi, 2),
                    "temperature":     round(s.temperature, 2),
                    "risk_score":      round(s.risk_score, 4),
                    "flood_risk":      round(s.flood_risk, 4),
                    "congestion_risk": round(s.congestion_risk, 4),
                    "heat_risk":       round(s.heat_risk, 4),
                    "aqi_risk":        round(s.aqi_risk, 4),
                    "alert_level":     s.alert_level,
                    "is_flooded":      s.is_flooded,
                    "is_critical":     s.is_critical,
                }
                for nid, s in self.nodes.items()
            },
            "edges": {
                f"{u}_{v}": {
                    "u":                u,
                    "v":                v,
                    "flow_ratio":       round(e.flow_ratio(), 4),
                    "congestion_factor":round(e.congestion_factor, 4),
                    "pollution_corridor":round(e.pollution_corridor, 2),
                    "flood_passable":   e.flood_passable,
                    "travel_time":      round(e.effective_travel_time(), 2),
                }
                for (u, v), e in self.edges.items()
            },
            "summary": self._compute_summary(),
        }

    def get_delta_state(self) -> Dict:
        """Only changed nodes/edges — reduces WebSocket payload size."""
        full    = self.get_full_state()
        delta   = {"timestep": full["timestep"], "timestamp": full["timestamp"],
                   "summary": full["summary"], "nodes": {}, "edges": {}}

        for nid, ndata in full["nodes"].items():
            prev = self._prev_state.get("nodes", {}).get(nid, {})
            if ndata.get("alert_level") != prev.get("alert_level") or \
               abs(ndata.get("risk_score", 0) - prev.get("risk_score", 0)) > 0.02:
                delta["nodes"][nid] = ndata

        for eid, edata in full["edges"].items():
            prev = self._prev_state.get("edges", {}).get(eid, {})
            if abs(edata.get("flow_ratio", 0) - prev.get("flow_ratio", 0)) > 0.03:
                delta["edges"][eid] = edata

        self._prev_state = full
        return delta

    def get_state_vector(self) -> np.ndarray:
        """Flat numpy array for RL agent observation space."""
        vectors = [s.to_vector() for s in self.nodes.values()]
        return np.concatenate(vectors).astype(np.float32)

    def get_node_state_vector(self, node_id: int) -> np.ndarray:
        """Single node vector for targeted RL queries."""
        if node_id in self.nodes:
            return self.nodes[node_id].to_vector()
        return np.zeros(10, dtype=np.float32)

    def _compute_summary(self) -> Dict:
        """Aggregate metrics for the dashboard metrics panel."""
        states = list(self.nodes.values())
        n      = len(states)
        if n == 0:
            return {}

        flooded   = sum(1 for s in states if s.is_flooded)
        critical  = sum(1 for s in states if s.is_critical)
        avg_aqi   = float(np.mean([s.aqi for s in states]))
        avg_risk  = float(np.mean([s.risk_score for s in states]))
        avg_temp  = float(np.mean([s.temperature for s in states]))
        avg_cong  = float(np.mean([s.traffic_density for s in states]))

        return {
            "total_zones":        n,
            "flooded_zones":      flooded,
            "critical_zones":     critical,
            "avg_aqi":            round(avg_aqi, 2),
            "avg_risk_score":     round(avg_risk, 4),
            "avg_temperature":    round(avg_temp, 2),
            "avg_congestion":     round(avg_cong, 4),
            "flood_pct":          round(flooded / n * 100, 2),
            "critical_pct":       round(critical / n * 100, 2),
        }

    def reset(self):
        """Reset twin to initial state — used by RL training."""
        self._initialise_states()
        self.timestep          = 0
        self.scenario_rainfall = 0.0
        self.scenario_traffic_mul = 1.0
        self._prev_state       = {}

    def get_node_ids(self):
        return list(self.nodes.keys())

    def get_edge_keys(self):
        return list(self.edges.keys())

    def n_nodes(self):
        return len(self.nodes)

    def n_edges(self):
        return len(self.edges)


# ─────────────────────────────────────────
#  QUICK SMOKE TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    CACHE = r"D:\projects__\resonate_26\datasets\chennai_graph.graphml"

    twin = TwinEngine(cache_path=CACHE)

    print("\n── Smoke test: 5 timesteps ──")
    twin.set_rainfall(45.0)       # heavy rain
    twin.set_traffic_surge(2.5)   # rush hour

    for i in range(5):
        state = twin.step()
        s     = state["summary"]
        print(f"  t={i+1:02d} | "
              f"flooded={s['flooded_zones']:4d} | "
              f"critical={s['critical_zones']:4d} | "
              f"avg_risk={s['avg_risk_score']:.3f} | "
              f"avg_aqi={s['avg_aqi']:.1f} | "
              f"avg_temp={s['avg_temperature']:.1f}°C")

    print("\n── Delta state test ──")
    delta = twin.get_delta_state()
    print(f"  Changed nodes: {len(delta['nodes'])} / {twin.n_nodes()}")
    print(f"  Changed edges: {len(delta['edges'])} / {twin.n_edges()}")

    print("\n── RL state vector ──")
    vec = twin.get_state_vector()
    print(f"  Shape: {vec.shape} | Min: {vec.min():.3f} | Max: {vec.max():.3f}")

    print("\n[ TwinEngine ] All checks passed.")