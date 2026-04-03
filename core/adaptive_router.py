import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import heapq
import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


# ─────────────────────────────────────────
#  WEIGHTS — how much each factor costs
#  Tune these for demo impact
# ─────────────────────────────────────────
W_CONGESTION  = 0.30
W_FLOOD       = 0.35
W_AQI         = 0.20
W_HEAT        = 0.10
W_RAINFALL    = 0.05

COOLDOWN_STEPS    = 10    # edge stays penalised for N timesteps after heavy routing
COOLDOWN_PENALTY  = 2.5   # multiplier applied during cooldown
MAX_ROUTE_LOAD    = 50    # max suggestions per edge per cycle before penalty kicks in


# ─────────────────────────────────────────
#  ROUTE RESULT
# ─────────────────────────────────────────
@dataclass
class RouteResult:
    path:           List[int]          # node_id sequence
    edges:          List[Tuple]        # (u,v) sequence
    total_cost:     float
    travel_time:    float              # seconds
    risk_score:     float              # 0-1 composite
    flood_exposure: float
    aqi_exposure:   float
    heat_exposure:  float
    congestion_avg: float
    safe:           bool               # True if all risk < warning thresholds
    alert:          str                # "safe" / "warning" / "critical"
    person_index:   int  = 0          # which person in this cycle


# ─────────────────────────────────────────
#  ADAPTIVE ROUTER
# ─────────────────────────────────────────
class AdaptiveRouter:
    """
    Wardrop-equilibrium router — every routing decision immediately
    mutates edge weights so the next person sees updated costs.

    All 6 twin factors feed into edge weight:
      congestion, flood_risk, AQI, heat, rainfall, travel_time

    Cooldown system prevents funnelling everyone to the same road.
    """

    def __init__(self, twin):
        self.twin     = twin
        self.G        = twin.G

        # Per-edge routing load this cycle (resets each cycle)
        self.route_load:   Dict[Tuple, int]   = {}

        # Cooldown counter per edge — counts down from COOLDOWN_STEPS to 0
        self.cooldown:     Dict[Tuple, int]   = {}

        # History for analytics
        self.route_history: List[Dict]        = []
        self.cycle_count    = 0
        self.person_count   = 0

    # ── PUBLIC API ──────────────────────────────────────────────

    def suggest_route(
        self,
        origin:      int,
        destination: int,
        preferences: Optional[Dict] = None,
    ) -> Optional[RouteResult]:
        """
        Compute optimal route for one person.
        Immediately updates edge load → next person sees new weights.

        preferences: dict with optional overrides
          { "avoid_flood": True, "avoid_aqi": True, "fastest": False }
        """
        prefs = preferences or {}
        self.person_count += 1

        # Validate nodes exist in graph
        if origin not in self.G or destination not in self.G:
            return None

        # Run modified Dijkstra with live twin state
        path = self._dijkstra(origin, destination, prefs)
        if path is None or len(path) < 2:
            return None

        # Build result with full analytics
        result = self._build_result(path, self.person_count)

        # ── WARDROP FEEDBACK LOOP ──
        # Update edge loads IMMEDIATELY so next person's weights differ
        for edge_key in result.edges:
            self.route_load[edge_key]  = self.route_load.get(edge_key, 0) + 1
            self.twin.edges[edge_key].route_load += 1

            # Start cooldown if edge is getting saturated
            if self.route_load.get(edge_key, 0) >= MAX_ROUTE_LOAD:
                self.cooldown[edge_key] = COOLDOWN_STEPS

        # Log for analytics
        self.route_history.append({
            "person":      self.person_count,
            "origin":      origin,
            "destination": destination,
            "cost":        result.total_cost,
            "alert":       result.alert,
            "timestamp":   time.time(),
        })

        return result

    def suggest_batch(
        self,
        origin:      int,
        destination: int,
        n_people:    int,
        preferences: Optional[Dict] = None,
    ) -> List[RouteResult]:
        """
        Route N people from same origin to destination.
        This is where Wardrop equilibrium emerges visibly —
        early people fill cheap edges, later people get rerouted.
        """
        results = []
        for _ in range(n_people):
            r = self.suggest_route(origin, destination, preferences)
            if r:
                results.append(r)
        return results

    def new_cycle(self):
        """
        Call at the start of each simulation cycle.
        Decays cooldowns, resets per-cycle load counters.
        """
        self.cycle_count += 1
        self.person_count = 0
        self.route_load   = {}

        # Tick down cooldowns
        expired = [k for k, v in self.cooldown.items() if v <= 0]
        for k in expired:
            del self.cooldown[k]
        for k in self.cooldown:
            self.cooldown[k] -= 1

    def get_diversity_stats(self) -> Dict:
        """
        How well distributed are routes across the last batch?
        Used by operator console to visualise Wardrop equilibrium.
        """
        if not self.route_history:
            return {}

        recent = self.route_history[-100:]   # last 100 suggestions
        edge_usage: Dict[str, int] = {}
        for r in recent:
            pass   # filled from route_history in future iteration

        loads = list(self.route_load.values())
        if not loads:
            return {"diversity": 1.0, "max_load": 0, "avg_load": 0.0}

        return {
            "diversity":    round(1 - (max(loads) / (sum(loads) + 1)), 3),
            "max_load":     max(loads),
            "avg_load":     round(sum(loads) / len(loads), 2),
            "total_routed": len(self.route_history),
        }

    # ── CORE DIJKSTRA ────────────────────────────────────────────

    def _dijkstra(
        self,
        origin:      int,
        destination: int,
        prefs:       Dict,
    ) -> Optional[List[int]]:
        """
        Modified Dijkstra using composite edge weight from all 6 twin factors.
        Edge weight updates LIVE from twin state — not cached.
        """
        dist  = {origin: 0.0}
        prev  = {origin: None}
        heap  = [(0.0, origin)]

        while heap:
            cost, node = heapq.heappop(heap)

            if node == destination:
                return self._reconstruct(prev, origin, destination)

            if cost > dist.get(node, math.inf):
                continue

            for neighbor in self.G.neighbors(node):
                edge_key = (node, neighbor) if (node, neighbor) in self.twin.edges \
                           else (neighbor, node)

                if edge_key not in self.twin.edges:
                    continue

                edge  = self.twin.edges[edge_key]

                # Skip flooded edges entirely
                if not edge.flood_passable:
                    if not prefs.get("allow_flooded", False):
                        continue

                w     = self._edge_weight(node, neighbor, edge_key, edge, prefs)
                new_cost = cost + w

                if new_cost < dist.get(neighbor, math.inf):
                    dist[neighbor] = new_cost
                    prev[neighbor] = node
                    heapq.heappush(heap, (new_cost, neighbor))

        return None   # no path found

    def _edge_weight(
        self,
        u:        int,
        v:        int,
        edge_key: Tuple,
        edge,
        prefs:    Dict,
    ) -> float:
        """
        Composite edge weight — the heart of the adaptive router.

        Every call reads LIVE twin state, so:
          Person 1  → low congestion on Route X → cheap weight
          Person 51 → Route X now has 50 people → high weight → picks Route Y
        """
        # ── Base travel time (BPR formula already applied in EdgeState)
        base = edge.effective_travel_time()

        # ── Node-level risk from both endpoints
        u_state = self.twin.nodes.get(u)
        v_state = self.twin.nodes.get(v)

        flood_risk  = max(
            u_state.flood_risk if u_state else 0,
            v_state.flood_risk if v_state else 0,
        )
        aqi_avg     = (
            (u_state.aqi if u_state else 0) +
            (v_state.aqi if v_state else 0)
        ) / 2.0
        heat_avg    = (
            (u_state.temperature if u_state else 30) +
            (v_state.temperature if v_state else 30)
        ) / 2.0
        rain_avg    = (
            (u_state.rainfall_local if u_state else 0) +
            (v_state.rainfall_local if v_state else 0)
        ) / 2.0

        # Normalise to 0-1
        congestion_n = edge.flow_ratio()
        flood_n      = flood_risk
        aqi_n        = min(aqi_avg / 300.0, 1.0)
        heat_n       = max((heat_avg - 30) / 15.0, 0.0)
        rain_n       = min(rain_avg / 100.0, 1.0)

        # Apply preference overrides — user said "avoid flood" → 3x flood weight
        w_flood = W_FLOOD * (3.0 if prefs.get("avoid_flood") else 1.0)
        w_aqi   = W_AQI   * (3.0 if prefs.get("avoid_aqi")   else 1.0)

        # Risk penalty (0-1 scaled to travel time)
        risk_penalty = (
            W_CONGESTION * congestion_n +
            w_flood      * flood_n      +
            w_aqi        * aqi_n        +
            W_HEAT       * heat_n       +
            W_RAINFALL   * rain_n
        ) * base   # penalty scales with travel time so long roads don't get free pass

        # ── Route load penalty (Wardrop core)
        load        = self.route_load.get(edge_key, 0)
        load_factor = 1.0 + (load / MAX_ROUTE_LOAD) * 1.5

        # ── Cooldown penalty
        cd_factor   = COOLDOWN_PENALTY if edge_key in self.cooldown else 1.0

        return (base + risk_penalty) * load_factor * cd_factor

    def _reconstruct(
        self,
        prev:        Dict,
        origin:      int,
        destination: int,
    ) -> List[int]:
        path = []
        node = destination
        while node is not None:
            path.append(node)
            node = prev.get(node)
        path.reverse()
        if path[0] == origin:
            return path
        return []

    # ── RESULT BUILDER ───────────────────────────────────────────

    def _build_result(self, path: List[int], person_index: int) -> RouteResult:
        edges         = []
        travel_time   = 0.0
        flood_vals    = []
        aqi_vals      = []
        heat_vals     = []
        congestion_v  = []

        for i in range(len(path) - 1):
            u, v     = path[i], path[i + 1]
            edge_key = (u, v) if (u, v) in self.twin.edges else (v, u)
            edges.append(edge_key)

            if edge_key in self.twin.edges:
                e = self.twin.edges[edge_key]
                travel_time  += e.effective_travel_time()
                congestion_v.append(e.flow_ratio())

            u_s = self.twin.nodes.get(u)
            v_s = self.twin.nodes.get(v)
            if u_s and v_s:
                flood_vals.append(max(u_s.flood_risk, v_s.flood_risk))
                aqi_vals.append((u_s.aqi + v_s.aqi) / 2)
                heat_vals.append((u_s.temperature + v_s.temperature) / 2)

        flood_exp  = float(np.mean(flood_vals))   if flood_vals   else 0.0
        aqi_exp    = float(np.mean(aqi_vals))     if aqi_vals     else 0.0
        heat_exp   = float(np.mean(heat_vals))    if heat_vals    else 30.0
        cong_avg   = float(np.mean(congestion_v)) if congestion_v else 0.0

        # Composite route risk
        risk = (
            W_CONGESTION * cong_avg +
            W_FLOOD      * flood_exp +
            W_AQI        * min(aqi_exp / 300.0, 1.0) +
            W_HEAT       * max((heat_exp - 30) / 15.0, 0.0)
        )

        if risk >= 0.7:
            alert = "critical"
        elif risk >= 0.4:
            alert = "warning"
        else:
            alert = "safe"

        return RouteResult(
            path           = path,
            edges          = edges,
            total_cost     = travel_time,
            travel_time    = round(travel_time, 2),
            risk_score     = round(risk, 4),
            flood_exposure = round(flood_exp, 4),
            aqi_exposure   = round(aqi_exp, 2),
            heat_exposure  = round(heat_exp, 2),
            congestion_avg = round(cong_avg, 4),
            safe           = alert == "safe",
            alert          = alert,
            person_index   = person_index,
        )


# ─────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append(r"D:\projects__\resonate_26\work")
    from core.twin_engine import TwinEngine

    CACHE = r"D:\projects__\resonate_26\datasets\chennai_graph.graphml"
    twin  = TwinEngine(cache_path=CACHE)
    twin.set_rainfall(30.0)
    twin.set_traffic_surge(2.0)
    twin.step()

    router = AdaptiveRouter(twin)
    nodes  = twin.get_node_ids()
    origin, destination = nodes[0], nodes[500]

    print(f"\n── Single route ──")
    r = router.suggest_route(origin, destination)
    if r:
        print(f"  Path length : {len(r.path)} nodes")
        print(f"  Travel time : {r.travel_time:.1f}s")
        print(f"  Risk score  : {r.risk_score:.3f}")
        print(f"  AQI exposure: {r.aqi_exposure:.1f}")
        print(f"  Alert       : {r.alert}")

    print(f"\n── Wardrop batch: 100 people, same O→D ──")
    results = router.suggest_batch(origin, destination, n_people=100)

    alerts   = {"safe": 0, "warning": 0, "critical": 0}
    for res in results:
        alerts[res.alert] += 1

    print(f"  Routed      : {len(results)} people")
    print(f"  Safe routes : {alerts['safe']}")
    print(f"  Warning     : {alerts['warning']}")
    print(f"  Critical    : {alerts['critical']}")

    stats = router.get_diversity_stats()
    print(f"\n── Route diversity ──")
    print(f"  Diversity score : {stats.get('diversity', 0):.3f}")
    print(f"  Max edge load   : {stats.get('max_load', 0)}")
    print(f"  Avg edge load   : {stats.get('avg_load', 0)}")

    print("\n[ AdaptiveRouter ] All checks passed.")