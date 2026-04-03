# MARGDARSHAK


 ## MVP link -  https://youtu.be/XGyPCo7gzj0   
_**(ML pipeline completely done, Frontend refining is left, in utmost couple of hours whole project will be deployed)**_

 
### The city that heals itself before it breaks

> **Track:** Smart Infrastructure & Urban Resilience
> **Team:** Segfault — Ayushman Yashaswi, Dweep Khatki
> **Event:** Resonate 2026
>


---

## The Problem

Cities today are reactive systems. By the time a flood advisory goes out, roads are already underwater. By the time traffic is rerouted, the congestion has already cascaded three zones deep. By the time AQI warnings reach citizens, the exposure has already happened.

The fundamental gap is not data — Indian cities have sensors, satellites, and APIs. The gap is **time**. No system currently closes the loop between prediction and action fast enough to matter.

---

## What MARGDARSHAK Is

MARGDARSHAK is a live digital twin of Chennai running as a weighted graph of **68,495 zone-nodes** and **90,670 road and drainage edges**, pulled directly from OpenStreetMap via OSMnx. Every node carries a real-time state vector. Every edge carries live flow, pollution, and passability data. The entire graph updates every 500ms.

On top of this twin, four layers of intelligence operate simultaneously:

**Layer 1 — Graph Attention Network (flood + risk prediction)**

A 3-layer GAT with 185,028 parameters trained in a self-supervised loop on the twin's own physics output. It predicts flood risk, congestion risk, AQI risk, and composite zone risk per node. Crucially, it captures spatial dependencies: whether Zone B floods depends on Zone A's water level and the drainage edge capacity between them, not just Zone B's rainfall alone. Attention weights from the final GAT layer feed a causal attribution engine that traces root causes in plain English — "Zone B flood risk CRITICAL — primary cause: Zone A drainage saturation, attention weight 0.71."

**Layer 2 — Bidirectional LSTM (temporal forecasting)**

A 2-layer Bi-LSTM with 581,752 parameters forecasts traffic density, AQI, and flood risk at t+5, t+10, t+15, t+20, t+25 minutes per zone, with a confidence score per horizon step. The forward pass captures momentum (congestion building), the backward pass captures peak patterns (rush hour shape from both ends of the sequence). Hour-of-day is encoded as sin/cos to handle cyclical time correctly. The forecaster tells you not just what is happening, but what will happen — and how certain it is.

**Layer 3 — PPO Reinforcement Learning Agent (self-healing)**

A Proximal Policy Optimization agent trained inside the digital twin. The twin itself is the Gymnasium environment. The agent observes a 49-dimensional compact risk summary and selects from 30 discrete interventions across 6 action types: open drainage, reroute traffic, activate green wave, activate low-AQI corridor, emergency drain, or do nothing. The reward function penalises both the original problem and any cascade the agent creates solving it — if fixing Zone A's flood pushes traffic into Zone B and spikes Zone B's AQI, that counts against the reward. The agent learns surgical intervention, not brute force.

**Layer 4 — Adaptive Router (Wardrop equilibrium)**

Every routing decision immediately mutates the twin's edge weights. Person 51 gets a different route than Person 1 because Person 1's route already increased that edge's load. The next Dijkstra call sees updated weights. Load distribution across the city emerges naturally without any central coordinator. In testing, 100 users routed between the same origin-destination pair across 13 different routes with a diversity score of 0.993 — no explicit balancing, purely emergent from the feedback loop.

---

## The Six-Factor Physics Cascade

Each of the 6 physical phenomena is causally linked — they create and amplify each other:

```
Rainfall       →  Water level rises at nodes
               →  Propagates through drainage edges (topology-aware)
               →  Flooded edges reduce road capacity by 60%
               →  Traffic reroutes to alternate corridors
               →  Rerouted traffic increases AQI on those corridors
               →  High AQI + traffic density raises local temperature
               →  Heat feedback increases human health risk score
               →  GNN detects cascade, RL agent intervenes
```

MARGDARSHAK models this entire chain — not individual isolated metrics.

---

## System Architecture

```
OpenWeatherMap API  ──→  WeatherFeed (background thread, 5min poll)
                                │
                                ↓
OSMnx Chennai Graph  ──→  TwinEngine.step() every 500ms
                                │
            ┌───────────────────┼──────────────────┐
            ↓                   ↓                  ↓
       GAT Flood GNN       Bi-LSTM             PPO Agent
       (risk per node)     (t+5 to t+25)       (30 actions)
            │                   │                  │
            └───────────────────┼──────────────────┘
                                ↓
                      FastAPI + WebSocket
                      Redis pub/sub (500ms delta)
                                │
                   ┌────────────┴────────────┐
                   ↓                         ↓
            Citizen View              Operator Console
          (no login required)        (JWT authenticated)
          Safe route guidance         Full graph + controls
          Flood + AQI alerts          RL agent action log
          Heatwave zones              Causal attribution
          Rainfall intensity          AI ON/OFF toggle
```

---

## Why Not Google Maps

Google Maps is a routing tool. MARGDARSHAK is a prediction and intervention system.

**Google Maps routes one person at a time** — it has no model of what happens when it gives 10,000 people the same alternate route. Our adaptive router updates edge weights after every single suggestion, so the 847th person never gets sent to an already-saturated road.

**Google Maps has no causal model** — it knows Zone C is slow but not why. Our attribution engine traces the root cause: "Zone C AQI spiked because 600 vehicles were rerouted 12 minutes ago from Zone A."

**Google Maps models traffic only** — a flooded road that no one has reported yet is invisible to their system. In our twin, flood propagates through drainage graph topology within seconds of rainfall injection.

**Google Maps observes and reports. MARGDARSHAK predicts and acts.**

---

## Why This Matters for India

Chennai floods every monsoon. Delhi's AQI hits 400+ in winter. Bengaluru's traffic runs at permanent near-capacity. Indian cities do not have infrastructure headroom — 80mm of rainfall that a European city handles easily floods Chennai because drainage is already at 70% capacity on a normal day. Our GNN is calibrated for this baseline saturation, not for cities with slack.

MARGDARSHAK aligns directly with India's Smart Cities Mission — real infrastructure data, real intervention logic, deployable on existing city sensor networks.

---

## Measured Results

| Metric | Value |
|---|---|
| City zones modelled | 68,495 |
| Road and drainage corridors | 90,670 |
| GNN parameters | 185,028 |
| LSTM parameters | 581,752 |
| RL observation dimensions | 49 |
| RL action space | 30 discrete interventions |
| Route diversity score (100 users, same O→D) | 0.993 |
| Max edge load before cooldown | 51 |
| Avg edge load across 100 routed users | 7.47 |
| Forecast horizon | t+5 to t+25 minutes |
| GNN training best loss | 0.0075 |
| LSTM training best loss | 0.0079 |

---

## Tech Stack

| Component | Technology | Why this, not something else |
|---|---|---|
| City graph | OSMnx + NetworkX | Real Chennai road topology, not synthetic grid |
| Flood GNN | PyTorch Geometric GAT | Attention learns which neighbors matter — GCN cannot |
| Forecasting | PyTorch Bi-LSTM | Captures momentum (forward) and peak shape (backward) |
| Self-healing | Stable-Baselines3 PPO | On-policy, stable, fast iteration in custom env |
| Backend | FastAPI + WebSockets | Async, real-time, production-grade |
| State sync | Redis pub/sub | Delta broadcast — only changed nodes sent each 500ms |
| Frontend | React + Deck.gl | Handles 90,670 live edges on map without lag |
| Weather | OpenWeatherMap API | Live Chennai rainfall, wind vector, temperature |

---

## Project Structure

```
margdarshak/
├── core/
│   ├── twin_engine.py        # City graph + 6-factor physics engine
│   ├── adaptive_router.py    # Wardrop dynamic routing
│   └── weather_feed.py       # OWM live feed + demo override
├── models/
│   ├── gnn_flood.py          # GAT GNN + causal attribution
│   ├── lstm_forecast.py      # Bi-LSTM forecaster + confidence heads
│   └── rl_agent.py           # PPO agent + CityEnv
├── api/
│   ├── main.py               # FastAPI + WebSocket + JWT auth
│   └── routes.py             # REST endpoints
├── urban-sentinel-ui/        # React + Deck.gl frontend
└── config.py                 # Central configuration
```

---

## Demo Flow

```
1. Inject stress — 80mm/hr rainfall + 8AM rush hour simultaneously
2. Watch flood propagate through drainage graph in real time
3. Watch AQI spike on commute corridors as traffic reroutes
4. AI OFF — city cascades, zones turn red, no intervention
5. AI ON  — RL agent detects cascade, attributes root cause
6. Agent opens Zone A drainage, activates green corridor
7. Adaptive router distributes next users across multiple routes
8. Zones recover — red to yellow to green — live on map
9. Operator console shows every agent decision with explanation
```

---

*Built at Resonate 2026 · Team Segfault · Ayushman Yashaswi, Dweep Khatki*
