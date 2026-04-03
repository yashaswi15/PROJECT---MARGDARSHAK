import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import torch
torch._dynamo.config.suppress_errors = True

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import sys

sys.path.append(r"D:\projects__\resonate_26\work")
from config import (
    OWM_API_KEY, GRAPH_CACHE, TIMESTEP_SEC,
    JWT_SECRET, JWT_ALGO, JWT_EXPIRE_MIN, DEMO_USERS,
    REDIS_HOST, REDIS_PORT, MODEL_DIR
)

from core.twin_engine import TwinEngine
from core.adaptive_router import AdaptiveRouter
from core.weather_feed import WeatherFeed
from models.gnn_flood import FloodGNN, TwinGraphConverter, CausalAttributor, load_model as load_gnn
from models.lstm_forecast import CityLSTM, ForecastEngine, load_model as load_lstm
from models.rl_agent import CityEnv, SelfHealingAgent, load_agent
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
import numpy as np


# ─────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────
twin:        Optional[TwinEngine]     = None
router:      Optional[AdaptiveRouter] = None
weather:     Optional[WeatherFeed]    = None
gnn:         Optional[FloodGNN]       = None
converter:   Optional[TwinGraphConverter] = None
attributor:  Optional[CausalAttributor]   = None
forecaster:  Optional[ForecastEngine] = None
rl_env:      Optional[CityEnv]        = None
agent:       Optional[SelfHealingAgent] = None
scheduler:   Optional[AsyncIOScheduler] = None

# Connected WebSocket clients
clients: Dict[str, WebSocket] = {}   # client_id → websocket


# ─────────────────────────────────────────
#  STARTUP / SHUTDOWN
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global twin, router, weather, gnn, converter, attributor
    global forecaster, rl_env, agent, scheduler

    print("[ MARGDARSHAK ] Starting up...")

    # 1. Twin engine
    twin    = TwinEngine(cache_path=GRAPH_CACHE)
    router  = AdaptiveRouter(twin)
    weather = WeatherFeed(api_key=OWM_API_KEY, twin=twin)
    weather.start()

    # 2. GNN
    gnn_path = os.path.join(MODEL_DIR, "gnn_flood.pt")
    if os.path.exists(gnn_path):
        gnn = load_gnn(gnn_path)
        gnn.eval()
    else:
        gnn = FloodGNN()
    converter  = TwinGraphConverter(twin)
    attributor = CausalAttributor(gnn, twin, converter)

    # 3. LSTM forecaster
    lstm_path = os.path.join(MODEL_DIR, "lstm_forecast.pt")
    if os.path.exists(lstm_path):
        lstm_model = load_lstm(lstm_path)
    else:
        lstm_model = CityLSTM()
    forecaster = ForecastEngine(twin, model=lstm_model)

    # 4. RL agent
    rl_env   = CityEnv(twin)
    agent_path = os.path.join(MODEL_DIR, "rl_agent")
    if os.path.exists(agent_path + ".zip"):
        rl_model = load_agent(agent_path, twin)
        agent    = SelfHealingAgent(rl_model, rl_env)
    else:
        print("[ MARGDARSHAK ] No trained RL agent found — AI interventions disabled")

    # 5. Simulation scheduler — steps twin every 500ms
    scheduler = AsyncIOScheduler()
    scheduler.add_job(simulation_tick, "interval", seconds=TIMESTEP_SEC)
    scheduler.start()

    print("[ MARGDARSHAK ] All systems online")
    yield

    # Shutdown
    weather.stop()
    scheduler.shutdown()
    print("[ MARGDARSHAK ] Shutdown complete")


# ─────────────────────────────────────────
#  SIMULATION TICK — runs every 500ms
# ─────────────────────────────────────────
async def simulation_tick():
    global twin, forecaster, agent, rl_env, clients

    if twin is None:
        return

    # Step twin
    state = twin.step()

    # Update LSTM buffer
    if forecaster:
        forecaster.update()

    # RL agent acts if AI is enabled
    if twin.ai_enabled and agent is not None:
        rl_env._update_targets()
        action_result = agent.act()
        state["last_action"] = action_result
    else:
        state["last_action"] = None

    # Broadcast delta to all connected WebSocket clients
    if clients:
        delta = twin.get_delta_state()
        if twin.ai_enabled and state.get("last_action"):
            delta["last_action"] = state["last_action"]

        message = json.dumps(delta)
        disconnected = []
        for client_id, ws in clients.items():
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(client_id)

        for cid in disconnected:
            clients.pop(cid, None)


# ─────────────────────────────────────────
#  APP
# ─────────────────────────────────────────
app = FastAPI(
    title="MARGDARSHAK API",
    description="Self-healing digital twin of Chennai",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# ─────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────
def create_token(email: str, role: str) -> str:
    payload = {
        "sub":   email,
        "role":  role,
        "exp":   datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGO]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def operator_only(payload=Depends(verify_token)):
    if payload.get("role") != "operator":
        raise HTTPException(status_code=403, detail="Operator access required")
    return payload


# ─────────────────────────────────────────
#  REST ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":    "online",
        "twin":      twin is not None,
        "ai":        twin.ai_enabled if twin else False,
        "timestep":  twin.timestep if twin else 0,
        "clients":   len(clients),
    }


@app.post("/auth/login")
async def login(body: dict):
    email    = body.get("email", "")
    password = body.get("password", "")

    user = DEMO_USERS.get(email)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(email, user["role"])
    return {
        "token": token,
        "role":  user["role"],
        "email": email,
    }


@app.get("/twin/state")
async def get_state():
    """Full twin state snapshot — used on initial frontend load."""
    if twin is None:
        raise HTTPException(status_code=503, detail="Twin not initialised")
    return twin.get_full_state()


@app.get("/twin/summary")
async def get_summary():
    if twin is None:
        raise HTTPException(status_code=503, detail="Twin not initialised")
    return twin._compute_summary()


@app.post("/twin/scenario")
async def set_scenario(body: dict):
    """Citizen and operator: set rainfall and traffic surge sliders."""
    if twin is None:
        raise HTTPException(status_code=503, detail="Twin not initialised")

    rainfall    = body.get("rainfall", 0.0)
    traffic_mul = body.get("traffic_mul", 1.0)

    twin.set_rainfall(float(rainfall))
    twin.set_traffic_surge(float(traffic_mul))

    if weather:
        weather.set_override(
            rainfall    = float(rainfall),
            temperature = body.get("temperature", 30.0),
        )

    return {"status": "ok", "rainfall": rainfall, "traffic_mul": traffic_mul}


@app.post("/twin/ai")
async def toggle_ai(body: dict, payload=Depends(operator_only)):
    """Operator only: toggle AI self-healing on/off."""
    enabled = body.get("enabled", True)
    twin.set_ai_enabled(bool(enabled))
    return {"ai_enabled": twin.ai_enabled}


@app.post("/twin/reset")
async def reset_twin(payload=Depends(operator_only)):
    """Operator only: reset twin to baseline."""
    twin.reset()
    if weather:
        weather.clear_override()
    router.new_cycle()
    return {"status": "reset", "timestep": twin.timestep}


@app.get("/route")
async def get_route(origin: int, destination: int,
                    avoid_flood: bool = False, avoid_aqi: bool = False):
    """
    Citizen: get optimal route between two node IDs.
    Returns path + risk breakdown + future forecast warning.
    """
    if router is None:
        raise HTTPException(status_code=503, detail="Router not ready")

    prefs  = {"avoid_flood": avoid_flood, "avoid_aqi": avoid_aqi}
    result = router.suggest_route(origin, destination, prefs)

    if result is None:
        raise HTTPException(status_code=404, detail="No route found")

    # Add LSTM forecast for the route
    forecast_warning = None
    if forecaster:
        route_fc = forecaster.get_route_forecast(result.path[:20])
        forecast_warning = route_fc

    return {
        "path":             result.path,
        "travel_time":      result.travel_time,
        "risk_score":       result.risk_score,
        "flood_exposure":   result.flood_exposure,
        "aqi_exposure":     result.aqi_exposure,
        "heat_exposure":    result.heat_exposure,
        "congestion_avg":   result.congestion_avg,
        "alert":            result.alert,
        "forecast_warning": forecast_warning,
    }


@app.get("/forecast/{node_id}")
async def get_forecast(node_id: int):
    """Get LSTM forecast for a specific zone."""
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster not ready")

    results = forecaster.predict([node_id])
    if node_id not in results:
        raise HTTPException(status_code=404, detail="Zone not found or buffer not ready")

    return results[node_id]


@app.get("/attribution/{node_id}")
async def get_attribution(node_id: int, payload=Depends(operator_only)):
    """Operator only: GNN causal attribution for a zone."""
    if attributor is None:
        raise HTTPException(status_code=503, detail="Attributor not ready")

    result = attributor.explain_node(node_id)
    return result


@app.get("/high-risk")
async def get_high_risk(threshold: float = 0.5, payload=Depends(operator_only)):
    """Operator only: get top high-risk zones from GNN."""
    if attributor is None:
        raise HTTPException(status_code=503, detail="Attributor not ready")

    return attributor.get_high_risk_nodes(threshold=threshold, max_nodes=20)


@app.get("/diversity")
async def get_diversity(payload=Depends(operator_only)):
    """Operator only: route diversity stats."""
    if router is None:
        raise HTTPException(status_code=503, detail="Router not ready")
    return router.get_diversity_stats()


# ─────────────────────────────────────────
#  WEBSOCKET — live twin state broadcast
# ─────────────────────────────────────────
@app.websocket("/ws/twin")
async def websocket_twin(websocket: WebSocket):
    """
    Live state stream — connects to city twin.
    Sends delta state every 500ms.
    Both citizen and operator connect here.
    Frontend filters what to show based on role.
    """
    await websocket.accept()
    client_id = f"client_{int(time.time() * 1000)}"
    clients[client_id] = websocket

    print(f"[ WS ] Client connected: {client_id} | Total: {len(clients)}")

    try:
        # Send full state on connect
        if twin:
            full = twin.get_full_state()
            await websocket.send_text(json.dumps(full))

        # Keep alive — receive messages from client (scenario updates etc)
        while True:
            data = await websocket.receive_text()
            msg  = json.loads(data)

            # Client can send scenario updates via WebSocket too
            if msg.get("type") == "scenario":
                twin.set_rainfall(float(msg.get("rainfall", 0)))
                twin.set_traffic_surge(float(msg.get("traffic_mul", 1.0)))

            elif msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        clients.pop(client_id, None)
        print(f"[ WS ] Client disconnected: {client_id} | Total: {len(clients)}")
    except Exception as e:
        clients.pop(client_id, None)
        print(f"[ WS ] Error {client_id}: {e}")


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,
    )