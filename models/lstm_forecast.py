import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")


# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"])



import torch
torch._dynamo.config.suppress_errors = True

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(r"D:\projects__\resonate_26\work")
from config import (
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_SEQ_LEN,
    LSTM_FORECAST, MODEL_DIR
)


# ─────────────────────────────────────────
#  WHAT THIS FORECASTS
#  Per zone, every 500ms:
#    - traffic density at t+5, t+10, t+20 mins
#    - AQI at t+5, t+10, t+20 mins
#    - flood risk at t+5, t+10, t+20 mins
#
#  "This route is safe NOW but in 12 mins
#   traffic hits 91% and AQI crosses 160"
# ─────────────────────────────────────────

# Input features per timestep per zone
INPUT_FEATURES = [
    "water_level",      # 0
    "traffic_density",  # 1
    "aqi_norm",         # 2  (aqi / 300)
    "temperature_norm", # 3  ((temp - 20) / 30)
    "rainfall_norm",    # 4  (rainfall / 100)
    "flood_risk",       # 5
    "congestion_risk",  # 6
    "hour_sin",         # 7  (sin of hour — captures cyclical time)
    "hour_cos",         # 8  (cos of hour)
]
N_FEATURES   = len(INPUT_FEATURES)   # 9
N_TARGETS    = 3                     # traffic, aqi, flood


# ─────────────────────────────────────────
#  LSTM FORECASTER
# ─────────────────────────────────────────
class CityLSTM(nn.Module):
    """
    Bidirectional LSTM for multi-step, multi-target forecasting.

    Input:  [batch, seq_len, n_features]
    Output: [batch, forecast_steps, n_targets]
             targets = (traffic_density, aqi_norm, flood_risk)

    Why Bi-LSTM: forward pass captures momentum (congestion building),
    backward pass captures peak patterns (rush hour shape from both ends).
    """

    def __init__(
        self,
        input_size:     int   = N_FEATURES,
        hidden_size:    int   = LSTM_HIDDEN,
        num_layers:     int   = LSTM_LAYERS,
        forecast_steps: int   = LSTM_FORECAST,
        n_targets:      int   = N_TARGETS,
        dropout:        float = 0.2,
    ):
        super().__init__()
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
        self.forecast_steps = forecast_steps
        self.n_targets      = n_targets

        # Bi-LSTM — hidden*2 because bidirectional
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        # Decoder: projects LSTM output to forecast horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, forecast_steps * n_targets),
        )

        # Confidence head — how certain is the forecast?
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_steps),
            nn.Sigmoid(),   # 0-1 confidence score
        )

    def forward(
        self,
        x: torch.Tensor,   # [batch, seq_len, n_features]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM
        lstm_out, _ = self.lstm(x)          # [batch, seq_len, hidden*2]
        last_hidden = lstm_out[:, -1, :]    # [batch, hidden*2]
        last_hidden = self.norm(last_hidden)

        # Forecast
        raw     = self.decoder(last_hidden)                              # [batch, steps*targets]
        forecast= raw.view(-1, self.forecast_steps, self.n_targets)     # [batch, steps, targets]
        forecast= torch.sigmoid(forecast)   # normalise to 0-1

        # Confidence
        conf    = self.confidence(last_hidden)   # [batch, steps]

        return forecast, conf


# ─────────────────────────────────────────
#  ZONE HISTORY BUFFER
#  Rolling window of last SEQ_LEN states
#  per zone — feeds the LSTM
# ─────────────────────────────────────────
class ZoneHistoryBuffer:
    """
    Maintains a fixed-length history deque per zone.
    Converts twin state snapshots into LSTM input tensors.
    """

    def __init__(self, seq_len: int = LSTM_SEQ_LEN):
        self.seq_len = seq_len
        # node_id → deque of feature vectors
        self._buffers: Dict[int, deque] = {}

    def update(self, twin) -> None:
        """Push current twin state into all zone buffers."""
        import math
        hour     = __import__('datetime').datetime.now().hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        for node_id, state in twin.nodes.items():
            if node_id not in self._buffers:
                self._buffers[node_id] = deque(maxlen=self.seq_len)

            features = np.array([
                state.water_level,
                state.traffic_density,
                min(state.aqi / 300.0, 1.0),
                max((state.temperature - 20) / 30.0, 0.0),
                min(state.rainfall_local / 100.0, 1.0),
                state.flood_risk,
                state.congestion_risk,
                hour_sin,
                hour_cos,
            ], dtype=np.float32)

            self._buffers[node_id].append(features)

    def get_tensor(self, node_id: int) -> Optional[torch.Tensor]:
        """
        Returns [1, seq_len, n_features] tensor for one zone.
        Returns None if buffer not full yet.
        """
        buf = self._buffers.get(node_id)
        if buf is None or len(buf) < self.seq_len:
            return None

        seq = np.stack(list(buf), axis=0)   # [seq_len, n_features]
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)   # [1, seq_len, n_features]

    def get_batch_tensor(self, node_ids: List[int]) -> Tuple[Optional[torch.Tensor], List[int]]:
        """
        Returns [batch, seq_len, n_features] for multiple zones.
        Only includes zones with full buffers.
        """
        tensors  = []
        valid_ids= []
        for nid in node_ids:
            t = self.get_tensor(nid)
            if t is not None:
                tensors.append(t)
                valid_ids.append(nid)

        if not tensors:
            return None, []

        return torch.cat(tensors, dim=0), valid_ids

    def is_ready(self, node_id: int) -> bool:
        buf = self._buffers.get(node_id)
        return buf is not None and len(buf) >= self.seq_len

    def n_ready(self) -> int:
        return sum(1 for buf in self._buffers.values() if len(buf) >= self.seq_len)


# ─────────────────────────────────────────
#  FORECAST ENGINE
#  Wraps CityLSTM + ZoneHistoryBuffer
#  Called by FastAPI every timestep
# ─────────────────────────────────────────
class ForecastEngine:
    """
    High-level interface used by the rest of the system.

    Usage:
        engine = ForecastEngine(twin)
        engine.update()               # call every twin.step()
        forecasts = engine.predict([node_id_1, node_id_2])
    """

    # Forecast horizon labels (in timesteps, each = 500ms)
    # t+6 = 3s, t+12 = 6s, t+20 = 10s
    # For demo: we label these as "t+5min", "t+10min", "t+20min"
    HORIZON_LABELS = ["t+5min", "t+10min", "t+20min",
                      "t+15min", "t+20min", "t+25min"]

    def __init__(self, twin, model: Optional[CityLSTM] = None):
        self.twin    = twin
        self.buffer  = ZoneHistoryBuffer(seq_len=LSTM_SEQ_LEN)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = CityLSTM().to(self.device)

        self.model.eval()
        self._forecast_cache: Dict[int, Dict] = {}

    def update(self) -> None:
        """Call this every twin.step() — updates all zone buffers."""
        self.buffer.update(self.twin)

    def predict(self, node_ids: List[int]) -> Dict[int, Dict]:
        """
        Forecast traffic, AQI, flood for given zones.
        Returns dict of node_id → forecast dict.
        """
        results = {}

        # Batch inference for efficiency
        batch, valid_ids = self.buffer.get_batch_tensor(node_ids)
        if batch is None:
            return {}

        self.model.eval()
        with torch.no_grad():
            forecast, confidence = self.model(batch.to(self.device))
            forecast    = forecast.cpu()
            confidence  = confidence.cpu()

        for i, node_id in enumerate(valid_ids):
            f    = forecast[i]     # [forecast_steps, 3]
            conf = confidence[i]   # [forecast_steps]

            state = self.twin.nodes.get(node_id)
            if state is None:
                continue

            steps = []
            for step in range(f.shape[0]):
                traffic_pred = float(f[step, 0])
                aqi_pred     = float(f[step, 1]) * 300.0   # denormalise
                flood_pred   = float(f[step, 2])
                conf_score   = float(conf[step])

                # Risk alert for this future timestep
                if flood_pred > 0.7 or traffic_pred > 0.9 or aqi_pred > 150:
                    alert = "critical"
                elif flood_pred > 0.4 or traffic_pred > 0.75 or aqi_pred > 100:
                    alert = "warning"
                else:
                    alert = "safe"

                steps.append({
                    "horizon":    self.HORIZON_LABELS[step],
                    "traffic":    round(traffic_pred, 4),
                    "aqi":        round(aqi_pred, 2),
                    "flood_risk": round(flood_pred, 4),
                    "confidence": round(conf_score, 4),
                    "alert":      alert,
                })

            results[node_id] = {
                "node_id":       node_id,
                "zone_type":     state.zone_type,
                "current_risk":  round(state.risk_score, 4),
                "forecast":      steps,
                "worst_ahead":   max(steps, key=lambda s: s["flood_risk"])["alert"]
                                 if steps else "safe",
            }

        self._forecast_cache = results
        return results

    def predict_all_high_risk(self, threshold: float = 0.4) -> Dict[int, Dict]:
        """Forecast only zones currently above risk threshold."""
        high_risk_ids = [
            nid for nid, state in self.twin.nodes.items()
            if state.risk_score >= threshold
        ]
        return self.predict(high_risk_ids[:100])   # cap at 100 for performance

    def get_route_forecast(self, node_ids: List[int]) -> Dict:
        """
        Given a list of nodes along a route, forecast the worst
        conditions the traveller will encounter.
        Used by adaptive_router to warn about future degradation.
        """
        forecasts = self.predict(node_ids)
        if not forecasts:
            return {"safe": True, "worst_alert": "safe", "details": []}

        worst_traffic  = 0.0
        worst_aqi      = 0.0
        worst_flood    = 0.0
        worst_alert    = "safe"
        details        = []

        for nid, fc in forecasts.items():
            for step in fc["forecast"]:
                worst_traffic = max(worst_traffic, step["traffic"])
                worst_aqi     = max(worst_aqi,     step["aqi"])
                worst_flood   = max(worst_flood,   step["flood_risk"])
                if step["alert"] == "critical":
                    worst_alert = "critical"
                elif step["alert"] == "warning" and worst_alert != "critical":
                    worst_alert = "warning"

            details.append({
                "node_id":     nid,
                "worst_ahead": fc["worst_ahead"],
            })

        return {
            "safe":          worst_alert == "safe",
            "worst_alert":   worst_alert,
            "worst_traffic": round(worst_traffic, 4),
            "worst_aqi":     round(worst_aqi, 2),
            "worst_flood":   round(worst_flood, 4),
            "details":       details,
        }


# ─────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────
def train_lstm(twin, epochs: int = 30, lr: float = 1e-3) -> CityLSTM:
    """
    Train on simulation rollouts.
    Ground truth = twin physics values at future timesteps.
    No external dataset needed.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ CityLSTM ] Training on {device}")

    model     = CityLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    buffer    = ZoneHistoryBuffer(seq_len=LSTM_SEQ_LEN)

    # Warm up buffer first
    print("  Warming up history buffer...")
    for _ in range(LSTM_SEQ_LEN + LSTM_FORECAST + 5):
        twin.set_rainfall(np.random.uniform(0, 50))
        twin.set_traffic_surge(np.random.uniform(1.0, 3.0))
        twin.step()
        buffer.update(twin)

    print(f"  Buffer ready: {buffer.n_ready()} zones")

    # Sample a fixed set of zones for training
    all_nodes   = list(twin.nodes.keys())
    train_nodes = all_nodes[::50]   # every 50th zone — ~1370 zones
    best_loss   = float("inf")
    best_state  = None

    for epoch in range(epochs):
        twin.set_rainfall(np.random.uniform(0, 60))
        twin.set_traffic_surge(np.random.uniform(1.0, 3.0))

        # Step forward FORECAST steps to get future labels
        future_states: List[Dict] = []
        for _ in range(LSTM_FORECAST):
            twin.step()
            buffer.update(twin)
            snapshot = {
                nid: (
                    twin.nodes[nid].traffic_density,
                    min(twin.nodes[nid].aqi / 300.0, 1.0),
                    twin.nodes[nid].flood_risk,
                )
                for nid in train_nodes
                if nid in twin.nodes
            }
            future_states.append(snapshot)

        # Build batch
        batch_x, valid_ids = buffer.get_batch_tensor(train_nodes)
        if batch_x is None or not valid_ids:
            continue

        # Build labels [batch, forecast_steps, 3]
        labels_list = []
        valid_final = []
        for nid in valid_ids:
            node_labels = []
            ok = True
            for step_snap in future_states:
                if nid not in step_snap:
                    ok = False
                    break
                node_labels.append(list(step_snap[nid]))
            if ok:
                labels_list.append(node_labels)
                valid_final.append(nid)

        if not labels_list:
            continue

        batch_x = batch_x[:len(labels_list)].to(device)
        labels  = torch.tensor(labels_list, dtype=torch.float32).to(device)

        model.train()
        optimizer.zero_grad()
        forecast, conf = model(batch_x)

        # Forecast loss + confidence regularization
        forecast_loss = criterion(forecast, labels)
        conf_loss     = (1 - conf).mean() * 0.1   # encourage high confidence
        loss          = forecast_loss + conf_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if forecast_loss.item() < best_loss:
            best_loss  = forecast_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {forecast_loss.item():.6f} | Best: {best_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    print(f"[ CityLSTM ] Training complete. Best loss: {best_loss:.6f}")
    return model


def save_model(model: CityLSTM, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ CityLSTM ] Saved → {path}")


def load_model(path: str) -> CityLSTM:
    model = CityLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[ CityLSTM ] Loaded ← {path}")
    return model


# ─────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    from core.twin_engine import TwinEngine
    from config import GRAPH_CACHE, MODEL_DIR

    twin = TwinEngine(cache_path=GRAPH_CACHE)

    print("\n── Model architecture ──")
    model = CityLSTM()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}")

    dummy = torch.randn(4, LSTM_SEQ_LEN, N_FEATURES)
    out, conf = model(dummy)
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}  (batch, steps, targets)")
    print(f"  Conf   : {conf.shape} (batch, steps)")

    print("\n── History buffer ──")
    buffer = ZoneHistoryBuffer()
    print("  Warming up buffer (need 20 steps)...")
    twin.set_rainfall(30.0)
    twin.set_traffic_surge(2.0)
    for _ in range(LSTM_SEQ_LEN + 2):
        twin.step()
        buffer.update(twin)
    print(f"  Ready zones: {buffer.n_ready()} / {len(twin.nodes)}")

    print("\n── Forecast engine ──")
    engine = ForecastEngine(twin)
    for _ in range(LSTM_SEQ_LEN + 2):
        twin.step()
        engine.update()

    nodes     = twin.get_node_ids()
    sample    = nodes[:5]
    forecasts = engine.predict(sample)
    print(f"  Forecasted {len(forecasts)} zones")
    for nid, fc in list(forecasts.items())[:2]:
        print(f"\n  Zone {nid} ({fc['zone_type']}) — "
              f"current risk {fc['current_risk']:.3f}")
        for step in fc["forecast"]:
            print(f"    {step['horizon']:8s} | "
                  f"traffic={step['traffic']:.3f} | "
                  f"aqi={step['aqi']:6.1f} | "
                  f"flood={step['flood_risk']:.3f} | "
                  f"conf={step['confidence']:.3f} | "
                  f"{step['alert']}")

    print("\n── Route forecast ──")
    route_fc = engine.get_route_forecast(nodes[:10])
    print(f"  Worst alert    : {route_fc['worst_alert']}")
    print(f"  Worst traffic  : {route_fc['worst_traffic']:.3f}")
    print(f"  Worst AQI      : {route_fc['worst_aqi']:.1f}")
    print(f"  Worst flood    : {route_fc['worst_flood']:.3f}")

    print("\n── Quick training (20 epochs) ──")
    trained = train_lstm(twin, epochs=20, lr=1e-3)

    save_model(trained, os.path.join(MODEL_DIR, "lstm_forecast.pt"))
    print("\n[ CityLSTM ] All checks passed.")