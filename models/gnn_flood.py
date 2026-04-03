import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")

import torch
torch._dynamo.config.suppress_errors = True


# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"])



import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(r"D:\projects__\resonate_26\work")
from config import GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_NODE_FEATURES


# ─────────────────────────────────────────
#  GAT-BASED FLOOD + RISK GNN
# ─────────────────────────────────────────
class FloodGNN(nn.Module):
    def __init__(
        self,
        in_channels: int   = GNN_NODE_FEATURES,
        hidden:      int   = GNN_HIDDEN_DIM,
        num_layers:  int   = GNN_NUM_LAYERS,
        heads:       int   = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout    = dropout
        self.heads      = heads

        self.input_proj = nn.Linear(in_channels, hidden)

        self.gat_layers = nn.ModuleList()
        self.norms      = nn.ModuleList()

        for i in range(num_layers):
            in_dim  = hidden * heads if i > 0 else hidden
            out_dim = hidden
            self.gat_layers.append(
                GATConv(in_dim, out_dim, heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(out_dim * heads))

        final_dim = hidden * heads

        self.flood_head = nn.Sequential(
            nn.Linear(final_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.congestion_head = nn.Sequential(
            nn.Linear(final_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.aqi_head = nn.Sequential(
            nn.Linear(final_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.risk_head = nn.Sequential(
            nn.Linear(final_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        # Stored after forward with return_attention=True
        self._last_attn_edge_index: Optional[torch.Tensor] = None
        self._last_attn_weights:    Optional[torch.Tensor] = None

    def forward(
        self,
        x:               torch.Tensor,
        edge_index:      torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:

        h = F.relu(self.input_proj(x))

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            is_last = (i == self.num_layers - 1)

            if return_attention and is_last:
                h_new, (attn_ei, attn_w) = gat(
                    h, edge_index, return_attention_weights=True
                )
                self._last_attn_edge_index = attn_ei.detach().cpu()
                self._last_attn_weights    = attn_w.detach().cpu()
            else:
                h_new = gat(h, edge_index)

            h_new = norm(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if h.shape == h_new.shape:
                h = F.relu(h_new + h)
            else:
                h = F.relu(h_new)

        return {
            "flood":      self.flood_head(h).squeeze(-1),
            "congestion": self.congestion_head(h).squeeze(-1),
            "aqi":        self.aqi_head(h).squeeze(-1),
            "risk":       self.risk_head(h).squeeze(-1),
        }


# ─────────────────────────────────────────
#  TWIN → PYGEOMETRIC DATA CONVERTER
# ─────────────────────────────────────────
class TwinGraphConverter:
    def __init__(self, twin):
        self.twin      = twin
        self.node_list = list(twin.nodes.keys())
        self.node_idx  = {nid: i for i, nid in enumerate(self.node_list)}
        self._edge_index = self._build_edge_index()
        print(f"[ TwinGraphConverter ] "
              f"{len(self.node_list)} nodes, "
              f"{self._edge_index.shape[1]} edges")

    def _build_edge_index(self) -> torch.Tensor:
        rows, cols = [], []
        for (u, v) in self.twin.edges.keys():
            if u in self.node_idx and v in self.node_idx:
                ui, vi = self.node_idx[u], self.node_idx[v]
                rows.append(ui); cols.append(vi)
                rows.append(vi); cols.append(ui)
        return torch.tensor([rows, cols], dtype=torch.long)

    def to_pyg_data(self) -> Data:
        x = torch.tensor(
            np.stack([
                self.twin.nodes[nid].to_vector()
                for nid in self.node_list
            ]),
            dtype=torch.float32,
        )
        return Data(x=x, edge_index=self._edge_index)

    def to_subgraph(self, center_nodes: List[int], hops: int = 2) -> Tuple[Data, List[int]]:
        import networkx as nx

        subgraph_nodes = set()
        for nid in center_nodes:
            if nid in self.twin.G:
                ego = nx.ego_graph(self.twin.G, nid, radius=hops)
                subgraph_nodes.update(ego.nodes())

        subgraph_nodes = list(subgraph_nodes)
        sub_idx        = {nid: i for i, nid in enumerate(subgraph_nodes)}

        x_list = []
        for nid in subgraph_nodes:
            if nid in self.twin.nodes:
                x_list.append(self.twin.nodes[nid].to_vector())
            else:
                x_list.append(np.zeros(GNN_NODE_FEATURES, dtype=np.float32))

        x = torch.tensor(np.stack(x_list), dtype=torch.float32)

        rows, cols = [], []
        for (u, v) in self.twin.edges.keys():
            if u in sub_idx and v in sub_idx:
                rows.append(sub_idx[u]); cols.append(sub_idx[v])
                rows.append(sub_idx[v]); cols.append(sub_idx[u])

        if rows:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index), subgraph_nodes


# ─────────────────────────────────────────
#  CAUSAL ATTRIBUTION ENGINE
# ─────────────────────────────────────────
class CausalAttributor:
    def __init__(self, model: FloodGNN, twin, converter: TwinGraphConverter):
        self.model     = model
        self.twin      = twin
        self.converter = converter

    @torch.no_grad()
    def explain_node(self, node_id: int, top_k: int = 3) -> Dict:
        if node_id not in self.converter.node_idx:
            return {"error": f"Node {node_id} not in graph"}

        device = next(self.model.parameters()).device

        # Build subgraph around target node
        data, sub_nodes = self.converter.to_subgraph([node_id], hops=2)
        sub_idx_map     = {nid: i for i, nid in enumerate(sub_nodes)}
        sub_idx_rev     = {i: nid for i, nid in enumerate(sub_nodes)}

        local_i = sub_idx_map.get(node_id, 0)

        self.model.eval()
        preds = self.model(
            data.x.to(device),
            data.edge_index.to(device),
            return_attention=True,
        )

        # Attention stored on CPU already
        attn_ei = self.model._last_attn_edge_index  # [2, E_attn]
        attn_w  = self.model._last_attn_weights      # [E_attn, heads]

        contributions = []
        if attn_ei is not None and attn_w is not None:
            # Find edges whose TARGET is our local node
            incoming_mask = (attn_ei[1] == local_i)
            src_indices   = attn_ei[0][incoming_mask].tolist()
            weights       = attn_w[incoming_mask].mean(dim=1).tolist()

            for src_local, weight in zip(src_indices, weights):
                src_nid = sub_idx_rev.get(src_local, -1)
                if src_nid in self.twin.nodes:
                    s = self.twin.nodes[src_nid]
                    contributions.append({
                        "node_id":     src_nid,
                        "attention":   round(float(weight), 4),
                        "water_level": round(s.water_level, 4),
                        "zone_type":   s.zone_type,
                        "alert":       s.alert_level,
                    })

        contributions.sort(key=lambda x: x["attention"], reverse=True)
        top_contributors = contributions[:top_k]

        target_state = self.twin.nodes[node_id]
        flood_pred   = float(preds["flood"][local_i].cpu())
        risk_pred    = float(preds["risk"][local_i].cpu())

        explanation  = self._build_explanation(
            node_id, target_state, flood_pred, risk_pred, top_contributors
        )

        return {
            "node_id":          node_id,
            "zone_type":        target_state.zone_type,
            "flood_prediction": round(flood_pred, 4),
            "risk_prediction":  round(risk_pred, 4),
            "top_contributors": top_contributors,
            "explanation":      explanation,
        }

    def _build_explanation(
        self,
        node_id:      int,
        state,
        flood_pred:   float,
        risk_pred:    float,
        contributors: List[Dict],
    ) -> str:
        level = "CRITICAL" if flood_pred > 0.7 else \
                "WARNING"  if flood_pred > 0.4 else "LOW"

        lines = [
            f"Zone {node_id} ({state.zone_type}) — "
            f"Flood risk {level} ({flood_pred:.2f}), "
            f"Overall risk {risk_pred:.2f}"
        ]

        if contributors:
            lines.append("Primary causes:")
            for i, c in enumerate(contributors, 1):
                lines.append(
                    f"  {i}. Zone {c['node_id']} ({c['zone_type']}) — "
                    f"attention {c['attention']:.3f}, "
                    f"water level {c['water_level']:.3f}, "
                    f"status {c['alert']}"
                )

        if state.water_level > 0.3:
            lines.append(
                f"Current water level: {state.water_level:.3f} "
                f"(drainage capacity strained)"
            )

        return "\n".join(lines)

    @torch.no_grad()
    def get_high_risk_nodes(
        self, threshold: float = 0.5, max_nodes: int = 20
    ) -> List[Dict]:
        self.model.eval()
        device = next(self.model.parameters()).device
        data   = self.converter.to_pyg_data()
        preds  = self.model(
            data.x.to(device),
            data.edge_index.to(device),
        )

        # flood_scores = preds["flood"].cpu().numpy()
        # risk_scores  = preds["risk"].cpu().numpy()
        flood_scores = preds["flood"].cpu().tolist()
        risk_scores  = preds["risk"].cpu().tolist()

        high_risk = []
        for i, (flood, risk) in enumerate(zip(flood_scores, risk_scores)):
            if risk > threshold:
                node_id = self.converter.node_list[i]
                high_risk.append({
                    "node_id":    node_id,
                    "flood_pred": round(float(flood), 4),
                    "risk_pred":  round(float(risk),  4),
                })

        high_risk.sort(key=lambda x: x["risk_pred"], reverse=True)
        return high_risk[:max_nodes]


# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────
def train_gnn(twin, epochs: int = 50, lr: float = 1e-3) -> FloodGNN:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ FloodGNN ] Training on {device}")

    converter = TwinGraphConverter(twin)
    model     = FloodGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = None

    for epoch in range(epochs):
        twin.set_rainfall(np.random.uniform(0, 60))
        twin.set_traffic_surge(np.random.uniform(1.0, 3.0))
        twin.step()

        data = converter.to_pyg_data().to(device)

        flood_labels = torch.tensor(
            [twin.nodes[nid].flood_risk      for nid in converter.node_list],
            dtype=torch.float32, device=device
        )
        congestion_labels = torch.tensor(
            [twin.nodes[nid].congestion_risk for nid in converter.node_list],
            dtype=torch.float32, device=device
        )
        aqi_labels = torch.tensor(
            [twin.nodes[nid].aqi_risk        for nid in converter.node_list],
            dtype=torch.float32, device=device
        )
        risk_labels = torch.tensor(
            [twin.nodes[nid].risk_score      for nid in converter.node_list],
            dtype=torch.float32, device=device
        )

        model.train()
        optimizer.zero_grad()
        preds = model(data.x, data.edge_index)

        loss = (
            criterion(preds["flood"],      flood_labels)      * 0.35 +
            criterion(preds["congestion"], congestion_labels) * 0.25 +
            criterion(preds["aqi"],        aqi_labels)        * 0.20 +
            criterion(preds["risk"],       risk_labels)       * 0.20
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.6f} | Best: {best_loss:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    print(f"[ FloodGNN ] Training complete. Best loss: {best_loss:.6f}")
    return model


def save_model(model: FloodGNN, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ FloodGNN ] Saved → {path}")


def load_model(path: str) -> FloodGNN:
    model = FloodGNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[ FloodGNN ] Loaded ← {path}")
    return model


# ─────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    from core.twin_engine import TwinEngine
    from config import GRAPH_CACHE, MODEL_DIR

    twin = TwinEngine(cache_path=GRAPH_CACHE)
    twin.set_rainfall(40.0)
    twin.set_traffic_surge(2.0)
    twin.step()

    print("\n── Model architecture ──")
    model = FloodGNN()
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total:,}")

    print("\n── Graph converter ──")
    converter = TwinGraphConverter(twin)
    data      = converter.to_pyg_data()
    print(f"  Full graph : x={data.x.shape}, edges={data.edge_index.shape}")

    print("\n── Subgraph test ──")
    nodes    = twin.get_node_ids()
    sub_data, sub_nodes = converter.to_subgraph([nodes[0], nodes[100]], hops=2)
    print(f"  Subgraph   : {len(sub_nodes)} nodes, x={sub_data.x.shape}")

    print("\n── Inference test (untrained) ──")
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
    for k, v in preds.items():
        print(f"  {k:12s}: shape={v.shape}, "
              f"min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")

    print("\n── Quick training (10 epochs) ──")
    trained = train_gnn(twin, epochs=10, lr=1e-3)

    print("\n── Causal attribution ──")
    attributor = CausalAttributor(trained, twin, converter)
    result     = attributor.explain_node(nodes[0])
    print(f"  {result['explanation']}")

    print("\n── High risk nodes ──")
    high_risk = attributor.get_high_risk_nodes(threshold=0.3, max_nodes=5)
    for hr in high_risk:
        print(f"  Node {hr['node_id']} — "
              f"flood={hr['flood_pred']:.3f}, risk={hr['risk_pred']:.3f}")

    save_model(trained, os.path.join(MODEL_DIR, "gnn_flood.pt"))
    print("\n[ FloodGNN ] All checks passed.")