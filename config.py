# ─────────────────────────────────────────────────────────────
#  Urban Sentinel — Central Config
#  All keys, paths, ports in one place.
#  Never hardcode these in any other file.
# ─────────────────────────────────────────────────────────────

# ── OpenWeatherMap ───────────────────────
OWM_API_KEY   = "49a3f59ba2121035c701f1c27b136a2b"
CITY_NAME     = "Chennai,IN"

# ── Paths ────────────────────────────────
GRAPH_CACHE   = r"D:\projects__\resonate_26\datasets\chennai_graph.graphml"
MODEL_DIR     = r"D:\projects__\resonate_26\work\models\saved"

# ── Simulation ───────────────────────────
TIMESTEP_SEC  = 0.5       # twin step interval
WEATHER_POLL  = 300       # OWM fetch interval in seconds

# ── Redis ────────────────────────────────
REDIS_HOST    = "localhost"
REDIS_PORT    = 6379
REDIS_DB      = 0

# ── PostgreSQL ───────────────────────────
DB_HOST       = "localhost"
DB_PORT       = 5432
# DB_NAME       = "urbansentinel"
DB_NAME       = "margdarshak"
DB_USER       = "postgres"
DB_PASSWORD   = "postgres"
DB_URL        = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── InfluxDB ─────────────────────────────
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = ""          # fill after Docker setup
INFLUX_ORG    = "margdarshak"
INFLUX_BUCKET = "city_metrics"

# ── FastAPI ──────────────────────────────
API_HOST      = "0.0.0.0"
API_PORT      = 8000

# ── Auth / JWT ───────────────────────────
JWT_SECRET    = "urbansentinel_secret_key_change_in_prod"
JWT_ALGO      = "HS256"
JWT_EXPIRE_MIN = 60 * 24    # 24 hours

# ── Demo users (hardcoded for hackathon) ─
DEMO_USERS = {
    "citizen@urban.in":  {"password": "citizen123", "role": "citizen"},
    "operator@urban.in": {"password": "operator123", "role": "operator"},
}

# ── RL Agent ─────────────────────────────
RL_TIMESTEPS      = 100_000
RL_LEARNING_RATE  = 3e-4
RL_N_STEPS        = 512 
RL_BATCH_SIZE     = 64

# ── GNN ──────────────────────────────────
GNN_HIDDEN_DIM    = 64
GNN_NUM_LAYERS    = 3
GNN_NODE_FEATURES = 10     # matches ZoneState.to_vector() dims

# ── LSTM ─────────────────────────────────
LSTM_HIDDEN       = 128
LSTM_LAYERS       = 2
LSTM_SEQ_LEN      = 20     # look back 20 timesteps = 10 seconds
LSTM_FORECAST     = 6      # predict next 6 steps = 3 seconds ahead