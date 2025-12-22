# Cat Brain Service

## System

```mermaid
graph TB
    subgraph Training["Training Pipeline"]
        ENV[Cat Environment<br/>Gymnasium]
        PPO[PPO Algorithm<br/>Stable-Baselines3]
        CALLBACKS[Training Callbacks<br/>Logging, Checkpoints]
        MODELS[(Model Storage<br/>models/)]
        
        ENV --> PPO
        PPO --> CALLBACKS
        CALLBACKS --> MODELS
    end
    
    subgraph API["FastAPI Service"]
        REQUEST[HTTP Request<br/>/predict]
        SCHEMAS[Pydantic Schemas<br/>cat_id + personality]
        MIDDLEWARE[Middleware<br/>Logging, Metrics]
        
        REQUEST --> MIDDLEWARE
        MIDDLEWARE --> SCHEMAS
    end
    
    subgraph Inference["Inference Engine"]
        PREDICTOR[Batch Predictor]
        PERSONALITY[Personality Modifier<br/>lazy/foodie/playful]
        CACHE[Redis Cache]
        LOADER[Model Loader<br/>Default + Individual]
        
        SCHEMAS --> PREDICTOR
        PREDICTOR --> PERSONALITY
        PERSONALITY --> CACHE
        CACHE -.cache miss.-> LOADER
        LOADER --> MODELS
    end
    
    subgraph Output["Response"]
        ACTION[Cat Action<br/>0-3: idle/eat/play/sleep]
        METRICS[Prometheus Metrics]
        LOGS[Structured Logs<br/>JSON]
        
        PREDICTOR --> ACTION
        MIDDLEWARE --> METRICS
        MIDDLEWARE --> LOGS
    end
    
    style Training fill:#e1f5ff
    style API fill:#fff3e0
    style Inference fill:#f3e5f5
    style Output fill:#e8f5e9
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Middleware
    participant Predictor
    participant PersonalityMod as Personality Modifier
    participant Cache
    participant ModelLoader
    participant Model as PPO Model
    
    Client->>API: POST /predict<br/>{cat_id, personality, hunger, energy, ...}
    API->>Middleware: Log request
    Middleware->>Predictor: predict_single(obs, cat_id, personality)
    
    Predictor->>PersonalityMod: apply(observation, personality)
    PersonalityMod-->>Predictor: modified_observation
    
    Predictor->>Cache: get(modified_obs)
    
    alt Cache Hit
        Cache-->>Predictor: cached_action
    else Cache Miss
        Predictor->>ModelLoader: get_model(version, cat_id)
        
        alt Individual Brain Exists
            ModelLoader-->>Predictor: cat_specific_model
        else Use Default Brain
            ModelLoader-->>Predictor: default_model
        end
        
        Predictor->>Model: predict(modified_obs)
        Model-->>Predictor: action
        Predictor->>Cache: set(modified_obs, action)
    end
    
    Predictor-->>API: action_int
    API-->>Client: {action, action_name}
    Middleware->>Prometheus: record metrics
```

## Model Selection Logic

```mermaid
flowchart TD
    START([Request with cat_id]) --> CHECK_ID{cat_id<br/>provided?}
    
    CHECK_ID -->|No| DEFAULT[Use Default Brain<br/>models/latest/]
    CHECK_ID -->|Yes| CHECK_INDIVIDUAL{Individual model<br/>exists?}
    
    CHECK_INDIVIDUAL -->|Yes| INDIVIDUAL[Load Cat Brain<br/>models/cats/&lt;cat_id&gt;/latest/]
    CHECK_INDIVIDUAL -->|No| DEFAULT
    
    DEFAULT --> LOAD_DEFAULT[Load from cache<br/>or disk]
    INDIVIDUAL --> LOAD_INDIVIDUAL[Load from cache<br/>or disk]
    
    LOAD_DEFAULT --> PREDICT[Predict Action]
    LOAD_INDIVIDUAL --> PREDICT
    
    PREDICT --> END([Return Action])
    
    style DEFAULT fill:#90caf9
    style INDIVIDUAL fill:#ce93d8
    style PREDICT fill:#a5d6a7
```

## Personality System

```mermaid
graph LR
    subgraph Input
        OBS[Original Observation<br/>hunger, energy, dist_food, dist_toy]
    end
    
    subgraph Modifiers["Personality Modifiers"]
        BALANCED[Balanced<br/>×1.0 all]
        LAZY[Lazy<br/>energy×1.5, hunger×0.8]
        FOODIE[Foodie<br/>hunger×1.4, energy×0.7]
        PLAYFUL[Playful<br/>dist_toy×0.6, hunger×0.7]
    end
    
    subgraph Processing
        APPLY[Apply Modifier] --> CLIP[Clip Values<br/>0-100, 0-10]
    end
    
    subgraph Output
        MODIFIED[Modified Observation]
    end
    
    OBS --> BALANCED & LAZY & FOODIE & PLAYFUL
    BALANCED & LAZY & FOODIE & PLAYFUL --> APPLY
    CLIP --> MODIFIED
    MODIFIED --> MODEL[ML Model]
    
    style LAZY fill:#b3e5fc
    style FOODIE fill:#ffccbc
    style PLAYFUL fill:#c5e1a5
```

## Training

```mermaid
flowchart LR
    subgraph Development
        CODE[Code Changes] --> TRAIN[Run Trainer<br/>python -m src.training.trainer]
        TRAIN --> EVAL[Evaluate Model<br/>10 episodes]
    end
    
    subgraph Storage
        EVAL --> SAVE[Save Model<br/>models/&lt;timestamp&gt;/]
        SAVE --> SYMLINK[Update symlink<br/>models/latest/]
        SAVE --> META[Save Metadata<br/>version, reward, etc]
    end
    
    subgraph Production
        SYMLINK --> RELOAD{API Running?}
        RELOAD -->|Yes| HOT[Hot Reload<br/>Load new model]
        RELOAD -->|No| COLD[Cold Start<br/>Load on startup]
        HOT --> SERVE[Serve Predictions]
        COLD --> SERVE
    end
    
    style TRAIN fill:#ffeb3b
    style SAVE fill:#4caf50
    style SERVE fill:#2196f3
```

## Cat Decision Making

```mermaid
stateDiagram-v2
    [*] --> Observe: Cat State
    Observe --> ApplyPersonality: hunger, energy, distances
    ApplyPersonality --> CheckCache: modified observation
    
    CheckCache --> CacheHit: found
    CheckCache --> LoadModel: not found
    
    LoadModel --> SelectModel: check cat_id
    SelectModel --> IndividualBrain: has custom model
    SelectModel --> DefaultBrain: use default
    
    IndividualBrain --> Predict
    DefaultBrain --> Predict
    
    Predict --> CacheResult: action
    CacheResult --> DecideAction
    CacheHit --> DecideAction
    
    DecideAction --> Idle: action = 0
    DecideAction --> MoveToFood: action = 1
    DecideAction --> MoveToToy: action = 2
    DecideAction --> Sleep: action = 3
    
    Idle --> [*]
    MoveToFood --> [*]
    MoveToToy --> [*]
    Sleep --> [*]
```

## Component Dependencies

```mermaid
graph TD
    subgraph Core
        ENV[environment.py<br/>Gymnasium Env]
        CONFIG[config.py<br/>Settings]
    end
    
    subgraph Training
        TRAINER[trainer.py<br/>PPO Training]
        CALLBACKS[callbacks.py<br/>Logging]
    end
    
    subgraph Inference
        LOADER[model_loader.py<br/>Load Models]
        PREDICTOR[predictor.py<br/>Batch + Personality]
        CACHE_MOD[cache.py<br/>Redis Cache]
    end
    
    subgraph API
        MAIN[main.py<br/>FastAPI App]
        SCHEMAS[schemas.py<br/>Pydantic Models]
        MIDDLEWARE_MOD[middleware.py<br/>Logging]
        HEALTH[health.py<br/>Health Checks]
    end
    
    subgraph Utils
        LOGGER[logger.py<br/>Structlog]
        METRICS_MOD[metrics.py<br/>Prometheus]
    end
    
    CONFIG --> TRAINER
    CONFIG --> PREDICTOR
    CONFIG --> MAIN
    
    ENV --> TRAINER
    TRAINER --> CALLBACKS
    
    LOADER --> PREDICTOR
    CACHE_MOD --> PREDICTOR
    
    SCHEMAS --> MAIN
    PREDICTOR --> MAIN
    MIDDLEWARE_MOD --> MAIN
    HEALTH --> MAIN
    
    LOGGER --> TRAINER
    LOGGER --> MAIN
    METRICS_MOD --> MIDDLEWARE_MOD
    
    style Core fill:#e3f2fd
    style Training fill:#fff9c4
    style Inference fill:#f3e5f5
    style API fill:#e8f5e9
    style Utils fill:#fce4ec
```
## How to Actually Run This Thing

### Dev Mode

```bash
# Install deps (use venv, don't be that person)
python -m venv venv
.\venv\Scripts\activate  # windows gang
pip install -r requirements.txt

# Run the API (hot reload included, cuz we're civilized)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or train a new brain (takes ~1 min depending on your potato)
python -m src.training.trainer
```

### Docker

```bash
cd docker
docker-compose up --build  # grab some coffee, first build takes a min
```

API's at `localhost:8000`, docs at `localhost:8000/docs` (FastAPI auto-docs ftw)

## Important Stuff You Should Know

### Cache TTL & Why It Matters

Redis cache is set to **1hr TTL** by default. Why? Cuz cat behaviors change throughout the day irl. If you're testing and cache is screwing with you:
- Either wait it out (lol no)
- Restart Redis: `docker restart cat-brain-redis`
- Or just change `CACHE_TTL` in config

### Model Versioning

Models are saved as `models/<timestamp>/` and `models/latest/` is a symlink. Why?
- API always loads from `latest/` 
- You can rollback by just changing the symlink
- Individual cat brains live in `models/cats/<cat_id>/latest/`

**IMPORTANT**: If you train a new default brain and your cats start acting weird, it's cuz they're using the new default (unless they have individual brains). Check `metadata.json` to see what version is actually loaded.

### Personality Modifiers

Personalities aren't trained - they're multipliers applied at inference. This means:
- Zero overhead, instant switching
- Same model serves all personalities
- But also means they're not "real" learned behaviors, just biases

For custom personalities, just add to `personality_config` in [config.py](src/core/config.py) and restart the API. Format's pretty obvious when you see it.

### Individual Brains vs Default

When you request with `cat_id="fluffy"`:
1. Checks `models/cats/fluffy/latest/` first
2. Falls back to `models/latest/` if not found
3. Both get cached separately (different cache keys)

**Gotcha**: Individual brains need to be trained explicitly. The trainer doesn't auto-create them. You gotta run training with a specific cat context (not implemented yet, but that's the plan).

### Metrics & Monitoring

Prometheus metrics live at `/metrics`. Key ones to watch:
- `prediction_duration_seconds` - if this spikes, cache is prob down or model loading is slow
- `cache_hit_rate` - should be >70% in prod, otherwise you're burning CPU
- `model_load_duration_seconds` - first load is slow (10-15s), cached loads are instant



### Logs & Debugging

Logs are JSON (structured logging via structlog). Grep-friendly:
```bash
# Filter by level
cat logs.json | jq 'select(.level=="error")'

# Track specific cat
cat logs.json | jq 'select(.cat_id=="fluffy")'

# Monitor cache performance
cat logs.json | jq 'select(.event=="cache_access")'
```

Set `LOG_LEVEL=DEBUG` for verbose output (warning: it's VERY verbose during training).

### API Response Times

Typical latencies:
- Cache hit: ~5-10ms
- Cache miss (model loaded): ~20-50ms  
- First request (cold start): ~10-15s (loading model from disk)

If you're seeing >100ms on cached requests, something's wrong (check Redis connection, network, or if you're running on a toaster).

### When to Retrain

Signs you need a new model:
- Cats doing dumb stuff (sleeping when starving, ignoring food when hungry)
- Reward plateau in training (check tensorboard)
- Added new features to environment and model doesn't use them

Otherwise, just tweak personalities - way faster than retraining.