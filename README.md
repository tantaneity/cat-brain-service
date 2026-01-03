# Cat Brain Service

living cat ai with emotions, reactions, and realistic behavior

## System Architecture

```mermaid
graph TB
    subgraph Training["Training Pipeline"]
        ENV[Cat Environment<br/>Gymnasium]
        PPO[PPO Algorithm<br/>Stable-Baselines3]
        CALLBACKS[Training Callbacks<br/>Logging, Checkpoints]
        MODELS[(Model Storage<br/>models/ + models/cats/)]
        
        ENV --> PPO
        PPO --> CALLBACKS
        CALLBACKS --> MODELS
    end
    
    subgraph API["FastAPI Service"]
        REQUEST[HTTP Request<br/>/predict + stimuli]
        ROUTES[API Routes<br/>predictions, cats, models]
        MIDDLEWARE[Middleware<br/>Logging, Metrics]
        DEPS[Dependencies<br/>DI Container]
        
        REQUEST --> MIDDLEWARE
        MIDDLEWARE --> ROUTES
        ROUTES --> DEPS
    end
    
    subgraph Services["Service Layer"]
        CAT_SERVICE[CatService<br/>Business Logic]
        CONTEXTUAL[ContextualEngine<br/>Emotions + Reactions]
        ACTION_HISTORY[ActionHistory<br/>JSONL Storage]
        MEMORY[CatMemory<br/>Behavior Tracking]
        
        DEPS --> CAT_SERVICE
        DEPS --> CONTEXTUAL
        CONTEXTUAL --> MEMORY
        CAT_SERVICE --> ACTION_HISTORY
    end
    
    subgraph Inference["Base Inference"]
        PREDICTOR[Batch Predictor]
        PERSONALITY[Personality Config<br/>lazy/foodie/playful]
        CACHE[Redis Cache<br/>5min TTL]
        LOADER[Model Loader<br/>Default + Individual]
        
        DEPS --> PREDICTOR
        PREDICTOR --> PERSONALITY
        PREDICTOR --> ACTION_HISTORY
        PERSONALITY --> CACHE
        CACHE -.cache miss.-> LOADER
        LOADER --> MODELS
        CAT_SERVICE --> LOADER
    end
    
    subgraph Living["Living Cat System"]
        EMOTIONS[Emotion Engine<br/>14 emotions]
        REACTIONS[Reaction System<br/>Stimulus Response]
        BEHAVIOR[Stochastic Behavior<br/>Randomness + Quirks]
        PATTERNS[Behavior Patterns<br/>Zoomies, Lazy, etc]
        
        CONTEXTUAL --> EMOTIONS
        CONTEXTUAL --> REACTIONS
        CONTEXTUAL --> BEHAVIOR
        CONTEXTUAL --> PATTERNS
    end
    
    subgraph Output["Enhanced Response"]
        ACTION[Cat Action<br/>8 actions + emotion]
        EMOTION_DATA[Emotional State<br/>mood, arousal, valence]
        ANIMATION[Animation Hints<br/>purr, scared, excited]
        SOUND[Sound Hints<br/>meow, hiss, chirp]
        CAT_INFO[Cat Info<br/>brain path, actions]
        METRICS[Prometheus Metrics]
        LOGS[Structured Logs<br/>JSON]
        
        CONTEXTUAL --> ACTION
        CONTEXTUAL --> EMOTION_DATA
        CONTEXTUAL --> ANIMATION
        CONTEXTUAL --> SOUND
        CAT_SERVICE --> CAT_INFO
        MIDDLEWARE --> METRICS
        MIDDLEWARE --> LOGS
    end
    
    style Training fill:#e1f5ff
    style API fill:#fff3e0
    style Services fill:#e8f5e9
    style Inference fill:#f3e5f5
    style Living fill:#ffebee
    style Output fill:#ffe0b2
```

## Request Flow with Living Cat System

```mermaid
sequenceDiagram
    participant Client
    participant Routes as API Routes
    participant Predictor as Base Predictor
    participant ContextEngine as Contextual Engine
    participant Emotions as Emotion Engine
    participant Reactions as Reaction System
    participant Behavior as Stochastic Behavior
    participant Memory as Cat Memory
    participant Model as PPO Model
    participant History as ActionHistory
    
    Client->>Routes: POST /predict<br/>{hunger, energy, mood,<br/>player_nearby, is_being_petted,<br/>loud_noise_level, etc}
    
    Routes->>Predictor: predict_single(obs, cat_id, personality)
    Note over Predictor: Apply personality modifiers<br/>(hunger×1.4 for foodie, etc)
    
    Predictor->>Model: predict(modified_obs)
    Model-->>Predictor: base_action (deterministic)
    
    Routes->>ContextEngine: process_action(base_action, state)
    
    ContextEngine->>Memory: get_or_create_memory(cat_id)
    Memory-->>ContextEngine: activity_level, recent_actions
    
    ContextEngine->>Emotions: get_emotional_state(mood, hunger, energy)
    Note over Emotions: Calculate arousal<br/>Determine emotion (14 types)<br/>Calculate intensity
    Emotions-->>ContextEngine: EmotionalState {emotion, arousal, valence}
    
    ContextEngine->>ContextEngine: extract_stimuli(state)
    Note over ContextEngine: Check: is_being_petted?<br/>loud_noise? new_toy?<br/>player_nearby? etc
    
    ContextEngine->>Reactions: get_reaction(stimulus, emotion)
    Note over Reactions: Match rules:<br/>happy + petted = purr<br/>scared + noise = hide<br/>hungry + food = rush
    Reactions-->>ContextEngine: ReactionModifier {action_override,<br/>mood_delta, animation, sound}
    
    alt Reaction Triggered
        ContextEngine->>Reactions: apply_reaction(base_action, reaction)
        Reactions-->>ContextEngine: final_action (overridden)
    else No Reaction
        ContextEngine->>Behavior: add_noise_to_prediction(base_action)
        Note over Behavior: 20% randomness<br/>mood-based variance<br/>random quirks
        Behavior-->>ContextEngine: final_action (with noise)
    end
    
    ContextEngine->>Memory: is_repeating_behavior()
    alt Repeating Too Much
        ContextEngine->>Behavior: introduce_distraction(action)
        Behavior-->>ContextEngine: distracted_action
    end
    
    ContextEngine->>Memory: record_action(action, mood)
    ContextEngine->>History: log_action(cat_id, obs, action)
    
    ContextEngine-->>Routes: {action, emotional_state,<br/>mood_delta, animation_hint,<br/>sound_hint, reaction_triggered}
    
    Routes-->>Client: {<br/>  action: 4,<br/>  action_name: "groom",<br/>  emotion: "happy",<br/>  emotion_intensity: "moderate",<br/>  mood_change: 15.0,<br/>  arousal_level: 0.35,<br/>  animation_hint: "purr",<br/>  sound_hint: "purr_soft",<br/>  reaction_triggered: true<br/>}
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
        OBS[Original Observation<br/>8 params: hunger, energy,<br/>dist_food, dist_toy,<br/>mood, lazy_score,<br/>foodie_score, playful_score]
    end
    
    subgraph Config["PERSONALITY_CONFIG (config.py)"]
        BALANCED[Balanced<br/>×1.0 all]
        LAZY[Lazy<br/>energy×1.5, hunger×0.8]
        FOODIE[Foodie<br/>hunger×1.4, energy×0.7]
        PLAYFUL[Playful<br/>dist_toy×0.6, hunger×0.7]
    end
    
    subgraph Processing
        GET[Get Modifiers] --> APPLY[Apply to Observation<br/>Only modifies indices 0-3<br/>mood & scores unchanged]
        APPLY --> CLIP[Clip Values<br/>hunger,energy: 0-100<br/>distances: 0-10<br/>mood: 0-100<br/>scores: 0-100]
    end
    
    subgraph Output
        MODIFIED[Modified Observation<br/>8 params with personality applied]
    end
    
    OBS --> GET
    BALANCED & LAZY & FOODIE & PLAYFUL --> GET
    CLIP --> MODIFIED
    MODIFIED --> MODEL[ML Model<br/>expects 8 features]
    
    style Config fill:#e1f5fe
    style LAZY fill:#b3e5fc
    style FOODIE fill:#ffccbc
    style PLAYFUL fill:#c5e1a5
```

## Training

```mermaid
flowchart LR
    subgraph Development
        CODE[Code Changes] --> CHOOSE{Training Type?}
        CHOOSE -->|Default| TRAIN_DEFAULT["Train Default Brain<br/>python -m src.training.trainer"]
        CHOOSE -->|Individual| TRAIN_CAT["Fine-tune Cat Brain<br/>trainer fine_tune method"]
        TRAIN_DEFAULT --> EVAL[Evaluate Model<br/>10 episodes]
        TRAIN_CAT --> EVAL
    end
    
    subgraph Storage
        EVAL --> SAVE{Save Location?}
        SAVE -->|Default| SAVE_DEFAULT["models/timestamp/"]
        SAVE -->|Individual| SAVE_CAT["models/cats/cat_id/timestamp/"]
        SAVE_DEFAULT --> SYMLINK_DEFAULT["Update symlink<br/>models/latest/"]
        SAVE_CAT --> SYMLINK_CAT["Update symlink<br/>models/cats/cat_id/latest/"]
        SAVE_DEFAULT & SAVE_CAT --> META[Save Metadata<br/>version, reward, cat_id]
    end
    
    subgraph Production
        SYMLINK_DEFAULT & SYMLINK_CAT --> RELOAD{API Running?}
        RELOAD -->|Yes| HOT[Hot Reload<br/>Load new model]
        RELOAD -->|No| COLD[Cold Start<br/>Load on startup]
        HOT --> SERVE[Serve Predictions]
        COLD --> SERVE
    end
    
    style TRAIN_DEFAULT fill:#ffeb3b
    style TRAIN_CAT fill:#ff9800
    style SAVE_DEFAULT fill:#4caf50
    style SAVE_CAT fill:#66bb6a
    style SERVE fill:#2196f3
```

## Living Cat Decision Flow

```mermaid
stateDiagram-v2
    [*] --> ReceiveState: client request
    ReceiveState --> BasePredict: hunger, energy, mood, stimuli
    BasePredict --> ApplyPersonality: personality modifiers
    ApplyPersonality --> MLModel: modified observation
    
    MLModel --> BaseAction: deterministic action
    
    BaseAction --> CheckMemory: cat memory lookup
    CheckMemory --> CalcEmotion: recent activity level
    
    CalcEmotion --> EmotionState: arousal + valence
    EmotionState --> ExtractStimuli: 14 emotion types
    
    ExtractStimuli --> CheckReaction: player_pet, loud_noise, etc
    
    CheckReaction --> ReactionMatch: match (stimulus, emotion)
    
    state ReactionMatch <<choice>>
    ReactionMatch --> ApplyReaction: rule found
    ReactionMatch --> AddNoise: no match
    
    ApplyReaction --> FinalAction: override or bias action
    AddNoise --> RandomQuirk: 20% randomness
    RandomQuirk --> FinalAction
    
    FinalAction --> CheckRepetition: is repeating?
    
    state CheckRepetition <<choice>>
    CheckRepetition --> Distraction: yes, too repetitive
    CheckRepetition --> RecordMemory: no, natural variation
    Distraction --> RecordMemory
    
    RecordMemory --> BuildResponse: store action + mood
    
    BuildResponse --> Response: {<br/>action, emotion,<br/>animation, sound,<br/>mood_change<br/>}
    
    Response --> [*]
    
    note right of EmotionState
        happy, playful, scared,
        grumpy, content, sleepy,
        hungry, anxious, etc
    end note
    
    note right of ApplyReaction
        scared + loud_noise = hide
        happy + petted = purr
        hungry + food_refill = rush
    end note
```

## Component Dependencies

```mermaid
graph TD
    subgraph Core
        ENV[environment.py<br/>Gymnasium Env]
        CONFIG[config.py<br/>Settings + PersonalityConfig]
        EMOTIONS[emotions.py<br/>14 Emotion Types]
        REACTIONS[reactions.py<br/>Stimulus Response Rules]
        BEHAVIOR[behavior.py<br/>Stochastic Patterns]
    end
    
    subgraph Training
        TRAINER[trainer.py<br/>PPO Training + Fine-tuning]
        CALLBACKS[callbacks.py<br/>Logging]
    end
    
    subgraph Services
        CAT_SERVICE[cat_service.py<br/>Cat Management]
        CONTEXTUAL[contextual_engine.py<br/>Living Cat System]
        ACTION_HIST[action_history.py<br/>JSONL Storage]
    end
    
    subgraph Inference
        LOADER[model_loader.py<br/>Load Models]
        PREDICTOR[predictor.py<br/>Batch + Personality]
        CACHE_MOD[cache.py<br/>Redis Cache]
    end
    
    subgraph API
        MAIN[main.py<br/>Entry Point]
        APP[app.py<br/>FastAPI Factory]
        DEPS[dependencies.py<br/>DI Container]
        ROUTES[routes/<br/>predictions, cats, models, monitoring]
        SCHEMAS[schemas.py<br/>Pydantic Models + Stimuli]
        MIDDLEWARE_MOD[middleware.py<br/>Logging]
        HEALTH[health.py<br/>Health Checks]
    end
    
    subgraph Utils
        LOGGER[logger.py<br/>Structlog]
        METRICS_MOD[metrics.py<br/>Prometheus]
    end
    
    CONFIG --> TRAINER
    CONFIG --> PREDICTOR
    CONFIG --> APP
    
    ENV --> TRAINER
    TRAINER --> CALLBACKS
    TRAINER --> CAT_SERVICE
    
    EMOTIONS --> CONTEXTUAL
    REACTIONS --> CONTEXTUAL
    BEHAVIOR --> CONTEXTUAL
    
    LOADER --> PREDICTOR
    LOADER --> CAT_SERVICE
    CACHE_MOD --> PREDICTOR
    
    CONTEXTUAL --> ACTION_HIST
    CAT_SERVICE --> ACTION_HIST
    
    MAIN --> APP
    APP --> ROUTES
    APP --> DEPS
    DEPS --> PREDICTOR
    DEPS --> CAT_SERVICE
    DEPS --> CONTEXTUAL
    ROUTES --> SCHEMAS
    ROUTES --> MIDDLEWARE_MOD
    ROUTES --> HEALTH
    
    LOGGER --> TRAINER
    LOGGER --> ROUTES
    METRICS_MOD --> MIDDLEWARE_MOD
    
    style Core fill:#e3f2fd
    style Training fill:#fff9c4
    style Services fill:#e8f5e9
    style Inference fill:#f3e5f5
    style API fill:#ffe0b2
    style Utils fill:#fce4ec
```
## How to Actually Run This Thing

### Dev Mode

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

python -m src.training.trainer
```

### Docker

```bash
cd docker
docker-compose up --build
```

api: `localhost:8000`  
docs: `localhost:8000/docs`

## Living Cat Features

### 14 emotional states
- **positive**: happy, excited, playful, affectionate, content, relaxed
- **negative**: scared, anxious, grumpy, annoyed  
- **neutral**: curious, sleepy, hungry, demanding

emotions calculated from mood, hunger, energy, arousal

### environmental stimuli
- `player_pet` - being petted
- `player_call` - player calling
- `loud_noise` - environmental noise (0-1)
- `new_toy` - toy appeared
- `food_refill` - bowl refilled
- `player_approach` - player nearby
- `sudden_movement` - unexpected motion

### reactions
emotion + stimulus = unique response:
- happy + petted = purr (85% chance)
- scared + loud_noise = hide (95% chance)
- hungry + food_refill = rush to food (95% chance)
- grumpy + petted = tail flick (60% chance)

### unpredictability
- 20% base randomness
- mood-based variance
- spontaneous quirks (grooming, exploring)
- attention span modifiers
- behavioral patterns (zoomies, lazy sunday, midnight madness)

### memory system
tracks last 50 actions:
- prevents robotic repetition
- calculates activity levels
- introduces distractions if too repetitive

## api usage

### minimal request
```json
POST /predict
{
  "hunger": 60.0,
  "energy": 50.0,
  "distance_to_food": 5.0,
  "distance_to_toy": 10.0,
  "mood": 55.0
}
```

### full living cat request
```json
POST /predict
{
  "cat_id": "whiskers",
  "personality": "playful",
  "hunger": 45.0,
  "energy": 70.0,
  "distance_to_food": 8.0,
  "distance_to_toy": 3.0,
  "mood": 75.0,
  "lazy_score": 30.0,
  "foodie_score": 50.0,
  "playful_score": 80.0,
  
  "player_nearby": true,
  "player_distance": 12.0,
  "is_being_petted": false,
  "is_player_calling": true,
  "loud_noise_level": 0.0,
  "new_toy_appeared": false,
  "food_bowl_refilled": false,
  "sudden_movement": false,
  "time_of_day": "evening"
}
```

### enhanced response
```json
{
  "action": 5,
  "action_name": "play",
  "emotion": "playful",
  "emotion_intensity": "strong",
  "mood_change": 12.0,
  "arousal_level": 0.65,
  "animation_hint": "playful_approach",
  "sound_hint": "chirp",
  "reaction_triggered": true
}
```

## unity integration

### send state
```csharp
var response = await brainService.Predict(new CatState {
    hunger = catStats.hunger,
    energy = catStats.energy,
    mood = catStats.mood,
    is_being_petted = isPetting,
    loud_noise_level = GetEnvironmentNoise(),
    player_nearby = IsPlayerNearby()
});
```

### apply response
```csharp
ExecuteCatAction(response.action);

if (!string.IsNullOrEmpty(response.animation_hint)) {
    animator.Play(response.animation_hint);
}

if (!string.IsNullOrEmpty(response.sound_hint)) {
    audioSource.PlayOneShot(GetSound(response.sound_hint));
}

catStats.mood += response.mood_change;
```

### animation hints
- `purr` - happy purring
- `scared` - frightened
- `excited` - bouncy
- `startle` - sudden jump
- `hide` - run and hide
- `run_to_food` - sprint to bowl
- `playful_approach` - bouncy walk
- `rub_legs` - affection
- `tail_flick` - annoyance
- `ignore` - turn away

### sound hints
- `purr` / `purr_soft` - purring
- `meow_excited` / `meow_annoyed` / `meow_urgent` - various meows
- `meow_response` - acknowledgment
- `hiss` - scared/angry
- `chirp` - playful trill

## Important Stuff You Should Know

### living cat system
model provides base action, then:
1. **emotion engine** calculates emotional state (14 types)
2. **reaction system** checks for stimulus responses  
3. **stochastic layer** adds 20% randomness
4. **memory** prevents repetitive behavior
5. **contextual engine** combines everything

result: unpredictable, lifelike cat behavior

### cache ttl
redis cache set to **5min ttl** (caching base ml predictions only)
reactions and emotions calculated real-time every request

restart redis: `docker restart cat-brain-redis`  
change ttl: `CACHE_TTL` in config (seconds)

### response latencies
- cache hit: ~10-15ms (ml cached, emotions calculated)
- cache miss: ~30-60ms (ml + emotions)
- first request: ~10-15s (model loading)

living cat processing adds ~5-10ms (emotion + reaction calculations)

### model versioning
models saved as `models/<timestamp>/`  
`models/latest/` is symlink  
individual cats: `models/cats/<cat_id>/latest/`

check `metadata.json` for loaded version

### personality modifiers
personalities are multipliers at inference (not trained)  
add custom personalities in [config.py](src/core/config.py)

format:
```python
PERSONALITY_CONFIG = {
    "custom": {
        "hunger": 1.2,
        "energy": 0.9,
        "distance_food": 0.8,
        "distance_toy": 1.1
    }
}
```

### individual brains vs default
request with `cat_id="fluffy"`:
1. checks `models/cats/fluffy/latest/`
2. falls back to `models/latest/`
3. cached separately

individual brains must be trained explicitly

### metrics & monitoring
prometheus metrics at `/metrics`

key metrics:
- `prediction_duration_seconds` - should be <100ms
- `cache_hit_rate` - should be >70%
- `model_load_duration_seconds` - first load 10-15s, cached instant

### logs & debugging
structured json logs (structlog)

```bash
cat logs.json | jq 'select(.level=="error")'
cat logs.json | jq 'select(.cat_id=="fluffy")'
cat logs.json | jq 'select(.event=="reaction_triggered")'
```

set `LOG_LEVEL=DEBUG` for verbose output

### when to retrain
signs you need new model:
- cats doing dumb stuff (sleeping when starving)
- reward plateau in training (check tensorboard)
- added new environment features

otherwise tweak reactions/emotions - faster than retraining

### customizing reactions
edit [reactions.py](src/core/reactions.py):

```python
REACTION_RULES = {
    (StimulusType.PLAYER_PET, EmotionType.HAPPY): ReactionModifier(
        action_probabilities={4: 0.6},  # 60% groom
        mood_delta=15.0,
        animation_hint="purr",
        sound_hint="purr",
        probability=0.85  # 85% chance
    ),
}
```

### adding new emotions
extend [emotions.py](src/core/emotions.py):

```python
class EmotionType(Enum):
    MISCHIEVOUS = "mischievous"

EMOTION_THRESHOLDS = {
    EmotionType.MISCHIEVOUS: {
        "mood_min": 60,
        "energy_min": 70,
        "arousal_min": 0.6
    },
}
```

### tuning randomness
modify [behavior.py](src/core/behavior.py):

```python
randomness = 0.2  # 20% chance of random action
```

higher = more unpredictable, lower = more consistent

### behavioral patterns
automatic patterns trigger based on context:
- **zoomies** - high energy bursts
- **lazy_sunday** - low energy lounging  
- **midnight_madness** - nighttime activity
- **morning_routine** - wake-up behaviors
- **food_obsession** - hunger-driven focus