from src.api.schemas import CatState
from src.core.behavior import BehaviorLibrary, StochasticBehavior
from src.core.environment import CatAction
from src.services.contextual_engine import ContextualBehaviorEngine


def _build_state(**overrides) -> CatState:
    data = {
        "cat_id": "cat-1",
        "hunger": 40.0,
        "energy": 80.0,
        "distance_to_food": 30.0,
        "distance_to_toy": 15.0,
        "distance_to_bed": 20.0,
        "mood": 60.0,
        "lazy_score": 20.0,
        "foodie_score": 40.0,
        "playful_score": 80.0,
        "is_bowl_empty": False,
        "is_bowl_tipped": False,
        "player_nearby": False,
        "player_distance": 100.0,
        "is_being_petted": False,
        "is_player_calling": False,
        "loud_noise_level": 0.0,
        "new_toy_appeared": False,
        "food_bowl_refilled": False,
        "sudden_movement": False,
        "laser_distance": 7.0,
        "laser_velocity": 0.0,
        "laser_visible": True,
        "laser_active": True,
        "laser_play_skill": 0.0,
        "laser_caught": False,
        "time_of_day": "afternoon",
    }
    data.update(overrides)
    return CatState(**data)


def _patch_behavior_noise(monkeypatch):
    monkeypatch.setattr(
        BehaviorLibrary,
        "get_random_quirk_action",
        staticmethod(lambda mood, energy: None),
    )
    monkeypatch.setattr(
        StochasticBehavior,
        "add_noise_to_prediction",
        staticmethod(lambda action, confidence=0.75, mood=50.0: action),
    )
    monkeypatch.setattr(
        StochasticBehavior,
        "introduce_distraction",
        staticmethod(lambda action, environment_richness=0.6: action),
    )


def test_visible_laser_with_high_interest_biases_chase_or_play(monkeypatch):
    _patch_behavior_noise(monkeypatch)
    monkeypatch.setattr("src.services.contextual_engine.random.uniform", lambda a, b: 0.0)
    engine = ContextualBehaviorEngine()

    far_laser_result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(laser_distance=7.0),
        cat_id="cat-1",
    )
    close_laser_result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(laser_distance=1.8),
        cat_id="cat-1",
    )

    assert far_laser_result["action"] == CatAction.MOVE_TO_TOY
    assert close_laser_result["action"] == CatAction.PLAY


def test_low_energy_suppresses_laser_chase(monkeypatch):
    _patch_behavior_noise(monkeypatch)
    engine = ContextualBehaviorEngine()

    result = engine.process_action(
        base_action=CatAction.MOVE_TO_TOY,
        state=_build_state(energy=20.0),
        cat_id="cat-2",
    )

    assert result["action"] == CatAction.IDLE


def test_invisible_active_laser_triggers_short_search_then_fallback(monkeypatch):
    _patch_behavior_noise(monkeypatch)
    engine = ContextualBehaviorEngine()

    engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(laser_visible=True),
        cat_id="cat-3",
    )

    search_result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(laser_visible=False),
        cat_id="cat-3",
    )

    assert search_result["action"] == CatAction.EXPLORE

    engine.laser_last_seen["cat-3"] = 0.0
    fallback_result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(laser_visible=False),
        cat_id="cat-3",
    )

    assert fallback_result["action"] == CatAction.IDLE


def test_activation_grace_biases_laser_chase_with_low_interest(monkeypatch):
    _patch_behavior_noise(monkeypatch)
    monkeypatch.setattr("src.services.contextual_engine.random.uniform", lambda a, b: 0.0)
    engine = ContextualBehaviorEngine()

    grace_result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(
            playful_score=5.0,
            lazy_score=95.0,
            laser_distance=5.0,
            laser_visible=True,
        ),
        cat_id="cat-4",
    )
    assert grace_result["action"] == CatAction.MOVE_TO_TOY

    engine.laser_last_active["cat-4"] = True
    engine.laser_activated_at["cat-4"] = 0.0
    no_grace_result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(
            playful_score=5.0,
            lazy_score=95.0,
            laser_distance=5.0,
            laser_visible=True,
        ),
        cat_id="cat-4",
    )
    assert no_grace_result["action"] == CatAction.IDLE


def test_higher_laser_skill_reduces_prediction_error(monkeypatch):
    monkeypatch.setattr("src.services.contextual_engine.random.uniform", lambda a, b: 0.2)
    engine = ContextualBehaviorEngine()

    low_skill_error = engine._calculate_laser_prediction_error(0.0)
    high_skill_error = engine._calculate_laser_prediction_error(0.9)

    assert abs(high_skill_error) < abs(low_skill_error)


def test_player_call_biases_social_explore(monkeypatch):
    _patch_behavior_noise(monkeypatch)
    engine = ContextualBehaviorEngine()

    result = engine.process_action(
        base_action=CatAction.IDLE,
        state=_build_state(
            is_player_calling=True,
            player_nearby=False,
            player_distance=70.0,
            energy=75.0,
            laser_active=False,
            laser_visible=False,
        ),
        cat_id="cat-call-1",
    )

    assert result["action"] == CatAction.EXPLORE


def test_low_energy_overrides_player_call(monkeypatch):
    _patch_behavior_noise(monkeypatch)
    engine = ContextualBehaviorEngine()

    result = engine.process_action(
        base_action=CatAction.EXPLORE,
        state=_build_state(
            is_player_calling=True,
            energy=12.0,
            laser_active=False,
            laser_visible=False,
        ),
        cat_id="cat-call-2",
    )

    assert result["action"] == CatAction.IDLE
