import time
from dataclasses import dataclass
from typing import Optional

from src.core.behavior import BehaviorLibrary, CatMemory, StochasticBehavior
from src.core.emotions import EmotionEngine, EmotionalState, EmotionType, BehaviorIntensity
from src.core.reactions import ReactionSystem, Stimulus, StimulusType, ReactionModifier
from src.api.schemas import CatState, EmotionAxis, EmotionAxes, VisualLayer


@dataclass
class AxisPersistenceState:
    current_emotion: Optional[str] = None
    current_intensity: Optional[str] = None
    changed_at: float = 0.0
    pending_emotion: Optional[str] = None
    pending_votes: int = 0


@dataclass
class ReactionAxisState:
    emotion: Optional[str] = None
    intensity: Optional[str] = None
    arousal: float = 0.0
    valence: float = 0.0
    updated_at: float = 0.0
    expires_at: float = 0.0


class ContextualBehaviorEngine:
    
    def __init__(self):
        self.cat_memories: dict[str, CatMemory] = {}
        self.axis_states: dict[str, dict[str, AxisPersistenceState]] = {}
        self.reaction_states: dict[str, ReactionAxisState] = {}
        self.mood_ema: dict[str, float] = {}
        self.default_emotion_hold_seconds = 45.0
        self.pending_votes_required = 3
        self.axis_pending_votes = {
            "base": 3,
            "mood": 2,
        }
        self.mood_ema_alpha = 0.2
    
    EMOTION_HOLD_SECONDS = {
        "sleepy": 240.0,
        "relaxed": 180.0,
        "content": 120.0,
        "hungry": 120.0,
        "anxious": 90.0,
        "scared": 20.0,
        "annoyed": 45.0,
        "happy": 60.0,
        "playful": 60.0,
        "curious": 45.0,
        "grumpy": 90.0,
        "demanding": 60.0,
        "affectionate": 60.0,
        "excited": 30.0,
    }

    BASE_EMOTION_HOLD_SECONDS = {key: value * 2 for key, value in EMOTION_HOLD_SECONDS.items()}

    AXIS_WEIGHTS = {
        "base": 0.4,
        "mood": 0.6,
        "reaction": 1.0,
    }

    INTENSITY_WEIGHTS = {
        "subtle": 0.4,
        "moderate": 0.6,
        "strong": 0.8,
        "intense": 1.0,
    }

    REACTION_EMOTION_BY_STIMULUS = {
        StimulusType.PLAYER_PET: EmotionType.AFFECTIONATE,
        StimulusType.PLAYER_CALL: EmotionType.CURIOUS,
        StimulusType.PLAYER_APPROACH: EmotionType.CURIOUS,
        StimulusType.LOUD_NOISE: EmotionType.SCARED,
        StimulusType.NEW_TOY: EmotionType.PLAYFUL,
        StimulusType.FOOD_REFILL: EmotionType.EXCITED,
        StimulusType.DOOR_OPEN: EmotionType.CURIOUS,
        StimulusType.SUDDEN_MOVEMENT: EmotionType.ANXIOUS,
        StimulusType.UNKNOWN_PERSON: EmotionType.SCARED,
    }

    REACTION_VALENCE = {
        EmotionType.CONTENT: 0.3,
        EmotionType.HAPPY: 0.8,
        EmotionType.EXCITED: 0.9,
        EmotionType.PLAYFUL: 0.7,
        EmotionType.CURIOUS: 0.3,
        EmotionType.RELAXED: 0.4,
        EmotionType.SLEEPY: 0.0,
        EmotionType.HUNGRY: -0.4,
        EmotionType.GRUMPY: -0.6,
        EmotionType.ANNOYED: -0.6,
        EmotionType.SCARED: -0.9,
        EmotionType.ANXIOUS: -0.7,
        EmotionType.AFFECTIONATE: 0.8,
        EmotionType.DEMANDING: -0.5,
    }
    
    def get_or_create_memory(self, cat_id: str) -> CatMemory:
        if cat_id not in self.cat_memories:
            self.cat_memories[cat_id] = CatMemory()
        return self.cat_memories[cat_id]
    
    def determine_time_modifier(self, time_of_day: str) -> float:
        time_energy_map = {
            "morning": 1.1,
            "afternoon": 1.0,
            "evening": 0.9,
            "night": 0.7,
        }
        return time_energy_map.get(time_of_day, 1.0)
    
    def extract_stimuli(self, state: CatState) -> list[Stimulus]:
        stimuli = []
        
        if state.is_being_petted:
            stimuli.append(Stimulus(
                type=StimulusType.PLAYER_PET,
                intensity=1.0,
            ))
        
        if state.is_player_calling:
            stimuli.append(Stimulus(
                type=StimulusType.PLAYER_CALL,
                intensity=1.0,
            ))
        
        if state.loud_noise_level > 0.3:
            stimuli.append(Stimulus(
                type=StimulusType.LOUD_NOISE,
                intensity=state.loud_noise_level,
            ))
        
        if state.new_toy_appeared:
            stimuli.append(Stimulus(
                type=StimulusType.NEW_TOY,
                intensity=1.0,
            ))
        
        if state.food_bowl_refilled:
            stimuli.append(Stimulus(
                type=StimulusType.FOOD_REFILL,
                intensity=1.0,
            ))
        
        if state.player_nearby and state.player_distance < 20:
            stimuli.append(Stimulus(
                type=StimulusType.PLAYER_APPROACH,
                intensity=1.0 - (state.player_distance / 20),
            ))
        
        if state.sudden_movement:
            stimuli.append(Stimulus(
                type=StimulusType.SUDDEN_MOVEMENT,
                intensity=0.8,
            ))
        
        return stimuli
    
    def process_action(
        self,
        base_action: int,
        state: CatState,
        cat_id: Optional[str] = None,
    ) -> dict:
        memory = self.get_or_create_memory(cat_id or "default")
        
        activity_level = memory.get_recent_activity_level()
        
        base_mood = max(0.0, min(100.0, (state.energy + (100.0 - state.hunger)) / 2.0))
        mood_ema = self._update_mood_ema(cat_id or "default", state.mood)
        
        base_candidate = EmotionEngine.get_emotional_state(
            mood=base_mood,
            hunger=state.hunger,
            energy=state.energy,
            recent_activity=activity_level,
            noise_level=state.loud_noise_level,
        )
        
        mood_candidate = EmotionEngine.get_emotional_state(
            mood=mood_ema,
            hunger=state.hunger,
            energy=state.energy,
            recent_activity=activity_level,
            noise_level=state.loud_noise_level,
        )
        
        base_state = self._stabilize_axis(
            cat_id=cat_id or "default",
            axis="base",
            candidate=base_candidate,
        )
        mood_state = self._stabilize_axis(
            cat_id=cat_id or "default",
            axis="mood",
            candidate=mood_candidate,
        )
        
        stimuli = self.extract_stimuli(state)
        
        final_action = base_action
        mood_delta = 0.0
        animation_hint = None
        sound_hint = None
        reaction_triggered = False
        reaction_modifier: Optional[ReactionModifier] = None
        reaction_stimulus: Optional[Stimulus] = None
        
        for stimulus in stimuli:
            reaction = ReactionSystem.get_reaction(stimulus, mood_state)
            
            if reaction:
                final_action = ReactionSystem.apply_reaction(final_action, reaction)
                mood_delta += reaction.mood_delta
                animation_hint = reaction.animation_hint or animation_hint
                sound_hint = reaction.sound_hint or sound_hint
                reaction_triggered = True
                reaction_modifier = reaction
                reaction_stimulus = stimulus
                break
        
        reaction_axis = self._update_reaction_axis(
            cat_id=cat_id or "default",
            reaction=reaction_modifier,
            stimulus=reaction_stimulus,
            mood_state=mood_state,
        )
        
        emotion_axes = self._build_emotion_axes(
            cat_id=cat_id or "default",
            base_state=base_state,
            mood_state=mood_state,
            reaction_state=reaction_axis,
        )
        visual_layers, visual_primary = self._build_visual_layers(emotion_axes)
        
        if not reaction_triggered:
            quirk_action = BehaviorLibrary.get_random_quirk_action(
                state.mood, state.energy
            )
            if quirk_action is not None:
                final_action = quirk_action
        
        if not reaction_triggered:
            final_action = StochasticBehavior.add_noise_to_prediction(
                final_action,
                confidence=0.75,
                mood=state.mood,
            )
        
        if memory.is_repeating_behavior():
            final_action = StochasticBehavior.introduce_distraction(
                final_action,
                environment_richness=0.6,
            )
        
        memory.record_action(final_action, state.mood)
        
        return {
            "action": final_action,
            "emotional_state": mood_state,
            "mood_delta": mood_delta,
            "animation_hint": animation_hint,
            "sound_hint": sound_hint,
            "reaction_triggered": reaction_triggered,
            "activity_level": activity_level,
            "emotion_axes": emotion_axes,
            "visual_layers": visual_layers,
            "visual_primary": visual_primary,
        }

    def _update_mood_ema(self, cat_id: str, mood: float) -> float:
        previous = self.mood_ema.get(cat_id, mood)
        alpha = self.mood_ema_alpha
        updated = previous * (1 - alpha) + mood * alpha
        self.mood_ema[cat_id] = updated
        return updated

    def _get_axis_state(self, cat_id: str, axis: str) -> AxisPersistenceState:
        axis_map = self.axis_states.setdefault(cat_id, {})
        state = axis_map.get(axis)
        if state is None:
            state = AxisPersistenceState()
            axis_map[axis] = state
        return state

    def _stabilize_axis(
        self,
        cat_id: str,
        axis: str,
        candidate: EmotionalState,
    ) -> EmotionalState:
        state = self._get_axis_state(cat_id, axis)
        now = time.time()

        if state.current_emotion is None:
            state.current_emotion = candidate.primary_emotion.value
            state.current_intensity = candidate.intensity.value
            state.changed_at = now
            state.pending_emotion = None
            state.pending_votes = 0
            return candidate

        current_emotion = state.current_emotion
        next_emotion = candidate.primary_emotion.value

        if current_emotion == next_emotion:
            state.pending_emotion = None
            state.pending_votes = 0
            return candidate

        hold_map = self.BASE_EMOTION_HOLD_SECONDS if axis == "base" else self.EMOTION_HOLD_SECONDS
        hold = hold_map.get(current_emotion or "", self.default_emotion_hold_seconds)
        if now - state.changed_at < hold:
            return self._copy_emotion(candidate, current_emotion, state.current_intensity)

        if state.pending_emotion == next_emotion:
            state.pending_votes += 1
        else:
            state.pending_emotion = next_emotion
            state.pending_votes = 1

        required = self.axis_pending_votes.get(axis, self.pending_votes_required)
        if state.pending_votes < required:
            return self._copy_emotion(candidate, current_emotion, state.current_intensity)

        state.current_emotion = next_emotion
        state.current_intensity = candidate.intensity.value
        state.changed_at = now
        state.pending_emotion = None
        state.pending_votes = 0
        return candidate

    def _update_reaction_axis(
        self,
        cat_id: str,
        reaction: Optional[ReactionModifier],
        stimulus: Optional[Stimulus],
        mood_state: EmotionalState,
    ) -> Optional[ReactionAxisState]:
        now = time.time()
        if reaction and stimulus:
            reaction_emotion = reaction.reaction_emotion or self.REACTION_EMOTION_BY_STIMULUS.get(
                stimulus.type,
                mood_state.primary_emotion,
            )
            reaction_intensity = reaction.reaction_intensity or self._intensity_from_stimulus(
                stimulus.intensity
            )
            duration = reaction.reaction_duration if reaction.reaction_duration > 0 else (3.0 + 2.0 * stimulus.intensity)
            arousal = min(1.0, mood_state.arousal_level + reaction.arousal_boost)
            valence = self.REACTION_VALENCE.get(reaction_emotion, mood_state.valence)

            state = ReactionAxisState(
                emotion=reaction_emotion.value,
                intensity=reaction_intensity.value,
                arousal=arousal,
                valence=valence,
                updated_at=now,
                expires_at=now + duration,
            )
            self.reaction_states[cat_id] = state
            return state

        return self._get_active_reaction_axis(cat_id, now)

    def _get_active_reaction_axis(self, cat_id: str, now: float) -> Optional[ReactionAxisState]:
        state = self.reaction_states.get(cat_id)
        if state and now < state.expires_at:
            return state
        if state:
            self.reaction_states.pop(cat_id, None)
        return None

    def _build_emotion_axes(
        self,
        cat_id: str,
        base_state: EmotionalState,
        mood_state: EmotionalState,
        reaction_state: Optional[ReactionAxisState],
    ) -> EmotionAxes:
        base_meta = self.axis_states.get(cat_id, {}).get("base")
        mood_meta = self.axis_states.get(cat_id, {}).get("mood")

        base_axis = EmotionAxis(
            emotion=base_state.primary_emotion.value,
            intensity=base_state.intensity.value,
            arousal=base_state.arousal_level,
            valence=base_state.valence,
            updated_at=base_meta.changed_at if base_meta else time.time(),
            expires_at=None,
            source="base",
        )
        mood_axis = EmotionAxis(
            emotion=mood_state.primary_emotion.value,
            intensity=mood_state.intensity.value,
            arousal=mood_state.arousal_level,
            valence=mood_state.valence,
            updated_at=mood_meta.changed_at if mood_meta else time.time(),
            expires_at=None,
            source="mood",
        )

        reaction_axis = None
        if reaction_state is not None and reaction_state.emotion and reaction_state.intensity:
            reaction_axis = EmotionAxis(
                emotion=reaction_state.emotion,
                intensity=reaction_state.intensity,
                arousal=reaction_state.arousal,
                valence=reaction_state.valence,
                updated_at=reaction_state.updated_at,
                expires_at=reaction_state.expires_at,
                source="reaction",
            )

        return EmotionAxes(base=base_axis, mood=mood_axis, reaction=reaction_axis)

    def _build_visual_layers(
        self,
        axes: EmotionAxes,
    ) -> tuple[list[VisualLayer], Optional[str]]:
        layers: list[VisualLayer] = []

        def add_layer(axis: EmotionAxis, priority: int):
            axis_weight = self.AXIS_WEIGHTS.get(axis.source, 0.5)
            intensity_factor = self.INTENSITY_WEIGHTS.get(axis.intensity, 0.6)
            weight = max(0.0, min(1.0, axis_weight * intensity_factor))
            layers.append(VisualLayer(
                source=axis.source,
                emotion=axis.emotion,
                intensity=axis.intensity,
                priority=priority,
                weight=weight,
                expires_at=axis.expires_at,
            ))

        add_layer(axes.base, 1)
        add_layer(axes.mood, 2)
        if axes.reaction is not None:
            add_layer(axes.reaction, 3)

        primary = None
        if layers:
            primary_layer = max(layers, key=lambda layer: (layer.priority, layer.weight))
            primary = primary_layer.emotion

        return layers, primary

    @staticmethod
    def _intensity_from_stimulus(intensity: float) -> BehaviorIntensity:
        if intensity >= 0.85:
            return BehaviorIntensity.INTENSE
        if intensity >= 0.6:
            return BehaviorIntensity.STRONG
        if intensity >= 0.35:
            return BehaviorIntensity.MODERATE
        return BehaviorIntensity.SUBTLE

    @staticmethod
    def _copy_emotion(candidate: EmotionalState, emotion: Optional[str], intensity: Optional[str]) -> EmotionalState:
        if not emotion:
            return candidate

        primary = candidate.primary_emotion
        level = candidate.intensity
        for e in primary.__class__:
            if e.value == emotion:
                primary = e
                break
        for i in level.__class__:
            if i.value == intensity:
                level = i
                break

        return EmotionalState(
            primary_emotion=primary,
            intensity=level,
            mood_value=candidate.mood_value,
            arousal_level=candidate.arousal_level,
            valence=candidate.valence,
        )
