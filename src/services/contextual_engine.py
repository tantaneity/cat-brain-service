import time
from dataclasses import dataclass
from typing import Optional

from src.core.behavior import BehaviorLibrary, CatMemory, StochasticBehavior
from src.core.emotions import EmotionEngine, EmotionalState
from src.core.reactions import ReactionSystem, Stimulus, StimulusType
from src.api.schemas import CatState


@dataclass
class EmotionPersistenceState:
    current_emotion: Optional[str] = None
    current_intensity: Optional[str] = None
    changed_at: float = 0.0
    pending_emotion: Optional[str] = None
    pending_votes: int = 0


class ContextualBehaviorEngine:
    
    def __init__(self):
        self.cat_memories: dict[str, CatMemory] = {}
        self.emotion_states: dict[str, EmotionPersistenceState] = {}
        self.default_emotion_hold_seconds = 45.0
        self.pending_votes_required = 3
    
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
        
        emotional_state = EmotionEngine.get_emotional_state(
            mood=state.mood,
            hunger=state.hunger,
            energy=state.energy,
            recent_activity=activity_level,
            noise_level=state.loud_noise_level,
        )
        
        stimuli = self.extract_stimuli(state)
        
        final_action = base_action
        mood_delta = 0.0
        animation_hint = None
        sound_hint = None
        reaction_triggered = False
        
        for stimulus in stimuli:
            reaction = ReactionSystem.get_reaction(stimulus, emotional_state)
            
            if reaction:
                final_action = ReactionSystem.apply_reaction(final_action, reaction)
                mood_delta += reaction.mood_delta
                animation_hint = reaction.animation_hint or animation_hint
                sound_hint = reaction.sound_hint or sound_hint
                reaction_triggered = True
                break

        emotional_state = self._stabilize_emotion(
            cat_id=cat_id or "default",
            candidate=emotional_state,
            stimuli=stimuli,
            reaction_triggered=reaction_triggered,
        )
        
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
            "emotional_state": emotional_state,
            "mood_delta": mood_delta,
            "animation_hint": animation_hint,
            "sound_hint": sound_hint,
            "reaction_triggered": reaction_triggered,
            "activity_level": activity_level,
        }

    def _stabilize_emotion(
        self,
        cat_id: str,
        candidate: EmotionalState,
        stimuli: list[Stimulus],
        reaction_triggered: bool,
    ) -> EmotionalState:
        state = self.emotion_states.get(cat_id)
        if state is None:
            state = EmotionPersistenceState(
                current_emotion=candidate.primary_emotion.value,
                current_intensity=candidate.intensity.value,
                changed_at=time.time(),
            )
            self.emotion_states[cat_id] = state
            return candidate

        current_emotion = state.current_emotion
        next_emotion = candidate.primary_emotion.value
        now = time.time()

        if current_emotion == next_emotion:
            state.pending_emotion = None
            state.pending_votes = 0
            return candidate

        if reaction_triggered or self._has_urgent_stimulus(stimuli):
            state.current_emotion = next_emotion
            state.current_intensity = candidate.intensity.value
            state.changed_at = now
            state.pending_emotion = None
            state.pending_votes = 0
            return candidate

        hold = self.EMOTION_HOLD_SECONDS.get(current_emotion or "", self.default_emotion_hold_seconds)
        if now - state.changed_at < hold:
            return self._copy_emotion(candidate, current_emotion, state.current_intensity)

        if state.pending_emotion == next_emotion:
            state.pending_votes += 1
        else:
            state.pending_emotion = next_emotion
            state.pending_votes = 1

        if state.pending_votes < self.pending_votes_required:
            return self._copy_emotion(candidate, current_emotion, state.current_intensity)

        state.current_emotion = next_emotion
        state.current_intensity = candidate.intensity.value
        state.changed_at = now
        state.pending_emotion = None
        state.pending_votes = 0
        return candidate

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

    @staticmethod
    def _has_urgent_stimulus(stimuli: list[Stimulus]) -> bool:
        urgent_types = {
            StimulusType.LOUD_NOISE,
            StimulusType.SUDDEN_MOVEMENT,
            StimulusType.FOOD_REFILL,
        }
        return any(stim.type in urgent_types and stim.intensity >= 0.6 for stim in stimuli)
