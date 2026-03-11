import json

from src.services.cat_profile_store import CatProfileStore


def test_lazy_profile_has_stronger_lazy_than_playful(tmp_path):
    store = CatProfileStore(tmp_path / "cats")
    profile = store.create_profile("cat-lazy", "lazy")

    assert profile.version == CatProfileStore.CURRENT_PROFILE_VERSION
    assert profile.modifiers["lazy_score"] > profile.modifiers["playful_score"]
    assert profile.modifiers["lazy_score"] - profile.modifiers["playful_score"] >= 0.2


def test_playful_profile_has_stronger_playful_than_lazy(tmp_path):
    store = CatProfileStore(tmp_path / "cats")
    profile = store.create_profile("cat-playful", "playful")

    assert profile.version == CatProfileStore.CURRENT_PROFILE_VERSION
    assert profile.modifiers["playful_score"] > profile.modifiers["lazy_score"]
    assert profile.modifiers["playful_score"] - profile.modifiers["lazy_score"] >= 0.2


def test_existing_profile_is_upgraded_to_current_version(tmp_path):
    base_path = tmp_path / "cats"
    profile_path = base_path / "cat-old" / "profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "cat_id": "cat-old",
                "personality": "lazy",
                "created_at": "2025-01-01T00:00:00",
                "seed": 1337,
                "version": 1,
                "modifiers": {
                    "lazy_score": 0.99,
                    "foodie_score": 0.98,
                    "playful_score": 0.97,
                },
            },
            file,
        )

    store = CatProfileStore(base_path)
    profile = store.get_profile("cat-old")

    assert profile is not None
    assert profile.version == CatProfileStore.CURRENT_PROFILE_VERSION
    assert profile.modifiers["lazy_score"] > profile.modifiers["playful_score"]


def test_version2_profile_uses_soft_blend_on_upgrade(tmp_path):
    base_path = tmp_path / "cats"
    profile_path = base_path / "cat-v2" / "profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "cat_id": "cat-v2",
                "personality": "playful",
                "created_at": "2025-01-01T00:00:00",
                "seed": 2024,
                "version": 2,
                "modifiers": {
                    "lazy_score": 0.95,
                    "foodie_score": 0.94,
                    "playful_score": 0.96,
                },
            },
            file,
        )

    store = CatProfileStore(base_path)
    upgraded = store.get_profile("cat-v2")

    assert upgraded is not None
    assert upgraded.version == CatProfileStore.CURRENT_PROFILE_VERSION
    # Soft blend should preserve old neighborhood while nudging toward stronger playful profile.
    assert 0.85 <= upgraded.modifiers["playful_score"] <= 1.35
