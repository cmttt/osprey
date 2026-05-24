"""Tests for the mutates_state class attribute on EffectBase."""
from osprey.engine.language_types.effects import EffectBase
from osprey.engine.language_types.labels import LabelEffect
from osprey.engine.language_types.verdicts import VerdictEffect


def test_effect_base_defaults_mutates_state_false():
    """Default is False — explicit annotation required for state-mutating effects."""
    assert EffectBase.mutates_state is False


def test_label_effect_mutates_state_true():
    """LabelEffect writes to the labels service => mutates_state=True."""
    assert LabelEffect.mutates_state is True


def test_verdict_effect_does_not_mutate_state():
    """Verdicts are a response channel, not persistent state. Must be False."""
    assert VerdictEffect.mutates_state is False
