"""Tests for the latency_tier class attribute on UDFBase."""
from osprey.engine.stdlib.udfs.json_data import JsonData
from osprey.engine.stdlib.udfs.rules import Rule, WhenRules
from osprey.engine.stdlib.udfs.execution_mode import ExecutionMode
from osprey.engine.udf.base import UDFBase


def test_udfbase_defaults_latency_tier_fast():
    """All UDFs default to latency_tier='fast' — opt-in to 'slow' is required."""
    assert UDFBase.latency_tier == 'fast'


def test_existing_udfs_default_to_fast():
    """Existing UDFs must not require explicit annotation."""
    assert JsonData.latency_tier == 'fast'
    assert Rule.latency_tier == 'fast'
    assert WhenRules.latency_tier == 'fast'
    assert ExecutionMode.latency_tier == 'fast'


def test_subclass_can_override_to_slow():
    """A UDF subclass can explicitly tag itself slow."""

    class MySlowUDF(UDFBase):  # type: ignore[type-arg]
        latency_tier = 'slow'

        def execute(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

    assert MySlowUDF.latency_tier == 'slow'
