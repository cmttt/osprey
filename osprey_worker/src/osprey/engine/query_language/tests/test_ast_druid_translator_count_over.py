"""Tests for CountOver lowering in DruidQueryTransformer."""
from typing import Any, Callable, List

import pytest
from osprey.engine.ast import grammar
from osprey.engine.ast_validator.validators.imports_must_not_have_cycles import ImportsMustNotHaveCycles
from osprey.engine.ast_validator.validators.unique_stored_names import UniqueStoredNames
from osprey.engine.ast_validator.validators.validate_call_kwargs import ValidateCallKwargs
from osprey.engine.ast_validator.validators.validate_dynamic_calls_have_annotated_rvalue import (
    ValidateDynamicCallsHaveAnnotatedRValue,
)
from osprey.engine.ast_validator.validators.validate_static_types import ValidateStaticTypes
from osprey.engine.ast_validator.validators.variables_must_be_defined import VariablesMustBeDefined
from osprey.engine.conftest import RunValidationFunction
from osprey.engine.query_language.ast_druid_translator import DruidQueryTransformer
from osprey.engine.query_language.udfs.count_over import CountOver
from osprey.engine.stdlib.udfs.time_delta import TimeDelta
from osprey.engine.udf.registry import UDFRegistry

# Validators and UDF registry setup for CountOver translator tests
pytestmark: List[Callable[[Any], Any]] = [
    pytest.mark.use_validators(
        [
            UniqueStoredNames,
            ValidateStaticTypes,
            ValidateCallKwargs,
            ImportsMustNotHaveCycles,
            ValidateDynamicCallsHaveAnnotatedRValue,
            VariablesMustBeDefined,
        ]
    ),
    pytest.mark.use_udf_registry(UDFRegistry.with_udfs(CountOver, TimeDelta)),
]


def test_count_over_gte_with_key_smoke(
    run_validation: RunValidationFunction,
) -> None:
    """Smoke test: CountOver(...) >= N should produce SQL (tagged shape), not native filter."""
    # Validate the full query expression (rules + query combined)
    validated_sources = run_validation(
        """
A = 'hello'
UserId = 'UserId'
result = CountOver(predicate=A == 'hello', window=TimeDelta(minutes=10), key=UserId) >= 5
"""
    )

    # Debugging: Check what statements we have
    statements = validated_sources.sources.get_entry_point().ast_root.statements
    result_assign = None
    for stmt in statements:
        if isinstance(stmt, grammar.Assign) and stmt.target.identifier == 'result':
            result_assign = stmt
            break

    # For the transformer test, we need to manually pass the CountOver comparison
    # as self._root, bypassing the normal statement[0] logic
    # Create a custom transformer that uses the result assignment
    if result_assign is None:
        pytest.skip("Could not find 'result' assignment in validated sources")

    # Manually construct the transformer with the correct root
    transformer = DruidQueryTransformer(validated_sources=validated_sources)
    # Override the root to use the result assignment's value
    transformer._root = result_assign.value

    transformed_query = transformer.transform()

    # Should be tagged SQL shape, not native filter shape
    assert isinstance(transformed_query, dict)
    assert 'type' in transformed_query, f"Expected 'type' key in response: {transformed_query}"
    assert transformed_query['type'] == 'sql', f"Expected type='sql', got {transformed_query.get('type')}"
    assert 'sql' in transformed_query, f"Expected 'sql' key in response: {transformed_query}"
    assert isinstance(transformed_query['sql'], str)

    # Minimal SQL structure check: should contain SELECT, LAG, and window spec
    sql = transformed_query['sql'].upper()
    assert 'SELECT' in sql, f"SQL should contain SELECT. Got: {transformed_query['sql']}"
    assert 'LAG' in sql, f"SQL should contain LAG window function. Got: {transformed_query['sql']}"
    assert 'OVER' in sql, f"SQL should contain OVER clause. Got: {transformed_query['sql']}"
    assert 'PARTITION BY' in sql, f"SQL should contain PARTITION BY (with key). Got: {transformed_query['sql']}"
