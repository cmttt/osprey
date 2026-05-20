from typing import Any, Callable, List

import pytest
from osprey.engine.ast_validator.validators.unique_stored_names import UniqueStoredNames
from osprey.engine.ast_validator.validators.validate_call_kwargs import ValidateCallKwargs
from osprey.engine.ast_validator.validators.validate_dynamic_calls_have_annotated_rvalue import (
    ValidateDynamicCallsHaveAnnotatedRValue,
)
from osprey.engine.conftest import RunValidationFunction
from osprey.engine.query_language.udfs.count_over import CountOver
from osprey.engine.stdlib.udfs.time_delta import TimeDelta
from osprey.engine.udf.registry import UDFRegistry

pytestmark: List[Callable[[Any], Any]] = [
    pytest.mark.use_validators([ValidateCallKwargs, ValidateDynamicCallsHaveAnnotatedRValue, UniqueStoredNames]),
    pytest.mark.use_udf_registry(UDFRegistry.with_udfs(CountOver, TimeDelta)),
]


def test_count_over_with_key(run_validation: RunValidationFunction) -> None:
    run_validation("CountOver(predicate=UserLoginIp == '1.1.1.1', window=TimeDelta(minutes=10), key=UserId)")


def test_count_over_without_key(run_validation: RunValidationFunction) -> None:
    run_validation("CountOver(predicate=Endpoint == '/foo', window=TimeDelta(minutes=1))")


def test_count_over_to_druid_query_raises_not_implemented(run_validation: RunValidationFunction) -> None:
    from osprey.engine.ast import grammar
    from osprey.engine.ast.ast_utils import filter_nodes

    validated_sources = run_validation(
        "CountOver(predicate=UserLoginIp == '1.1.1.1', window=TimeDelta(minutes=10), key=UserId)"
    )

    udf_mapping = validated_sources.get_validator_result(ValidateCallKwargs)

    source = validated_sources.sources.get_entry_point()
    count_over_call = None
    for call_node in filter_nodes(source.ast_root, grammar.Call):
        if isinstance(call_node.func, grammar.Name) and call_node.func.identifier == 'CountOver':
            count_over_call = call_node
            break

    assert count_over_call is not None, "CountOver call node not found in AST"

    count_over_udf, _ = udf_mapping[id(count_over_call)]
    assert isinstance(count_over_udf, CountOver)

    with pytest.raises(NotImplementedError, match="CountOver Druid lowering ships in Phase 2"):
        count_over_udf.to_druid_query()
