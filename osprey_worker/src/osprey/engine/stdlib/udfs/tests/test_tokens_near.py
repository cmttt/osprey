from typing import Any, Callable, List

import pytest
from osprey.engine.ast_validator.validators.unique_stored_names import UniqueStoredNames
from osprey.engine.ast_validator.validators.validate_call_kwargs import ValidateCallKwargs
from osprey.engine.conftest import ExecuteFunction
from osprey.engine.stdlib.udfs.regex_match import TokensNear
from osprey.engine.udf.registry import UDFRegistry

pytestmark: List[Callable[[Any], Any]] = [
    pytest.mark.use_validators([ValidateCallKwargs, UniqueStoredNames]),
    pytest.mark.use_udf_registry(UDFRegistry.with_udfs(TokensNear)),
]

# first term set: sexual/commercial; second term set: a minor age token.
A = r'sex\w*|nudes?|porn|sell\w*'
B = r'[6-9]|1[0-7]'


@pytest.mark.parametrize(
    'text, gap, expected',
    (
        ('sex 14', 1, True),
        ('15 nudes', 1, True),
        ('sexting 13 here', 1, True),
        ('selling 16', 1, True),
        # per-token fullmatch: "sex" must NOT match inside another word
        ('sussex 15', 1, False),
        ('essex 14', 1, False),
        ('unisex 16', 1, False),
        # gap is respected
        ('sex random words 15', 1, False),
        ('sex random 15', 2, True),
        # needs BOTH a and b
        ('sex sells', 1, False),
        ('14 15 16', 1, False),
    ),
)
def test_tokens_near(execute: ExecuteFunction, text: str, gap: int, expected: bool) -> None:
    result = execute(
        f"""
    Near = TokensNear(targets=["{text}"], pattern_a=r"{A}", pattern_b=r"{B}", max_gap={gap})
    """
    )
    assert result == {'Near': expected}


def test_searches_all_targets(execute: ExecuteFunction) -> None:
    result = execute(
        f"""
    Near = TokensNear(targets=["benign here", "sex 15"], pattern_a=r"{A}", pattern_b=r"{B}")
    """
    )
    assert result == {'Near': True}
