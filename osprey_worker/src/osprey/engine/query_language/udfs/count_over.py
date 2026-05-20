from typing import Dict, Optional

from osprey.engine.ast import grammar
from osprey.engine.ast_validator.validation_context import ValidationContext
from osprey.engine.language_types.time_delta import TimeDeltaT
from osprey.engine.query_language.udfs.registry import register
from osprey.engine.udf.arguments import ArgumentsBase
from osprey.engine.udf.base import QueryUdfBase


class Arguments(ArgumentsBase):
    predicate: bool
    window: TimeDeltaT
    key: Optional[str] = None


@register
class CountOver(QueryUdfBase[Arguments, int]):
    """
    Counts occurrences of a predicate over a time window.

    # Examples

    `CountOver(predicate=UserLoginIp == '1.1.1.1', window=TimeDelta(minutes=10), key=UserId)`
    `CountOver(predicate=Endpoint == '/foo', window=TimeDelta(minutes=1))`
    """

    def __init__(self, validation_context: ValidationContext, arguments: Arguments):
        super().__init__(validation_context, arguments)

        if arguments.key is not None:
            key_node = arguments.get_argument_ast('key')
            if isinstance(key_node, grammar.Name):
                self.key = key_node.identifier
            else:
                self.key = None
                validation_context.add_error(
                    message='expected column reference',
                    span=key_node.span,
                    hint='argument `key` must be a column reference',
                )
        else:
            self.key = None

    def to_druid_query(self) -> Dict[str, object]:
        raise NotImplementedError(
            "CountOver Druid lowering ships in Phase 2; do not call directly."
        )
