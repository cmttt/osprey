from typing import Dict, Optional

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

    def to_druid_query(self) -> Dict[str, object]:
        # Phase 2 wires Druid lowering through osprey ast_druid_translator.py; until then this guard prevents silent use.
        raise NotImplementedError(
            "CountOver Druid lowering ships in Phase 2; do not call directly."
        )
