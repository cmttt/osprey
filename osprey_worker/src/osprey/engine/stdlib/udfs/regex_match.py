import re
from typing import List

from ._prelude import ArgumentsBase, ConstExpr, ExecutionContext, UDFBase, ValidationContext
from .categories import UdfCategories


class RegexArgumentsBase(ArgumentsBase):
    pattern: ConstExpr[str]
    """The regex pattern to evaluate."""

    case_insensitive: ConstExpr[bool] = ConstExpr.for_default('case_insensitive', False)
    """Optional: If `True`, ignores case. Default is `False`."""


class RegexUDFBase:
    def __init__(self, validation_context: 'ValidationContext', arguments: RegexArgumentsBase):
        super().__init__(validation_context, arguments)  # type: ignore

        flags = 0

        if arguments.case_insensitive.value:
            flags |= re.IGNORECASE

        with arguments.pattern.attribute_errors(message='invalid regex pattern'):
            self._compiled = re.compile(arguments.pattern.value, flags)


class RegexMatchArguments(RegexArgumentsBase):
    target: str
    """A target string to evaluate the regex pattern on."""


class RegexMatch(RegexUDFBase, UDFBase[RegexMatchArguments, bool]):
    """Returns `True` if a match for the provided regex is found."""

    category = UdfCategories.STRING

    def execute(self, execution_context: ExecutionContext, arguments: RegexMatchArguments) -> bool:
        return self._compiled.search(arguments.target) is not None


class RegexMatchMapArguments(RegexArgumentsBase):
    target: List[str]
    """A target string to evaluate the regex pattern on."""

    mode: ConstExpr[str] = ConstExpr.for_default('mode', 'any')
    """Are `all` or `any` matches required?"""


class RegexMatchMap(RegexUDFBase, UDFBase[RegexMatchMapArguments, bool]):
    """Returns `True` if a match for the provided regex is found."""

    def __init__(self, validation_context: 'ValidationContext', arguments: RegexMatchMapArguments):
        super().__init__(validation_context, arguments)

        mode = arguments.mode.value
        if mode not in ('all', 'any'):
            validation_context.add_error(
                message=f'mode must be one of `all` or `any`, not `{mode}`',
                span=arguments.mode.argument_span,
            )

        self._op = all if mode == 'all' else any

    def execute(self, execution_context: ExecutionContext, arguments: RegexMatchMapArguments) -> bool:
        return self._op(self._compiled.search(target) is not None for target in arguments.target)


_TOKEN_RE = re.compile(r'\w+')


class TokensNearArguments(ArgumentsBase):
    targets: List[str]
    """Text fields to search, e.g. [GuildNameLower, GuildDescriptionLower]."""

    pattern_a: ConstExpr[str]
    """Regex for the first term set (matched per-token), e.g. r'sex\\w*|nudes?|porn'."""

    pattern_b: ConstExpr[str]
    """Regex for the second term set (matched per-token), e.g. r'[6-9]|1[0-7]'."""

    max_gap: ConstExpr[int] = ConstExpr.for_default('max_gap', 1)
    """Max tokens allowed between an `a` token and a `b` token (1 = directly adjacent)."""

    case_insensitive: ConstExpr[bool] = ConstExpr.for_default('case_insensitive', True)
    """Optional: if `True` (default), ignores case."""


class TokensNear(UDFBase[TokensNearArguments, bool]):
    """Returns `True` if a token fully matching `pattern_a` occurs within `max_gap` tokens
    of a token fully matching `pattern_b`, in any target.

    Proximity over a tokenized string. Per-token `fullmatch` means a pattern only matches a
    WHOLE word -- so "sex" does not match inside "sussex"/"unisex" -- which avoids both the
    substring false positives and the bidirectional-duplication / word-boundary fragility of
    hand-written "term-near-term" regexes.
    """

    category = UdfCategories.STRING

    def __init__(self, validation_context: 'ValidationContext', arguments: TokensNearArguments):
        super().__init__(validation_context, arguments)

        flags = re.IGNORECASE if arguments.case_insensitive.value else 0
        with arguments.pattern_a.attribute_errors(message='invalid pattern_a'):
            self._a = re.compile(arguments.pattern_a.value, flags)
        with arguments.pattern_b.attribute_errors(message='invalid pattern_b'):
            self._b = re.compile(arguments.pattern_b.value, flags)

        self._max_gap = arguments.max_gap.value
        if self._max_gap < 1:
            validation_context.add_error(
                message='max_gap must be >= 1',
                span=arguments.max_gap.argument_span,
            )

    def execute(self, execution_context: ExecutionContext, arguments: TokensNearArguments) -> bool:
        for target in arguments.targets:
            tokens = _TOKEN_RE.findall(target)
            a_idx = [i for i, tok in enumerate(tokens) if self._a.fullmatch(tok)]
            if not a_idx:
                continue
            b_idx = [i for i, tok in enumerate(tokens) if self._b.fullmatch(tok)]
            for i in a_idx:
                for j in b_idx:
                    if i != j and abs(i - j) <= self._max_gap:
                        return True
        return False
