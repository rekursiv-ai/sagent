from os.path import (
    basename as basename,
    dirname as dirname,
    isfile as isfile,
    join as join,
)

from torch._export.db.case import (
    _EXAMPLE_CASES as _EXAMPLE_CASES,
    _EXAMPLE_CONFLICT_CASES as _EXAMPLE_CONFLICT_CASES,
    _EXAMPLE_REWRITE_CASES as _EXAMPLE_REWRITE_CASES,
    ExportCase as ExportCase,
    SupportLevel as SupportLevel,
    export_case as export_case,
)

def all_examples() -> dict[str, ExportCase]: ...

if len(_EXAMPLE_CONFLICT_CASES) > 0:
    def get_name(case) -> str: ...

    msg = ...

def filter_examples_by_support_level(
    support_level: SupportLevel,
) -> dict[str, ExportCase]: ...
def get_rewrite_cases(case) -> list[ExportCase]: ...
