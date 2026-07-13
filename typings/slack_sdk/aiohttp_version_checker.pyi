from collections.abc import Callable

"""Internal module for checking aiohttp compatibility of async modules"""

def validate_aiohttp_version(
    aiohttp_version: str, print_warning: Callable[[str], None] = ...
) -> None: ...
