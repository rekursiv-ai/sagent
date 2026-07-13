from datetime import datetime

from slack_sdk.models.basic_objects import BaseObject

class Link(BaseObject):
    def __init__(self, *, url: str, text: str) -> None: ...

class DateLink(Link):
    def __init__(
        self,
        *,
        date: datetime | int,
        date_format: str,
        fallback: str,
        link: str | None = ...,
    ) -> None: ...

class ObjectLink(Link):
    prefix_mapping = ...
    def __init__(self, *, object_id: str, text: str = ...) -> None: ...

class ChannelLink(Link):
    def __init__(self) -> None: ...

class HereLink(Link):
    def __init__(self) -> None: ...

class EveryoneLink(Link):
    def __init__(self) -> None: ...
