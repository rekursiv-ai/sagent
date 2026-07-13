class FrameHeader:
    fin: int
    rsv1: int
    rsv2: int
    rsv3: int
    opcode: int
    masked: int
    length: int
    OPCODE_CONTINUATION = ...
    OPCODE_TEXT = ...
    OPCODE_BINARY = ...
    OPCODE_CLOSE = ...
    OPCODE_PING = ...
    OPCODE_PONG = ...
    def __init__(
        self,
        opcode: int,
        fin: int = ...,
        rsv1: int = ...,
        rsv2: int = ...,
        rsv3: int = ...,
        masked: int = ...,
        length: int = ...,
    ) -> None: ...
