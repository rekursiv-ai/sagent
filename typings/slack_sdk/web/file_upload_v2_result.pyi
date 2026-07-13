class FileUploadV2Result:
    status: int
    body: str
    def __init__(self, status: int, body: str) -> None: ...
