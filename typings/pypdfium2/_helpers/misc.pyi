__all__ = ("PdfiumError", "PdfiumWarning")

class PdfiumError(RuntimeError):
    def __init__(self, msg, err_code=...) -> None: ...

class PdfiumWarning(Warning):
    def __init__(self, msg, err_code=...) -> None: ...
