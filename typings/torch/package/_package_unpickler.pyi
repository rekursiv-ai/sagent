import pickle

from .importer import Importer

class PackageUnpickler(pickle._Unpickler):
    def __init__(self, importer: Importer, *args, **kwargs) -> None: ...
    def find_class(self, module, name) -> Any: ...
