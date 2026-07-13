from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe

__all__ = ["traverse", "traverse_dps"]
type DataPipe = IterDataPipe | MapDataPipe
type DataPipeGraph = dict[int, tuple[DataPipe, DataPipeGraph]]

def traverse_dps(datapipe: DataPipe) -> DataPipeGraph: ...
def traverse(datapipe: DataPipe, only_datapipe: bool | None = ...) -> DataPipeGraph: ...
