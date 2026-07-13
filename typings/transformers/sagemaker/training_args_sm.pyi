from dataclasses import dataclass

from ..training_args import TrainingArguments

logger = ...

def is_sagemaker_model_parallel_available():  # -> bool:
    ...

@dataclass
class SageMakerTrainingArguments(TrainingArguments):
    mp_parameters: str = ...
    def __post_init__(self):  # -> None:
        ...
    @property
    def world_size(self):  # -> Literal[1]:
        ...
    @property
    def place_model_on_device(self):  # -> bool:
        ...
