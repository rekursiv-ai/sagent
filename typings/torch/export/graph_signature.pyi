from collections.abc import Callable
from collections.abc import Collection, Mapping
from enum import Enum

import dataclasses

from torch._library.fake_class_registry import FakeScriptObject

__all__ = [
    "ConstantArgument",
    "CustomObjArgument",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "InputKind",
    "InputSpec",
    "OutputKind",
    "OutputSpec",
    "SymBoolArgument",
    "SymFloatArgument",
    "SymIntArgument",
    "TensorArgument",
]

@dataclasses.dataclass
class TensorArgument:
    name: str

@dataclasses.dataclass
class TokenArgument:
    name: str

@dataclasses.dataclass
class SymIntArgument:
    name: str

@dataclasses.dataclass
class SymFloatArgument:
    name: str

@dataclasses.dataclass
class SymBoolArgument:
    name: str

@dataclasses.dataclass
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: FakeScriptObject | None = ...

@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: int | float | bool | str | None

type ArgumentSpec = (
    TensorArgument
    | SymIntArgument
    | SymFloatArgument
    | SymBoolArgument
    | ConstantArgument
    | CustomObjArgument
    | TokenArgument
)

class InputKind(Enum):
    USER_INPUT = ...
    PARAMETER = ...
    BUFFER = ...
    CONSTANT_TENSOR = ...
    CUSTOM_OBJ = ...
    TOKEN = ...

@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: str | None
    persistent: bool | None = ...
    def __post_init__(self) -> None: ...

class OutputKind(Enum):
    USER_OUTPUT = ...
    LOSS_OUTPUT = ...
    BUFFER_MUTATION = ...
    PARAMETER_MUTATION = ...
    GRADIENT_TO_PARAMETER = ...
    GRADIENT_TO_USER_INPUT = ...
    USER_INPUT_MUTATION = ...
    TOKEN = ...

@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: str | None
    def __post_init__(self) -> None: ...

@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str

@dataclasses.dataclass
class ExportGraphSignature:
    input_specs: list[InputSpec]
    output_specs: list[OutputSpec]
    @property
    def parameters(self) -> Collection[str]: ...
    @property
    def buffers(self) -> Collection[str]: ...
    @property
    def non_persistent_buffers(self) -> Collection[str]: ...
    @property
    def lifted_tensor_constants(self) -> Collection[str]: ...
    @property
    def lifted_custom_objs(self) -> Collection[str]: ...
    @property
    def user_inputs(self) -> Collection[int | float | bool | None | str]: ...
    @property
    def user_outputs(self) -> Collection[int | float | bool | None | str]: ...
    @property
    def inputs_to_parameters(self) -> Mapping[str, str]: ...
    @property
    def inputs_to_buffers(self) -> Mapping[str, str]: ...
    @property
    def buffers_to_mutate(self) -> Mapping[str, str]: ...
    @property
    def parameters_to_mutate(self) -> Mapping[str, str]: ...
    @property
    def user_inputs_to_mutate(self) -> Mapping[str, str]: ...
    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]: ...
    @property
    def inputs_to_lifted_custom_objs(self) -> Mapping[str, str]: ...
    @property
    def backward_signature(self) -> ExportBackwardSignature | None: ...
    @property
    def assertion_dep_token(self) -> Mapping[int, str] | None: ...
    @property
    def input_tokens(self) -> Collection[str]: ...
    @property
    def output_tokens(self) -> Collection[str]: ...
    def __post_init__(self) -> None: ...
    def replace_all_uses(self, old: str, new: str) -> None: ...
    def get_replace_hook(self, replace_inputs=...) -> Callable[..., None]: ...
