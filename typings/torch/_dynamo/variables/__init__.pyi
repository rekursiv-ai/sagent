from .base import VariableTracker as VariableTracker
from .builtin import BuiltinVariable as BuiltinVariable
from .constant import (
    ConstantVariable as ConstantVariable,
    EnumVariable as EnumVariable,
)
from .ctx_manager import (
    CatchWarningsCtxManagerVariable as CatchWarningsCtxManagerVariable,
    ContextWrappingVariable as ContextWrappingVariable,
    CUDADeviceVariable as CUDADeviceVariable,
    DeterministicAlgorithmsVariable as DeterministicAlgorithmsVariable,
    DisabledSavedTensorsHooksVariable as DisabledSavedTensorsHooksVariable,
    DualLevelContextManager as DualLevelContextManager,
    DynamoConfigPatchVariable as DynamoConfigPatchVariable,
    ErrorOnGraphBreakVariable as ErrorOnGraphBreakVariable,
    FSDPParamGroupUseTrainingStateVariable as FSDPParamGroupUseTrainingStateVariable,
    GradIncrementNestingCtxManagerVariable as GradIncrementNestingCtxManagerVariable,
    GradInplaceRequiresGradCtxManagerVariable as GradInplaceRequiresGradCtxManagerVariable,
    GradModeVariable as GradModeVariable,
    InferenceModeVariable as InferenceModeVariable,
    JvpIncrementNestingCtxManagerVariable as JvpIncrementNestingCtxManagerVariable,
    SDPAKernelVariable as SDPAKernelVariable,
    SetFwdGradEnabledContextManager as SetFwdGradEnabledContextManager,
    StreamContextVariable as StreamContextVariable,
    StreamVariable as StreamVariable,
    TemporarilyPopInterpreterStackCtxManagerVariable as TemporarilyPopInterpreterStackCtxManagerVariable,
    VmapIncrementNestingCtxManagerVariable as VmapIncrementNestingCtxManagerVariable,
    WithExitFunctionVariable as WithExitFunctionVariable,
)
from .dicts import (
    ConstDictVariable as ConstDictVariable,
    DefaultDictVariable as DefaultDictVariable,
    DictKeySetVariable as DictKeySetVariable,
    FrozensetVariable as FrozensetVariable,
    MappingProxyVariable as MappingProxyVariable,
    NNModuleHooksDictVariable as NNModuleHooksDictVariable,
    SetVariable as SetVariable,
)
from .distributed import (
    BackwardHookVariable as BackwardHookVariable,
    DistributedVariable as DistributedVariable,
    PlacementVariable as PlacementVariable,
)
from .functions import (
    BuiltinMethodVariable as BuiltinMethodVariable,
    CollectionsNamedTupleFunction as CollectionsNamedTupleFunction,
    CreateTMADescriptorExperimentalVariable as CreateTMADescriptorExperimentalVariable,
    CreateTMADescriptorStableVariable as CreateTMADescriptorStableVariable,
    FunctionDecoratedByContextlibContextManagerVariable as FunctionDecoratedByContextlibContextManagerVariable,
    FunctoolsPartialVariable as FunctoolsPartialVariable,
    FunctoolsWrapsVariable as FunctoolsWrapsVariable,
    LocalGeneratorFunctionVariable as LocalGeneratorFunctionVariable,
    LocalGeneratorObjectVariable as LocalGeneratorObjectVariable,
    NestedUserFunctionVariable as NestedUserFunctionVariable,
    PolyfilledFunctionVariable as PolyfilledFunctionVariable,
    SkipFunctionVariable as SkipFunctionVariable,
    TMADescriptorExperimentalVariable as TMADescriptorExperimentalVariable,
    TMADescriptorStableVariable as TMADescriptorStableVariable,
    UserFunctionVariable as UserFunctionVariable,
    UserMethodVariable as UserMethodVariable,
    WrapperUserFunctionVariable as WrapperUserFunctionVariable,
    WrapperUserMethodVariable as WrapperUserMethodVariable,
)
from .higher_order_ops import (
    FunctionalCallVariable as FunctionalCallVariable,
    FunctorchHigherOrderVariable as FunctorchHigherOrderVariable,
    ReparametrizeModuleCallVariable as ReparametrizeModuleCallVariable,
    TorchHigherOrderOperatorVariable as TorchHigherOrderOperatorVariable,
)
from .iter import (
    CountIteratorVariable as CountIteratorVariable,
    FilterVariable as FilterVariable,
    IteratorVariable as IteratorVariable,
    ItertoolsVariable as ItertoolsVariable,
    MapVariable as MapVariable,
    ObjectIteratorVariable as ObjectIteratorVariable,
    RepeatIteratorVariable as RepeatIteratorVariable,
    ZipVariable as ZipVariable,
)
from .lazy import LazyVariableTracker as LazyVariableTracker
from .lists import (
    BaseListVariable as BaseListVariable,
    ListIteratorVariable as ListIteratorVariable,
    ListVariable as ListVariable,
    NamedTupleVariable as NamedTupleVariable,
    RangeVariable as RangeVariable,
    SliceVariable as SliceVariable,
    TupleIteratorVariable as TupleIteratorVariable,
    TupleVariable as TupleVariable,
)
from .misc import (
    AutogradFunctionContextVariable as AutogradFunctionContextVariable,
    AutogradFunctionVariable as AutogradFunctionVariable,
    CellVariable as CellVariable,
    DeletedVariable as DeletedVariable,
    ExceptionVariable as ExceptionVariable,
    GetAttrVariable as GetAttrVariable,
    LambdaVariable as LambdaVariable,
    MethodWrapperVariable as MethodWrapperVariable,
    NewGlobalVariable as NewGlobalVariable,
    NumpyVariable as NumpyVariable,
    PythonModuleVariable as PythonModuleVariable,
    RandomClassVariable as RandomClassVariable,
    RandomVariable as RandomVariable,
    RegexPatternVariable as RegexPatternVariable,
    StringFormatVariable as StringFormatVariable,
    SuperVariable as SuperVariable,
    TorchVersionVariable as TorchVersionVariable,
    TypingVariable as TypingVariable,
    UnknownVariable as UnknownVariable,
    WeakRefVariable as WeakRefVariable,
)
from .nn_module import (
    FSDPManagedNNModuleVariable as FSDPManagedNNModuleVariable,
    NNModuleVariable as NNModuleVariable,
    UnspecializedBuiltinNNModuleVariable as UnspecializedBuiltinNNModuleVariable,
    UnspecializedNNModuleVariable as UnspecializedNNModuleVariable,
)
from .optimizer import OptimizerVariable as OptimizerVariable
from .sdpa import SDPAParamsVariable as SDPAParamsVariable
from .tensor import (
    DataPtrVariable as DataPtrVariable,
    FakeItemVariable as FakeItemVariable,
    NumpyNdarrayVariable as NumpyNdarrayVariable,
    SymNodeVariable as SymNodeVariable,
    TensorVariable as TensorVariable,
    UnspecializedPythonVariable as UnspecializedPythonVariable,
    UntypedStorageVariable as UntypedStorageVariable,
)
from .torch import (
    TorchCtxManagerClassVariable as TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable as TorchInGraphFunctionVariable,
)
from .user_defined import (
    FrozenDataClassVariable as FrozenDataClassVariable,
    MutableMappingVariable as MutableMappingVariable,
    RemovableHandleVariable as RemovableHandleVariable,
    UserDefinedClassVariable as UserDefinedClassVariable,
    UserDefinedDictVariable as UserDefinedDictVariable,
    UserDefinedExceptionClassVariable as UserDefinedExceptionClassVariable,
    UserDefinedExceptionObjectVariable as UserDefinedExceptionObjectVariable,
    UserDefinedListVariable as UserDefinedListVariable,
    UserDefinedObjectVariable as UserDefinedObjectVariable,
    UserDefinedSetVariable as UserDefinedSetVariable,
    UserDefinedTupleVariable as UserDefinedTupleVariable,
)

"""
This package implements variable tracking and symbolic execution capabilities for Dynamo,
which are essential for converting Python code into FX graphs. It provides a comprehensive
set of variable types that handle different Python constructs during tracing.

Each variable type (like BuiltinVariable, TensorVariable, NNModuleVariable, etc.) is responsible
for tracking and symbolically executing operations on specific Python objects. This enables
Dynamo to:
- Track the flow of values through Python code
- Maintain correct semantics during graph conversion
- Handle complex Python features like context managers, iterators, and custom objects
- Support both eager and symbolic execution modes

The VariableTracker base class provides the foundation for all variable types, with each
subclass implementing specific behavior for different Python constructs. This modular design
allows Dynamo to accurately trace and optimize Python code while preserving its semantics.
"""
__all__ = [
    "AutogradFunctionContextVariable",
    "AutogradFunctionVariable",
    "BackwardHookVariable",
    "BaseListVariable",
    "BuiltinVariable",
    "CUDADeviceVariable",
    "CatchWarningsCtxManagerVariable",
    "CellVariable",
    "ConstDictVariable",
    "ConstantVariable",
    "ContextWrappingVariable",
    "CountIteratorVariable",
    "CreateTMADescriptorExperimentalVariable",
    "CreateTMADescriptorStableVariable",
    "DataPtrVariable",
    "DefaultDictVariable",
    "DeletedVariable",
    "DeterministicAlgorithmsVariable",
    "DictKeySetVariable",
    "DynamoConfigPatchVariable",
    "EnumVariable",
    "ErrorOnGraphBreakVariable",
    "FakeItemVariable",
    "GetAttrVariable",
    "GradModeVariable",
    "IteratorVariable",
    "ItertoolsVariable",
    "LambdaVariable",
    "LazyVariableTracker",
    "ListIteratorVariable",
    "ListVariable",
    "MappingProxyVariable",
    "NNModuleVariable",
    "NamedTupleVariable",
    "NestedUserFunctionVariable",
    "NewGlobalVariable",
    "NumpyNdarrayVariable",
    "NumpyVariable",
    "OptimizerVariable",
    "PlacementVariable",
    "PolyfilledFunctionVariable",
    "PythonModuleVariable",
    "RangeVariable",
    "RegexPatternVariable",
    "RemovableHandleVariable",
    "RepeatIteratorVariable",
    "SDPAParamsVariable",
    "SkipFunctionVariable",
    "SliceVariable",
    "StringFormatVariable",
    "SuperVariable",
    "TMADescriptorExperimentalVariable",
    "TMADescriptorStableVariable",
    "TemporarilyPopInterpreterStackCtxManagerVariable",
    "TensorVariable",
    "TorchCtxManagerClassVariable",
    "TorchInGraphFunctionVariable",
    "TorchVersionVariable",
    "TupleVariable",
    "UnknownVariable",
    "UnspecializedNNModuleVariable",
    "UnspecializedPythonVariable",
    "UntypedStorageVariable",
    "UserDefinedClassVariable",
    "UserDefinedObjectVariable",
    "UserDefinedTupleVariable",
    "UserFunctionVariable",
    "UserMethodVariable",
    "VariableTracker",
    "WithExitFunctionVariable",
]
