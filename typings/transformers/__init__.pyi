from .cache_utils import (
    Cache as Cache,
    ChunkedSlidingLayer as ChunkedSlidingLayer,
    DynamicCache as DynamicCache,
    DynamicLayer as DynamicLayer,
    EncoderDecoderCache as EncoderDecoderCache,
    HQQQuantizedCache as HQQQuantizedCache,
    HQQQuantizedLayer as HQQQuantizedLayer,
    HybridCache as HybridCache,
    OffloadedCache as OffloadedCache,
    OffloadedStaticCache as OffloadedStaticCache,
    QuantizedCache as QuantizedCache,
    QuantoQuantizedCache as QuantoQuantizedCache,
    QuantoQuantizedLayer as QuantoQuantizedLayer,
    SinkCache as SinkCache,
    SlidingWindowCache as SlidingWindowCache,
    SlidingWindowLayer as SlidingWindowLayer,
    StaticCache as StaticCache,
    StaticLayer as StaticLayer,
    StaticSlidingWindowLayer as StaticSlidingWindowLayer,
)
from .configuration_utils import PretrainedConfig as PretrainedConfig
from .convert_slow_tokenizer import (
    SLOW_TO_FAST_CONVERTERS as SLOW_TO_FAST_CONVERTERS,
    convert_slow_tokenizer as convert_slow_tokenizer,
)
from .data import (
    DataProcessor as DataProcessor,
    InputExample as InputExample,
    InputFeatures as InputFeatures,
    SingleSentenceClassificationProcessor as SingleSentenceClassificationProcessor,
    SquadExample as SquadExample,
    SquadFeatures as SquadFeatures,
    SquadV1Processor as SquadV1Processor,
    SquadV2Processor as SquadV2Processor,
    glue_compute_metrics as glue_compute_metrics,
    glue_convert_examples_to_features as glue_convert_examples_to_features,
    glue_output_modes as glue_output_modes,
    glue_processors as glue_processors,
    glue_tasks_num_labels as glue_tasks_num_labels,
    squad_convert_examples_to_features as squad_convert_examples_to_features,
    xnli_compute_metrics as xnli_compute_metrics,
    xnli_output_modes as xnli_output_modes,
    xnli_processors as xnli_processors,
    xnli_tasks_num_labels as xnli_tasks_num_labels,
)
from .data.data_collator import (
    DataCollator as DataCollator,
    DataCollatorForLanguageModeling as DataCollatorForLanguageModeling,
    DataCollatorForMultipleChoice as DataCollatorForMultipleChoice,
    DataCollatorForPermutationLanguageModeling as DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq as DataCollatorForSeq2Seq,
    DataCollatorForSOP as DataCollatorForSOP,
    DataCollatorForTokenClassification as DataCollatorForTokenClassification,
    DataCollatorForWholeWordMask as DataCollatorForWholeWordMask,
    DataCollatorWithFlattening as DataCollatorWithFlattening,
    DataCollatorWithPadding as DataCollatorWithPadding,
    DefaultDataCollator as DefaultDataCollator,
    default_data_collator as default_data_collator,
)
from .data.datasets import (
    GlueDataset as GlueDataset,
    GlueDataTrainingArguments as GlueDataTrainingArguments,
    LineByLineTextDataset as LineByLineTextDataset,
    LineByLineWithRefDataset as LineByLineWithRefDataset,
    LineByLineWithSOPTextDataset as LineByLineWithSOPTextDataset,
    SquadDataset as SquadDataset,
    SquadDataTrainingArguments as SquadDataTrainingArguments,
    TextDataset as TextDataset,
    TextDatasetForNextSentencePrediction as TextDatasetForNextSentencePrediction,
)
from .feature_extraction_sequence_utils import (
    SequenceFeatureExtractor as SequenceFeatureExtractor,
)
from .feature_extraction_utils import (
    BatchFeature as BatchFeature,
    FeatureExtractionMixin as FeatureExtractionMixin,
)
from .generation import (
    AlternatingCodebooksLogitsProcessor as AlternatingCodebooksLogitsProcessor,
    AsyncTextIteratorStreamer as AsyncTextIteratorStreamer,
    BayesianDetectorConfig as BayesianDetectorConfig,
    BayesianDetectorModel as BayesianDetectorModel,
    BeamScorer as BeamScorer,
    ClassifierFreeGuidanceLogitsProcessor as ClassifierFreeGuidanceLogitsProcessor,
    CompileConfig as CompileConfig,
    ConstrainedBeamSearchScorer as ConstrainedBeamSearchScorer,
    Constraint as Constraint,
    ConstraintListState as ConstraintListState,
    DisjunctiveConstraint as DisjunctiveConstraint,
    EncoderNoRepeatNGramLogitsProcessor as EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor as EncoderRepetitionPenaltyLogitsProcessor,
    EosTokenCriteria as EosTokenCriteria,
    EpsilonLogitsWarper as EpsilonLogitsWarper,
    EtaLogitsWarper as EtaLogitsWarper,
    ExponentialDecayLengthPenalty as ExponentialDecayLengthPenalty,
    FlaxForcedBOSTokenLogitsProcessor as FlaxForcedBOSTokenLogitsProcessor,
    FlaxForcedEOSTokenLogitsProcessor as FlaxForcedEOSTokenLogitsProcessor,
    FlaxForceTokensLogitsProcessor as FlaxForceTokensLogitsProcessor,
    FlaxGenerationMixin as FlaxGenerationMixin,
    FlaxLogitsProcessor as FlaxLogitsProcessor,
    FlaxLogitsProcessorList as FlaxLogitsProcessorList,
    FlaxLogitsWarper as FlaxLogitsWarper,
    FlaxMinLengthLogitsProcessor as FlaxMinLengthLogitsProcessor,
    FlaxSuppressTokensAtBeginLogitsProcessor as FlaxSuppressTokensAtBeginLogitsProcessor,
    FlaxSuppressTokensLogitsProcessor as FlaxSuppressTokensLogitsProcessor,
    FlaxTemperatureLogitsWarper as FlaxTemperatureLogitsWarper,
    FlaxTopKLogitsWarper as FlaxTopKLogitsWarper,
    FlaxTopPLogitsWarper as FlaxTopPLogitsWarper,
    FlaxWhisperTimeStampLogitsProcessor as FlaxWhisperTimeStampLogitsProcessor,
    ForcedBOSTokenLogitsProcessor as ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor as ForcedEOSTokenLogitsProcessor,
    GenerationConfig as GenerationConfig,
    GenerationMixin as GenerationMixin,
    InfNanRemoveLogitsProcessor as InfNanRemoveLogitsProcessor,
    LogitNormalization as LogitNormalization,
    LogitsProcessor as LogitsProcessor,
    LogitsProcessorList as LogitsProcessorList,
    MaxLengthCriteria as MaxLengthCriteria,
    MaxTimeCriteria as MaxTimeCriteria,
    MinLengthLogitsProcessor as MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor as MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper as MinPLogitsWarper,
    NoBadWordsLogitsProcessor as NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor as NoRepeatNGramLogitsProcessor,
    PhrasalConstraint as PhrasalConstraint,
    PrefixConstrainedLogitsProcessor as PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor as RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor as SequenceBiasLogitsProcessor,
    StoppingCriteria as StoppingCriteria,
    StoppingCriteriaList as StoppingCriteriaList,
    StopStringCriteria as StopStringCriteria,
    SuppressTokensAtBeginLogitsProcessor as SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor as SuppressTokensLogitsProcessor,
    SynthIDTextWatermarkDetector as SynthIDTextWatermarkDetector,
    SynthIDTextWatermarkingConfig as SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor as SynthIDTextWatermarkLogitsProcessor,
    TemperatureLogitsWarper as TemperatureLogitsWarper,
    TextIteratorStreamer as TextIteratorStreamer,
    TextStreamer as TextStreamer,
    TFForcedBOSTokenLogitsProcessor as TFForcedBOSTokenLogitsProcessor,
    TFForcedEOSTokenLogitsProcessor as TFForcedEOSTokenLogitsProcessor,
    TFForceTokensLogitsProcessor as TFForceTokensLogitsProcessor,
    TFGenerationMixin as TFGenerationMixin,
    TFLogitsProcessor as TFLogitsProcessor,
    TFLogitsProcessorList as TFLogitsProcessorList,
    TFLogitsWarper as TFLogitsWarper,
    TFMinLengthLogitsProcessor as TFMinLengthLogitsProcessor,
    TFNoBadWordsLogitsProcessor as TFNoBadWordsLogitsProcessor,
    TFNoRepeatNGramLogitsProcessor as TFNoRepeatNGramLogitsProcessor,
    TFRepetitionPenaltyLogitsProcessor as TFRepetitionPenaltyLogitsProcessor,
    TFSuppressTokensAtBeginLogitsProcessor as TFSuppressTokensAtBeginLogitsProcessor,
    TFSuppressTokensLogitsProcessor as TFSuppressTokensLogitsProcessor,
    TFTemperatureLogitsWarper as TFTemperatureLogitsWarper,
    TFTopKLogitsWarper as TFTopKLogitsWarper,
    TFTopPLogitsWarper as TFTopPLogitsWarper,
    TopKLogitsWarper as TopKLogitsWarper,
    TopPLogitsWarper as TopPLogitsWarper,
    TypicalLogitsWarper as TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor as UnbatchedClassifierFreeGuidanceLogitsProcessor,
    WatermarkDetector as WatermarkDetector,
    WatermarkingConfig as WatermarkingConfig,
    WatermarkLogitsProcessor as WatermarkLogitsProcessor,
    WhisperTimeStampLogitsProcessor as WhisperTimeStampLogitsProcessor,
)
from .hf_argparser import HfArgumentParser as HfArgumentParser
from .image_processing_base import ImageProcessingMixin as ImageProcessingMixin
from .image_processing_utils import BaseImageProcessor as BaseImageProcessor
from .image_processing_utils_fast import (
    BaseImageProcessorFast as BaseImageProcessorFast,
)
from .image_utils import ImageFeatureExtractionMixin as ImageFeatureExtractionMixin
from .integrations import (
    is_clearml_available as is_clearml_available,
    is_comet_available as is_comet_available,
    is_dvclive_available as is_dvclive_available,
    is_neptune_available as is_neptune_available,
    is_optuna_available as is_optuna_available,
    is_ray_available as is_ray_available,
    is_ray_tune_available as is_ray_tune_available,
    is_sigopt_available as is_sigopt_available,
    is_swanlab_available as is_swanlab_available,
    is_tensorboard_available as is_tensorboard_available,
    is_trackio_available as is_trackio_available,
    is_wandb_available as is_wandb_available,
)
from .integrations.executorch import (
    TorchExportableModuleWithStaticCache as TorchExportableModuleWithStaticCache,
    convert_and_export_with_cache as convert_and_export_with_cache,
)
from .keras_callbacks import (
    KerasMetricCallback as KerasMetricCallback,
    PushToHubCallback as PushToHubCallback,
)
from .masking_utils import AttentionMaskInterface as AttentionMaskInterface
from .model_debugging_utils import (
    model_addition_debugger_context as model_addition_debugger_context,
)
from .modelcard import ModelCard as ModelCard
from .modeling_flax_utils import FlaxPreTrainedModel as FlaxPreTrainedModel
from .modeling_layers import GradientCheckpointingLayer as GradientCheckpointingLayer
from .modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS as ROPE_INIT_FUNCTIONS,
    dynamic_rope_update as dynamic_rope_update,
)
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name as convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model as load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model as load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model as load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model as load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model as load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model as load_tf2_weights_in_pytorch_model,
)
from .modeling_tf_utils import (
    TFPreTrainedModel as TFPreTrainedModel,
    TFSequenceSummary as TFSequenceSummary,
    TFSharedEmbeddings as TFSharedEmbeddings,
    shape_list as shape_list,
)
from .modeling_utils import (
    AttentionInterface as AttentionInterface,
    PreTrainedModel as PreTrainedModel,
)
from .models import *
from .models.mamba.modeling_mamba import MambaCache as MambaCache
from .models.timm_wrapper import TimmWrapperImageProcessor as TimmWrapperImageProcessor
from .optimization import (
    Adafactor as Adafactor,
    get_constant_schedule as get_constant_schedule,
    get_constant_schedule_with_warmup as get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup as get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup as get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup as get_cosine_with_min_lr_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup_lr_rate as get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
    get_inverse_sqrt_schedule as get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup as get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup as get_polynomial_decay_schedule_with_warmup,
    get_scheduler as get_scheduler,
    get_wsd_schedule as get_wsd_schedule,
)
from .optimization_tf import (
    AdamWeightDecay as AdamWeightDecay,
    GradientAccumulator as GradientAccumulator,
    WarmUp as WarmUp,
    create_optimizer as create_optimizer,
)
from .pipelines import (
    AudioClassificationPipeline as AudioClassificationPipeline,
    AutomaticSpeechRecognitionPipeline as AutomaticSpeechRecognitionPipeline,
    CsvPipelineDataFormat as CsvPipelineDataFormat,
    DepthEstimationPipeline as DepthEstimationPipeline,
    DocumentQuestionAnsweringPipeline as DocumentQuestionAnsweringPipeline,
    FeatureExtractionPipeline as FeatureExtractionPipeline,
    FillMaskPipeline as FillMaskPipeline,
    ImageClassificationPipeline as ImageClassificationPipeline,
    ImageFeatureExtractionPipeline as ImageFeatureExtractionPipeline,
    ImageSegmentationPipeline as ImageSegmentationPipeline,
    ImageTextToTextPipeline as ImageTextToTextPipeline,
    ImageToImagePipeline as ImageToImagePipeline,
    ImageToTextPipeline as ImageToTextPipeline,
    JsonPipelineDataFormat as JsonPipelineDataFormat,
    KeypointMatchingPipeline as KeypointMatchingPipeline,
    MaskGenerationPipeline as MaskGenerationPipeline,
    NerPipeline as NerPipeline,
    ObjectDetectionPipeline as ObjectDetectionPipeline,
    PipedPipelineDataFormat as PipedPipelineDataFormat,
    Pipeline as Pipeline,
    PipelineDataFormat as PipelineDataFormat,
    QuestionAnsweringPipeline as QuestionAnsweringPipeline,
    SummarizationPipeline as SummarizationPipeline,
    TableQuestionAnsweringPipeline as TableQuestionAnsweringPipeline,
    Text2TextGenerationPipeline as Text2TextGenerationPipeline,
    TextClassificationPipeline as TextClassificationPipeline,
    TextGenerationPipeline as TextGenerationPipeline,
    TextToAudioPipeline as TextToAudioPipeline,
    TokenClassificationPipeline as TokenClassificationPipeline,
    TranslationPipeline as TranslationPipeline,
    VideoClassificationPipeline as VideoClassificationPipeline,
    VisualQuestionAnsweringPipeline as VisualQuestionAnsweringPipeline,
    ZeroShotAudioClassificationPipeline as ZeroShotAudioClassificationPipeline,
    ZeroShotClassificationPipeline as ZeroShotClassificationPipeline,
    ZeroShotImageClassificationPipeline as ZeroShotImageClassificationPipeline,
    ZeroShotObjectDetectionPipeline as ZeroShotObjectDetectionPipeline,
    pipeline as pipeline,
)
from .processing_utils import ProcessorMixin as ProcessorMixin
from .pytorch_utils import (
    Conv1D as Conv1D,
    apply_chunking_to_forward as apply_chunking_to_forward,
    prune_layer as prune_layer,
)
from .tokenization_utils import PreTrainedTokenizer as PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken as AddedToken,
    BatchEncoding as BatchEncoding,
    CharSpan as CharSpan,
    PreTrainedTokenizerBase as PreTrainedTokenizerBase,
    SpecialTokensMixin as SpecialTokensMixin,
    TokenSpan as TokenSpan,
)
from .tokenization_utils_fast import PreTrainedTokenizerFast as PreTrainedTokenizerFast
from .trainer import Trainer as Trainer
from .trainer_callback import (
    DefaultFlowCallback as DefaultFlowCallback,
    EarlyStoppingCallback as EarlyStoppingCallback,
    PrinterCallback as PrinterCallback,
    ProgressCallback as ProgressCallback,
    TrainerCallback as TrainerCallback,
    TrainerControl as TrainerControl,
    TrainerState as TrainerState,
)
from .trainer_pt_utils import (
    torch_distributed_zero_first as torch_distributed_zero_first,
)
from .trainer_seq2seq import Seq2SeqTrainer as Seq2SeqTrainer
from .trainer_utils import (
    EvalPrediction as EvalPrediction,
    IntervalStrategy as IntervalStrategy,
    SchedulerType as SchedulerType,
    enable_full_determinism as enable_full_determinism,
    set_seed as set_seed,
)
from .training_args import TrainingArguments as TrainingArguments
from .training_args_seq2seq import Seq2SeqTrainingArguments as Seq2SeqTrainingArguments
from .training_args_tf import TFTrainingArguments as TFTrainingArguments
from .utils import (
    CONFIG_NAME as CONFIG_NAME,
    MODEL_CARD_NAME as MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE as PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE as PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE as SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME as TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME as TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE as TRANSFORMERS_CACHE,
    WEIGHTS_NAME as WEIGHTS_NAME,
    TensorType as TensorType,
    add_end_docstrings as add_end_docstrings,
    add_start_docstrings as add_start_docstrings,
    is_apex_available as is_apex_available,
    is_av_available as is_av_available,
    is_bitsandbytes_available as is_bitsandbytes_available,
    is_datasets_available as is_datasets_available,
    is_faiss_available as is_faiss_available,
    is_flax_available as is_flax_available,
    is_keras_nlp_available as is_keras_nlp_available,
    is_matplotlib_available as is_matplotlib_available,
    is_phonemizer_available as is_phonemizer_available,
    is_psutil_available as is_psutil_available,
    is_py3nvml_available as is_py3nvml_available,
    is_pyctcdecode_available as is_pyctcdecode_available,
    is_sacremoses_available as is_sacremoses_available,
    is_safetensors_available as is_safetensors_available,
    is_scipy_available as is_scipy_available,
    is_sentencepiece_available as is_sentencepiece_available,
    is_sklearn_available as is_sklearn_available,
    is_speech_available as is_speech_available,
    is_tensorflow_text_available as is_tensorflow_text_available,
    is_tf_available as is_tf_available,
    is_timm_available as is_timm_available,
    is_tokenizers_available as is_tokenizers_available,
    is_torch_available as is_torch_available,
    is_torch_hpu_available as is_torch_hpu_available,
    is_torch_mlu_available as is_torch_mlu_available,
    is_torch_musa_available as is_torch_musa_available,
    is_torch_neuroncore_available as is_torch_neuroncore_available,
    is_torch_npu_available as is_torch_npu_available,
    is_torch_xla_available as is_torch_xla_available,
    is_torch_xpu_available as is_torch_xpu_available,
    is_torchaudio_available as is_torchaudio_available,
    is_torchvision_available as is_torchvision_available,
    is_vision_available as is_vision_available,
    logging as logging,
)
from .utils.quantization_config import (
    AqlmConfig as AqlmConfig,
    AutoRoundConfig as AutoRoundConfig,
    AwqConfig as AwqConfig,
    BitNetQuantConfig as BitNetQuantConfig,
    BitsAndBytesConfig as BitsAndBytesConfig,
    CompressedTensorsConfig as CompressedTensorsConfig,
    EetqConfig as EetqConfig,
    FbgemmFp8Config as FbgemmFp8Config,
    FineGrainedFP8Config as FineGrainedFP8Config,
    FPQuantConfig as FPQuantConfig,
    GPTQConfig as GPTQConfig,
    HiggsConfig as HiggsConfig,
    HqqConfig as HqqConfig,
    QuantoConfig as QuantoConfig,
    QuarkConfig as QuarkConfig,
    SpQRConfig as SpQRConfig,
    TorchAoConfig as TorchAoConfig,
    VptqConfig as VptqConfig,
)
from .video_processing_utils import BaseVideoProcessor as BaseVideoProcessor

__version__ = ...
logger = ...
_import_structure = ...
