from typing import Any, Literal, overload

import torch

from .audio_classification import AudioClassificationPipeline
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .base import (
    Pipeline,
)
from .depth_estimation import DepthEstimationPipeline
from .document_question_answering import DocumentQuestionAnsweringPipeline
from .feature_extraction import FeatureExtractionPipeline
from .fill_mask import FillMaskPipeline
from .image_classification import ImageClassificationPipeline
from .image_feature_extraction import ImageFeatureExtractionPipeline
from .image_segmentation import ImageSegmentationPipeline
from .image_text_to_text import ImageTextToTextPipeline
from .image_to_image import ImageToImagePipeline
from .image_to_text import ImageToTextPipeline
from .keypoint_matching import KeypointMatchingPipeline
from .mask_generation import MaskGenerationPipeline
from .object_detection import ObjectDetectionPipeline
from .question_answering import (
    QuestionAnsweringPipeline,
)
from .table_question_answering import (
    TableQuestionAnsweringPipeline,
)
from .text2text_generation import (
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TranslationPipeline,
)
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .text_to_audio import TextToAudioPipeline
from .token_classification import (
    TokenClassificationPipeline,
)
from .video_classification import VideoClassificationPipeline
from .visual_question_answering import VisualQuestionAnsweringPipeline
from .zero_shot_audio_classification import ZeroShotAudioClassificationPipeline
from .zero_shot_classification import (
    ZeroShotClassificationPipeline,
)
from .zero_shot_image_classification import ZeroShotImageClassificationPipeline
from .zero_shot_object_detection import ZeroShotObjectDetectionPipeline
from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..processing_utils import ProcessorMixin
from ..tokenization_utils import PreTrainedTokenizer
from ..tokenization_utils_fast import PreTrainedTokenizerFast

logger = ...
TASK_ALIASES = ...
SUPPORTED_TASKS = ...
PIPELINE_REGISTRY = ...

def get_supported_tasks() -> list[str]: ...
def get_task(model: str, token: str | None = ..., **deprecated_kwargs) -> str: ...
def check_task(task: str) -> tuple[str, dict, Any]: ...
def clean_custom_task(task_info):  # -> tuple[Any, None]:
    ...
@overload
def pipeline(
    task: None,
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> Pipeline: ...
@overload
def pipeline(
    task: Literal["audio-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> AudioClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["automatic-speech-recognition"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> AutomaticSpeechRecognitionPipeline: ...
@overload
def pipeline(
    task: Literal["depth-estimation"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> DepthEstimationPipeline: ...
@overload
def pipeline(
    task: Literal["document-question-answering"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> DocumentQuestionAnsweringPipeline: ...
@overload
def pipeline(
    task: Literal["feature-extraction"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> FeatureExtractionPipeline: ...
@overload
def pipeline(
    task: Literal["fill-mask"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> FillMaskPipeline: ...
@overload
def pipeline(
    task: Literal["image-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ImageClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["image-feature-extraction"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ImageFeatureExtractionPipeline: ...
@overload
def pipeline(
    task: Literal["image-segmentation"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ImageSegmentationPipeline: ...
@overload
def pipeline(
    task: Literal["image-text-to-text"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ImageTextToTextPipeline: ...
@overload
def pipeline(
    task: Literal["image-to-image"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ImageToImagePipeline: ...
@overload
def pipeline(
    task: Literal["image-to-text"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ImageToTextPipeline: ...
@overload
def pipeline(
    task: Literal["keypoint-matching"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> KeypointMatchingPipeline: ...
@overload
def pipeline(
    task: Literal["mask-generation"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> MaskGenerationPipeline: ...
@overload
def pipeline(
    task: Literal["object-detection"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ObjectDetectionPipeline: ...
@overload
def pipeline(
    task: Literal["question-answering"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> QuestionAnsweringPipeline: ...
@overload
def pipeline(
    task: Literal["summarization"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> SummarizationPipeline: ...
@overload
def pipeline(
    task: Literal["table-question-answering"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> TableQuestionAnsweringPipeline: ...
@overload
def pipeline(
    task: Literal["text-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> TextClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["text-generation"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> TextGenerationPipeline: ...
@overload
def pipeline(
    task: Literal["text-to-audio"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> TextToAudioPipeline: ...
@overload
def pipeline(
    task: Literal["text2text-generation"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> Text2TextGenerationPipeline: ...
@overload
def pipeline(
    task: Literal["token-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> TokenClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["translation"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> TranslationPipeline: ...
@overload
def pipeline(
    task: Literal["video-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> VideoClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["visual-question-answering"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> VisualQuestionAnsweringPipeline: ...
@overload
def pipeline(
    task: Literal["zero-shot-audio-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ZeroShotAudioClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["zero-shot-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ZeroShotClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["zero-shot-image-classification"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ZeroShotImageClassificationPipeline: ...
@overload
def pipeline(
    task: Literal["zero-shot-object-detection"],
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> ZeroShotObjectDetectionPipeline: ...
def pipeline(
    task: str | None = ...,
    model: str | PreTrainedModel | TFPreTrainedModel | None = ...,
    config: str | PretrainedConfig | None = ...,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None = ...,
    feature_extractor: str | PreTrainedFeatureExtractor | None = ...,
    image_processor: str | BaseImageProcessor | None = ...,
    processor: str | ProcessorMixin | None = ...,
    framework: str | None = ...,
    revision: str | None = ...,
    use_fast: bool = ...,
    token: str | bool | None = ...,
    device: int | str | torch.device | None = ...,
    device_map: str | dict[str, int | str] | None = ...,
    dtype: str | torch.dtype | None = ...,
    trust_remote_code: bool | None = ...,
    model_kwargs: dict[str, Any] | None = ...,
    pipeline_class: Any | None = ...,
    **kwargs: Any,
) -> Pipeline: ...
