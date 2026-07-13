from ...processing_utils import ProcessingKwargs, ProcessorMixin

class KyutaiSpeechToTextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = ...

class KyutaiSpeechToTextProcessor(ProcessorMixin):
    feature_extractor_class = ...
    tokenizer_class = ...
    valid_processor_kwargs = KyutaiSpeechToTextProcessorKwargs

__all__ = ["KyutaiSpeechToTextProcessor"]
