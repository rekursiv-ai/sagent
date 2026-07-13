from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin

"""
Feature extractor class for MarkupLM.
"""
logger = ...

class MarkupLMFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, **kwargs) -> None: ...
    def xpath_soup(self, element):  # -> tuple[list[Any], list[Any]]:
        ...
    def get_three_from_single(
        self, html_string
    ):  # -> tuple[list[Any], list[Any], list[Any]]:
        ...
    def construct_xpath(self, xpath_tags, xpath_subscripts):  # -> str:
        ...
    def __call__(self, html_strings) -> BatchFeature: ...

__all__ = ["MarkupLMFeatureExtractor"]
