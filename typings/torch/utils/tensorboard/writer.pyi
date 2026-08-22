from typing import Self

from matplotlib.figure import Figure

"""Provide an API for writing protocol buffers to event files to be consumed by TensorBoard for visualization."""
__all__ = ["FileWriter", "SummaryWriter"]

class FileWriter:
    def __init__(
        self, log_dir, max_queue=..., flush_secs=..., filename_suffix=...
    ) -> None: ...
    def get_logdir(self): ...
    def add_event(self, event, step=..., walltime=...) -> None: ...
    def add_summary(self, summary, global_step=..., walltime=...) -> None: ...
    def add_graph(self, graph_profile, walltime=...) -> None: ...
    def add_onnx_graph(self, graph, walltime=...) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def reopen(self) -> None: ...

class SummaryWriter:
    def __init__(
        self,
        log_dir=...,
        comment=...,
        purge_step=...,
        max_queue=...,
        flush_secs=...,
        filename_suffix=...,
    ) -> None: ...
    def get_logdir(self) -> str: ...
    def add_hparams(
        self,
        hparam_dict,
        metric_dict,
        hparam_domain_discrete=...,
        run_name=...,
        global_step=...,
    ) -> None: ...
    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=...,
        walltime=...,
        new_style=...,
        double_precision=...,
    ) -> None: ...
    def add_scalars(
        self, main_tag, tag_scalar_dict, global_step=..., walltime=...
    ) -> None: ...
    def add_tensor(self, tag, tensor, global_step=..., walltime=...) -> None: ...
    def add_histogram(
        self, tag, values, global_step=..., bins=..., walltime=..., max_bins=...
    ) -> None: ...
    def add_histogram_raw(
        self,
        tag,
        min,
        max,
        num,
        sum,
        sum_squares,
        bucket_limits,
        bucket_counts,
        global_step=...,
        walltime=...,
    ) -> None: ...
    def add_image(
        self, tag, img_tensor, global_step=..., walltime=..., dataformats=...
    ) -> None: ...
    def add_images(
        self, tag, img_tensor, global_step=..., walltime=..., dataformats=...
    ) -> None: ...
    def add_image_with_boxes(
        self,
        tag,
        img_tensor,
        box_tensor,
        global_step=...,
        walltime=...,
        rescale=...,
        dataformats=...,
        labels=...,
    ) -> None: ...
    def add_figure(
        self,
        tag: str,
        figure: Figure | list[Figure],
        global_step: int | None = ...,
        close: bool = ...,
        walltime: float | None = ...,
    ) -> None: ...
    def add_video(
        self, tag, vid_tensor, global_step=..., fps=..., walltime=...
    ) -> None: ...
    def add_audio(
        self, tag, snd_tensor, global_step=..., sample_rate=..., walltime=...
    ) -> None: ...
    def add_text(self, tag, text_string, global_step=..., walltime=...) -> None: ...
    def add_onnx_graph(self, prototxt) -> None: ...
    def add_graph(
        self, model, input_to_model=..., verbose=..., use_strict_trace=...
    ) -> None: ...
    def add_embedding(
        self,
        mat,
        metadata=...,
        label_img=...,
        global_step=...,
        tag=...,
        metadata_header=...,
    ) -> None: ...
    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=...,
        num_thresholds=...,
        weights=...,
        walltime=...,
    ) -> None: ...
    def add_pr_curve_raw(
        self,
        tag,
        true_positive_counts,
        false_positive_counts,
        true_negative_counts,
        false_negative_counts,
        precision,
        recall,
        global_step=...,
        num_thresholds=...,
        weights=...,
        walltime=...,
    ) -> None: ...
    def add_custom_scalars_multilinechart(
        self, tags, category=..., title=...
    ) -> None: ...
    def add_custom_scalars_marginchart(self, tags, category=..., title=...) -> None: ...
    def add_custom_scalars(self, layout) -> None: ...
    def add_mesh(
        self,
        tag,
        vertices,
        colors=...,
        faces=...,
        config_dict=...,
        global_step=...,
        walltime=...,
    ) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...
