import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger:
    """
    Enhanced TensorBoard logger with extensive logging capabilities.
    """

    def __init__(
        self, log_dir: str = "runs", experiment_name: str = "", config: dict = None
    ):
        # Configuration allows fine control over logging details.
        config = config or {}
        log_path = (
            os.path.join(log_dir, experiment_name) if experiment_name else log_dir
        )
        self.writer = SummaryWriter(log_dir=log_path)
        # Use config parameters to control verbosity and details.
        self.verbose = config.get("verbose", True)
        self.log_every_batch = config.get("log_every_batch", False)
        self.log_batch_sample_freq = config.get("log_batch_sample_freq", 50)
        self.log_memory = config.get("log_memory", True)
        self.log_exec_time = config.get("log_exec_time", True)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_text(self, tag: str, text: str, step: int):
        self.writer.add_text(tag, text, step)

    def log_figure(self, tag: str, figure, step: int):
        self.writer.add_figure(tag, figure, step)

    def log_histogram(self, tag: str, values, step: int, bins: int = 100):
        self.writer.add_histogram(tag, values, step, bins=bins)

    def log_image(self, tag: str, img_tensor, step: int):
        self.writer.add_image(tag, img_tensor, step)

    def log_tensor_info(self, tag: str, tensor, step: int):
        info = f"Shape: {tuple(tensor.shape)}, Dtype: {tensor.dtype}, Device: {tensor.device}"
        self.writer.add_text(tag, info, step)

    def log_model_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def log_memory_usage(self, step: int):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            mem_info = f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
        else:
            mem_info = "CUDA not available."
        self.writer.add_text("Memory/Usage", mem_info, step)

    def log_execution_time(self, tag: str, exec_time: float, step: int):
        self.writer.add_scalar(f"Time/{tag}", exec_time, step)

    def close(self):
        self.writer.close()
