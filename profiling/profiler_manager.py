import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from utils.logger import get_logger
import logging


class ProfilerManager:
    """
    A manager class that wraps PyTorch's profiler.
    Supports configurable scheduling, activities, and integrates a trace callback.
    """

    def __init__(
        self, config: dict, tensorboard_writer=None, logger: logging.Logger = None
    ):
        """
        Args:
            config (dict): Configuration dictionary with keys:
                - enabled (bool): Whether profiling is enabled.
                - schedule_wait (int): Number of steps to wait.
                - schedule_warmup (int): Number of warmup steps.
                - schedule_active (int): Number of active steps.
                - schedule_repeat (int): Number of repeats.
                - activities (list): List containing "cpu" and/or "cuda".
                - record_shapes (bool): Whether to record tensor shapes.
                - profile_memory (bool): Whether to profile memory.
                - with_stack (bool): Whether to capture stack traces.
                - trace_output_path (str): File path pattern for saving chrome trace.
            tensorboard_writer: (Optional) A SummaryWriter instance for logging traces.
            logger (logging.Logger): Logger for logging messages.
        """
        self.enabled = config.get("enabled", False)
        self.config = config
        self.tensorboard_writer = tensorboard_writer
        self.logger = logger or get_logger(name=self.__class__.__name__)
        self.profiler = None

        # Build schedule from config
        self.profiler_schedule = schedule(
            wait=config.get("schedule_wait", 1),
            warmup=config.get("schedule_warmup", 1),
            active=config.get("schedule_active", 1),
            repeat=config.get("schedule_repeat", 1),
        )

        # Determine activities
        activities = []
        if "cpu" in config.get("activities", []):
            activities.append(ProfilerActivity.CPU)
        if "cuda" in config.get("activities", []):
            activities.append(ProfilerActivity.CUDA)
        self.activities = activities

    def _trace_handler(self, prof):
        """
        Callback executed when a trace is ready.
        Exports the trace as a Chrome trace and logs a message.
        """
        trace_output_path = self.config.get(
            "trace_output_path", "./profiler_trace.json"
        )
        # Ensure directory exists
        os.makedirs(os.path.dirname(trace_output_path), exist_ok=True)
        prof.export_chrome_trace(trace_output_path)
        self.logger.info(f"Profiler trace exported to: {trace_output_path}")
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(
                "Profiler", f"Trace exported to: {trace_output_path}", global_step=0
            )

    def __enter__(self):
        if self.enabled:
            self.profiler = profile(
                activities=self.activities,
                schedule=self.profiler_schedule,
                on_trace_ready=self._trace_handler,
                record_shapes=self.config.get("record_shapes", False),
                profile_memory=self.config.get("profile_memory", False),
                with_stack=self.config.get("with_stack", False),
            )
            self.profiler.__enter__()
        return self

    def step(self):
        """
        Call this at each training iteration.
        """
        if self.profiler is not None:
            self.profiler.step()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
