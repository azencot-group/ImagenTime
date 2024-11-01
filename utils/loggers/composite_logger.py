from typing import Any, Dict, List

from .base_logger import BaseLogger


class CompositeLogger(BaseLogger):

    def __init__(self, loggers, *args, **kwargs):
        super(CompositeLogger, self).__init__(*args, **kwargs)
        self.loggers = loggers

    def __enter__(self):
        return self

    def stop(self):
        for logger in self.loggers:
            logger.stop()

    def __exit__(self, type, value, traceback):
        self.stop()

    def log(self, name: str, data: Any, step=None):
        for logger in self.loggers:
            logger.log(name, data, step)

    def _log_fig(self, name: str, fig: Any):
        for logger in self.loggers:
            logger.log_fig(name, fig)

    def log_hparams(self, params: Dict[str, Any]):
        for logger in self.loggers:
            logger.log_hparams(params)

    def log_params(self, params: Dict[str, Any]):
        for logger in self.loggers:
            logger.log_params(params)

    def add_tags(self, tags: List[str]):
        for logger in self.loggers:
            logger.add_tags(tags)

    def log_name_params(self, name : str, params: Any):
        for logger in self.loggers:
            logger.log_name_params(name, params)