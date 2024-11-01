from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union


class BaseLogger(ABC):

    def __init__(self, no_plot: bool = False, *args, **kwargs):
        super(BaseLogger, self).__init__()
        self.no_plot = no_plot

    def __enter__(self):
        return self

    @abstractmethod
    def stop(self):
        pass

    def __exit__(self, type, value, traceback):
        self.stop()

    @abstractmethod
    def log(self, name: str, data: Any, step=None):
        pass

    def log_dict(self, name: str, data: Dict[str, Any], step=None):
        for k, v in data.items():
            self.log(f'{name}/{k}', v.item(), step)

    def log_fig(self, name: str, fig: Any):
        if self.no_plot:
            return
        self._log_fig(name, fig)

    @abstractmethod
    def _log_fig(self, name: str, fig: Any):
        pass

    @abstractmethod
    def log_hparams(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def add_tags(self, tags: List[str]):
        pass

    @abstractmethod
    def log_name_params(self, name : str, params: Any):
        pass
