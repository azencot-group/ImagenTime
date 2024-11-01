import os

from .base_logger import BaseLogger
from typing import Dict, Any, List, Optional
from pprint import pprint
import numpy as np


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}


class MlflowLogger(BaseLogger):

    def __init__(self, ip, *args, **kwargs):
        super(MlflowLogger, self).__init__(*args, **kwargs)
        import mlflow
        from mlflow import log_metric, log_param, log_params, log_artifacts, log_figure
        from matplotlib import pyplot as plt
        mlflow.set_tracking_uri(f"http://{ip}")
        self.log_metric = log_metric
        self.log_param = log_param
        self._log_params = log_params
        self.log_artifacts = log_artifacts
        self.log_figure = log_figure
        self.mlflow = mlflow
        self.run = mlflow.start_run()
        self.plt = plt

    def stop(self):
        self.mlflow.end_run()

    def log(self, name: str, data: Any, step=None):
        self.log_metric(name, data, step)

    def _log_fig(self, name: str, fig: Any):
        if isinstance(fig, np.ndarray):
            fig, ax = self.plt.subplots()
            ax.set_axis_off()
            ax.imshow(fig)
        self.log_figure(fig, f'{name}.png')

    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            if isinstance(v, dict):
                self._log_params({f'{k}/{kk}': vv for kk, vv in v.items()})
            else:
                self.log_param(k, v)
        self.mlflow.log_dict(convert_no_basic_to_str(params), 'params.json')

    def add_tags(self, tags: List[str]):
        self.mlflow.set_tags({'tags': tags})
