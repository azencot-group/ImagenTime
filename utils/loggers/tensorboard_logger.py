from .base_logger import BaseLogger
from typing import Dict, Any, List


class TensorboardLogger(BaseLogger):

    def __init__(self, tb_dir, *args, **kwargs):
        super(TensorboardLogger, self).__init__(*args, **kwargs)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(tb_dir)

    def stop(self):
        self.writer.close()

    def log(self, name: str, data: Any, step=None):
        self.writer.add_scalar(name, data, step)

    def _log_fig(self, name: str, fig: Any):
        self.writer.add_figure(name, fig)

    def log_params(self, params: Dict[str, Any]):
        self.writer.add_hparams(params, {})

    def add_tags(self, tags: List[str]):
        self.writer.add_text('tags', str(tags))