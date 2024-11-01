import warnings

from .base_logger import BaseLogger
from typing import Dict, Any, List
import numpy as np
from PIL import Image


def is_basic(x):
    return isinstance(x, str) or isinstance(x, int) or isinstance(x, float) or isinstance(x, bool)


def convert_no_basic_to_str(sub_dict: Dict[str, Any]):
    return {k: v if is_basic(v)
    else str(v) if not isinstance(v, dict) else convert_no_basic_to_str(v)
            for k, v in sub_dict.items()}

def convert_no_basic_to_str_from_any(p: Any):
    if is_basic(p):
        return p
    elif isinstance(p, dict):
        return convert_no_basic_to_str(p)
    else:
        return str(p)



class NeptuneLogger(BaseLogger):

    def __init__(self, project=None, *args, **kwargs):
        super(NeptuneLogger, self).__init__(*args, **kwargs)
        import neptune
        from pathlib import Path
        home_path_api_token = Path.home() / '.neptune' / 'token.txt'
        local_path_api_token = Path('neptune') / 'token.txt'
        if local_path_api_token.exists():
            api_token = local_path_api_token
        elif home_path_api_token.exists():
            api_token = home_path_api_token
        else:
            warnings.warn('''Please create a file at .neptune/token.txt with your Neptune API token.
            Or add a file at neptune/token.txt''')
            raise FileNotFoundError('Neptune token not found')

        api_token = api_token.read_text().strip()
        if project is None:
            local_path_api_project = Path('neptune') / 'project.txt'
            if local_path_api_project.exists():
                project = local_path_api_project.read_text().strip()
            else:
                warnings.warn('''Please create a file at neptune/project.txt with your Neptune project name''')
                raise FileNotFoundError('Neptune project not found')

        self.run = neptune.init_run(
            project=project,
            api_token=api_token,
        )

    def stop(self):
        self.run.stop()

    def log(self, name: str, data: Any, step=None):
        self.run[name].append(data)

    def _log_fig(self, name: str, fig: Any):
        if isinstance(fig, np.ndarray):
            fig = Image.fromarray(fig)
        self.run[name].append(fig)

    def log_hparams(self, params: Dict[str, Any]):
        params = convert_no_basic_to_str(params)
        self.run['hyperparameters'] = params

    def log_params(self, params: Dict[str, Any]):
        params = convert_no_basic_to_str(params)
        self.run['parameters'] = params

    def add_tags(self, tags: List[str]):
        self.run['sys/tags'].add(tags)

    def log_name_params(self, name : str, params: Any):
        params = convert_no_basic_to_str_from_any(params)
        self.run[name] = params
