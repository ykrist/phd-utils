import oru.slurm
from pathlib import Path

_SLURM_PYTHON_FILE_TEMPLATE = r"""#!/bin/bash
source ~/.profile
conda activate or
"""

# FIXME this is a hack, not portable
def _repo_root() -> Path:
    return (Path(__file__).parent/'../../..').resolve()


class BaseExperiment(oru.slurm.Experiment):
    OUTPUTS = {"indexfile" : {"type" : "path","coerce":Path, "derived" : True}}
    ROOT_PATH = _repo_root()/'logs'

    def define_derived(self):
        super().define_derived()
        self.outputs['indexfile'] = self.get_output_path("index.json")

    def write_index_file(self):
        raise NotImplementedError

    @property
    def resource_slurm_script(self):
        return _SLURM_PYTHON_FILE_TEMPLATE

