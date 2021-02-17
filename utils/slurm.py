import oru.slurm
from oru.constants import get_conda_info
from .constants import LOGS_DIR
from pathlib import Path

_SLURM_PYTHON_FILE_TEMPLATE = r"""#!/bin/bash
source ~/.profile
conda activate {active_prefix_name}
""".format_map(get_conda_info())


class BaseExperiment(oru.slurm.Experiment):
    OUTPUTS = {"indexfile" : {"type" : "path","coerce": Path, "derived" : True}}
    ROOT_PATH = LOGS_DIR

    def define_derived(self):
        super().define_derived()
        self.outputs['indexfile'] = self.get_output_path("index.json")

    def write_index_file(self):
        raise NotImplementedError

    @property
    def resource_slurm_script(self):
        return _SLURM_PYTHON_FILE_TEMPLATE

