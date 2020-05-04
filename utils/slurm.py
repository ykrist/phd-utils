import oru.slurm

_SLURM_PYTHON_FILE_TEMPLATE = r"""#!/bin/bash
conda activate or
"""

class BaseExperiment(oru.slurm.Experiment):
    OUTPUTS = {"indexfile" : {"type" : "string", "derived" : True}}
    ROOT_PATH = 'logs/'

    def define_derived(self):
        super().define_derived()
        self.outputs['indexfile'] = self.get_output_path("index.json")

    def write_index_file(self):
        raise NotImplementedError

    @property
    def resource_slurm_script(self):
        return _SLURM_PYTHON_FILE_TEMPLATE

