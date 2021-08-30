import numpy as np
from typing import Union, Dict, Callable, Any, List, Tuple
from pathlib import Path
import pandas as pd
import json
import dataclasses
import subprocess
import argparse
from dataclasses import dataclass
from oru.json import resolve_files, flatten_dictionary, join_tuplekeys


@dataclass
class Config:
    aggregate_path: Path = dataclasses.field(default_factory=lambda: Path.cwd() / "agg")
    scratch_path: Path = dataclasses.field(default_factory=lambda: Path.cwd() / "raw")
    archive_sources: Path = dataclasses.field(default_factory=lambda: Path.cwd() / "result_parts")

class AggSimpleBase:
    def aggregate(self, values: np.ndarray):
        raise NotImplementedError

class AggOnSolvedBase:
    def aggregate(self, solved_mask: np.ndarray, values: np.ndarray):
        raise NotImplementedError

class AggSame(AggSimpleBase):
    def aggregate(self, values: np.ndarray):
        if values.dtype.kind in "fc":
            val = values.mean()
        elif values.dtype.kind == "O":
            assert len(set(values)) == 1
            return values[0]
        else:
            val = values[0]
        assert np.allclose(values, val)
        return val

class Mean(AggSimpleBase):
    def aggregate(self, values: np.ndarray):
        return values.mean()

class Median(AggSimpleBase):
    def aggregate(self, values: np.ndarray):
        return np.median(values)

class SameSolved(AggOnSolvedBase):
    def __init__(self, default=np.nan):
        self.default=default

    def aggregate(self, solved_mask: np.ndarray, values: np.ndarray):
        values = values[solved_mask]

        if len(values) == 0:
            return self.default

        return AggSame().aggregate(values)


class FinalObj(AggOnSolvedBase):
    def __init__(self, minimisation: bool):
        self.min = minimisation

    def aggregate(self, solved_mask: np.ndarray, values: np.ndarray):
        if not solved_mask.any():
            if self.min:
                return values.min()
            else:
                return values.max()
        return SameSolved().aggregate(solved_mask, values)



class Aggregator:
    def __init__(self,
                 file_config: Config,
                 aggregate_columns: Dict[str, Union[AggSimpleBase, AggOnSolvedBase]],
                 col_coverters: Dict[str, Callable[[str], Any]] = None,
                 col_defaults: Dict[str, Any] = None,
                 index_col="index",
                 ignore_missing_cols=False,
                 ):

        self.ignore_missing_cols = ignore_missing_cols
        self.file_config = file_config
        self.col_defaults = col_defaults or {}
        self.col_coverters = col_coverters or {}
        self.columns = aggregate_columns
        self.index_col = index_col

    def preprocess(self, df: pd.DataFrame):
        pass

    def load(self, p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, index_col=self.index_col, converters=self.col_coverters)
        self.preprocess(df)
        return df

    def get_solved_mask(self, samples: pd.DataFrame) -> np.ndarray:
        return samples['info.model.status'].values == 2

    def aggregate_group(self, group: List[Path]) -> Tuple[Dict, pd.DataFrame]:
        group = [self.load(p) for p in group]
        with open(self.parameter_file(group[0]), 'r') as fp:
            params = json.load(fp)
        params = self.extract_group_parameters(params)

        columns = {c: [] for c in self.columns }
        columns["index"] = []
        columns["num_solved"] = []
        columns["num_total"] = []
        for index, samples in pd.concat(group).groupby(level=0):
            columns["index"].append(index)
            solve_mask = self.get_solved_mask(samples)
            columns["num_solved"].append(solve_mask.sum())
            columns["num_total"].append(len(samples))

            for c, agg in self.columns.items():
                if c not in samples.columns:
                    if self.ignore_missing_cols:
                        continue
                    else:
                        raise KeyError(c)
                values = samples[c].values

                if values.dtype.kind != 'O':
                    if c in self.col_defaults:
                        values[np.isnan(values)] = self.col_defaults[c]
                    assert not np.isnan(values).any(), f"NaNs in column {c}"

                if isinstance(agg, AggSimpleBase):
                    val = agg.aggregate(values)
                elif isinstance(agg, AggOnSolvedBase):
                    val = agg.aggregate(solve_mask, values)
                else:
                    raise NotImplementedError("aggregators must subclass AggSimpleBase or AggOnSolvedBase")

                columns[c].append(val)

        columns = {n: c for n, c in columns.items() if len(c) > 0}
        aggdf = pd.DataFrame.from_dict(columns)
        aggdf.set_index("index", inplace=True)
        return params, aggdf

    def parameter_file(self, df: pd.DataFrame):
        return self.file_config.scratch_path / df['param.param_name'].values[0] / "parameters.json"

    def extract_group_parameters(self, params: Dict) -> Dict:
        del params['gurobi']['Seed']
        return params

    def find_csvs(self) -> Dict[str, List[Path]]:
        groups = {}
        for f in self.file_config.scratch_path.glob("*.csv"):
            g = f.stem.split('.')[0]
            groups.setdefault(g, []).append(f)
        return groups

    def run(self):
        self.file_config.aggregate_path.mkdir(exist_ok=True, parents=True)
        for groupname, g in self.find_csvs().items():
            params, df = self.aggregate_group(g)
            dest = self.file_config.aggregate_path / f"{groupname}.csv"
            df.to_csv(dest)
            print(f"wrote {dest.relative_to(Path.cwd())}")
            with open(self.file_config.aggregate_path / f"{groupname}_parameters.json", 'w') as fp:
                json.dump(params, fp, indent='  ')


class Extractor:
    def __init__(self, config: Config):
        self.config = config

    def run(self, force=False):
        archives = sorted(self.config.archive_sources.glob("*.tar.xz"))

        if not force and self.config.scratch_path.exists():
            return

        self.config.scratch_path.mkdir(exist_ok=True, parents=True)
        print(f"Extracting to {self.config.scratch_path.relative_to(Path.cwd())!s}/")
        for arcv in archives:
            print(f"Extracting {arcv.relative_to(Path.cwd())!s} ... ", end='')
            subprocess.check_output(["tar", "-Jxf", arcv.absolute()], cwd=self.config.scratch_path.absolute())
            print("done")

        self.post_extract()

        for p in self.config.scratch_path.iterdir():
            if p.is_dir():
                print(f"processing {p.relative_to(Path.cwd())}...")
                self.process_parameter_group(p)

    def process_parameter_group(self, path: Path):
        pname = path.name
        records = [
            self.load_one(f)
            for f in path.glob("*index.json")
        ]
        if len(records) == 0:
            print(f"no samples found for {path.relative_to(Path.cwd())}")
            return
        df = pd.DataFrame.from_records(records, index="index")
        csv = path.parent / f"{pname}.csv"
        df.to_csv(csv)
        print(f"wrote {csv.relative_to(Path.cwd())}")

    def load_one(self, path: Path) -> Dict:
        with open(path, "r") as fp:
            data = json.load(fp)
        data = resolve_files(data, path.parent, callback=lambda p, r: self.missing_hook(p, r))
        with open(path.with_name("parameters.json"), 'r') as fp:
            data['param'] = json.load(fp)
        return join_tuplekeys(flatten_dictionary(data), ".")


    def post_extract(self):
        pass

    def missing_hook(self, path: Path, reason):
        print(f"unable to find {path.relative_to(Path.cwd())}")

    @classmethod
    def parse_args(cls) -> bool:
        p = argparse.ArgumentParser()
        p.add_argument("f", "force", action="store_true")
        return p.parse_args().force

def parse_gurobi_status(s) -> int:
    if not s:
        return -1
    try:
        return int(s)
    except ValueError:
        pass
    return round(float(s))
