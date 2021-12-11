import numpy as np
from typing import Optional, Set, Union, Dict, Callable, Any, List, Tuple
from pathlib import Path
import pandas as pd
import json
import dataclasses
import subprocess
import argparse
from dataclasses import dataclass
from oru.json import resolve_files, flatten_dictionary, join_tuplekeys
import tqdm
import copy
import enum

@dataclass
class Config:
    aggregate_path: Path = dataclasses.field(default_factory=lambda: Path.cwd() / "agg")
    scratch_path: Path = dataclasses.field(default_factory=lambda: Path.cwd() / "raw")
    archive_sources: Path = dataclasses.field(default_factory=lambda: Path.cwd() / "result_parts")
    index_field : str = "index"

class COLUMN(enum.IntEnum):
    MASK = 0
    INDEX = 1

def default_column_map():
    return {
        COLUMN.MASK: 'is_optimal',
        COLUMN.INDEX: 'index',
    }

def _replace_sentinel_columns(cols: Tuple[str], colmap: Dict[COLUMN, str]) -> Tuple:
    cols = list(cols)
    for i, c in enumerate(cols):
        if c in colmap:
            cols[i] = colmap[c]

    return tuple(cols)

class AggBase:
    def __init__(self, src: Union[str, Tuple[str,...]], dst: Union[str, Tuple[str,...]]):
        self.src = src if isinstance(src, tuple) else (src, )
        self.dst = dst if isinstance(dst, tuple) else (dst, )

    def aggregate(self, *values):
        raise NotImplementedError

    def _replace_sentinels(self, mask_col: str):
        self.src = _replace_sentinel_columns(self.src, mask_col)
        self.dst = _replace_sentinel_columns(self.dst, mask_col)

    def __repr__(self) -> str:
        src = self.src[0] if len(self.src) == 1 else ", ".join(self.src)
        if self.src == self.dst:
            inner = src
        else:
            if len(self.dst) == 0:
                dst = '()'
            elif len(self.dst) == 1:
                dst = self.dst[0]
            else:
                dst = ", ".join(self.dst)
            inner = f"{src} => {dst}"

        return f'{self.__class__.__name__}({inner})'

class AggOneToOne(AggBase):
    def __init__(self, src: str, dst: str = None):
        dst = src if dst is None else dst
        super().__init__(src, dst)

    def aggregate(self, values: np.ndarray):
        raise NotImplementedError


class AggOneToOneMasked(AggBase):
    def __init__(self, src: str, dst: str = None):
        dst = src if dst is None else dst
        super().__init__((src, COLUMN.MASK), dst)

    def aggregate(self, values: np.ndarray, mask: np.ndarray):
        raise NotImplementedError

class Masked(AggOneToOneMasked):
    def __init__(self, agg: AggOneToOne):
        super().__init__(agg.src, agg.dst)
        self.agg = agg

    def aggregate(self, values: np.ndarray, mask: np.ndarray):
        if not mask.any():
            return np.nan
        else:
            return self.agg.aggregate(values)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.agg})"

def _agg_same(values: np.ndarray, rtol=None, atol=None):
    if values.dtype.kind in "fc":
        val = values.mean()
    elif values.dtype.kind == "O":
        assert len(set(values)) == 1
        return values[0]
    else:
        val = values[0]

    rtol = rtol if rtol is not None else 1.e-8
    atol = atol if atol is not None else 1.e-5
    assert np.allclose(values, val, rtol=rtol, atol=atol)
    return val


class Count(AggBase):
    def __init__(self, dst: str):
        super().__init__(COLUMN.MASK, dst)

    def aggregate(self, values):
        return len(values)

class CountTrue(AggOneToOne):
    def aggregate(self, values: np.ndarray):
        assert values.dtype == np.bool
        return int(values.sum())


class Same(AggOneToOne):
    def __init__(self, *args, atol=None, rtol=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.atol = atol
        self.rtol = rtol

    def aggregate(self, values: np.ndarray):
        return _agg_same(values, atol=self.atol, rtol=self.rtol)

class Mean(AggOneToOne):
    def aggregate(self, values: np.ndarray):
        return values.mean()

class Median(AggOneToOne):
    def aggregate(self, values: np.ndarray):
        return np.median(values)

class Max(AggOneToOne):
    def aggregate(self, values: np.ndarray):
        return values.max()

class Min(AggOneToOne):
    def aggregate(self, values: np.ndarray):
        return values.min()

class Drop(AggBase):
    def __init__(self, agg: AggBase):
        super().__init__(agg.src, ())
        self.agg = agg

    def aggregate(self, *values):
        self.agg.aggregate(*values)

class FinalObj(AggOneToOneMasked):
    def __init__(self, minimisation: bool, src: str, dst: str = None):
        super().__init__(src, dst)
        self.min = minimisation

    def aggregate(self, values: np.ndarray, mask: np.ndarray):
        if not mask.any():
            if self.min:
                return values.min()
            else:
                return values.max()
        return _agg_same(values[mask])



class Aggregator:
    def __init__(self,
                 file_config: Config,
                 aggregate_columns: List[AggBase],
                 col_coverters: Dict[str, Callable[[str], Any]] = None,
                 col_defaults: Dict[str, Any] = None,
                 col_map=None,
                 ignore_missing_cols=False,
                 assert_length=None,
                 ignore_nan_in_agg=None
                 ):

        self.ignore_nan_in_agg = ignore_nan_in_agg or set()
        self.ignore_missing_cols = ignore_missing_cols
        self.config = file_config
        self.col_defaults = col_defaults or {}
        self.col_coverters = col_coverters or {}
        self.aggregators = copy.deepcopy(aggregate_columns)
        self.col_map = default_column_map()
        self.assert_length = assert_length

        if col_map:
            self.col_map.update(col_map)
        self.progbar = None

        for agg in self.aggregators:
            agg._replace_sentinels(self.col_map)

    def preprocess(self, df: pd.DataFrame):
        pass


    def load(self, p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, index_col=self.col_map[COLUMN.INDEX], converters=self.col_coverters)
        df.fillna(self.col_defaults, inplace=True)
        for c, v in self.col_defaults.items():
            if c not in df.columns:
                df[c] = v
        self.preprocess(df)
        return df


    def aggregate_group(self, group_paths: List[Path]) -> Tuple[Dict, pd.DataFrame]:
        group = [self.load(p) for p in group_paths]
        with open(self.parameter_file(group[0]), 'r') as fp:
            params = json.load(fp)
        params = self.extract_group_parameters(params)

        group = pd.concat(group)
        src_columns = set(group.columns)
        assert self.mask_col in src_columns, "Mask column is not in dataset"
        dst_columns = {}
        aggregators = []

        for agg in self.aggregators:
            for c in agg.src:
                if c not in src_columns:
                    if self.ignore_missing_cols:
                        self.progbar.write(f'Dropping {agg}: column {c} is missing')
                        break
                    else:
                        raise KeyError(c)
            else:
                aggregators.append(agg)

                for c in agg.dst:
                    assert c not in dst_columns, "duplicate destination column"
                    dst_columns[c] = []

        dst_columns[self.index_col] = []

        for index, samples in group.groupby(level=0):
            if self.assert_length is not None:
                assert self.assert_length == len(samples), f"Expected {self.assert_length} samples, got {len(samples)}"
            dst_columns[self.index_col].append(index)

            for agg in aggregators:
                values = [samples[c].values for c in agg.src]
                values = agg.aggregate(*values)

                if len(agg.dst) == 0:
                    pass
                elif len(agg.dst) == 1:
                    dst_columns[agg.dst[0]].append(values)
                else:
                    assert len(values) == len(agg.dst), "number of return values doesn't match number of dst columns"
                    for c, v in zip(agg.dst, values):
                        dst_columns[c].append(v)

        aggdf = pd.DataFrame.from_dict(dst_columns)
        aggdf.set_index(self.index_col, inplace=True)
    
        na_cols = [c for c, has_na in aggdf.isna().any().iteritems() if has_na and c not in self.ignore_nan_in_agg]
        assert not na_cols, f"NaNs in aggregated data ({', '.join(na_cols)}), use col_defaults or ignore_nan_in_agg"

        return params, aggdf

    @property
    def index_col(self):
        return self.col_map[COLUMN.INDEX]

    @property
    def mask_col(self):
        return self.col_map[COLUMN.MASK]

    def parameter_file(self, df: pd.DataFrame) -> Path:
        return self.config.scratch_path / df['param.param_name'].values[0] / "parameters.json"

    def extract_group_parameters(self, params: Dict) -> Dict:
        del params['gurobi']['Seed']
        return params

    def find_csvs(self) -> Dict[str, List[Path]]:
        groups = {}
        for f in self.config.scratch_path.glob("*.csv"):
            g = f.stem.split('.')[0]
            groups.setdefault(g, []).append(f)
        return groups

    def run(self):
        self.config.aggregate_path.mkdir(exist_ok=True, parents=True)
        csvs = self.find_csvs()

        with tqdm.tqdm(total=len(csvs), desc="Aggregating samples") as bar:
            self.progbar = bar
            for groupname, g in self.find_csvs().items():
                params, df = self.aggregate_group(g)
                dest = self.config.aggregate_path / f"{groupname}.csv"
                df.to_csv(dest)
                bar.set_postfix_str(f"wrote {dest.relative_to(Path.cwd())}")
                with open(self.config.aggregate_path / f"{groupname}_parameters.json", 'w') as fp:
                    json.dump(params, fp, indent='  ')
                bar.update()
            bar.set_postfix_str("")

class Extractor:
    def __init__(self, config: Config):
        self.config = config
        self.progbar = None

    def run(self, force=False):
        archives = sorted(self.config.archive_sources.glob("*.tar.xz"))

        if not force and self.config.scratch_path.exists():
            return

        self.config.scratch_path.mkdir(exist_ok=True, parents=True)
        print(f"Extracting to {self.config.scratch_path.relative_to(Path.cwd())!s}/")
        with tqdm.tqdm(total=len(archives), desc="Extracting archives") as progbar:
            self.progbar = progbar

            for arcv in archives:
                progbar.set_postfix_str(str(arcv.relative_to(Path.cwd())))
                subprocess.check_output(["tar", "-Jxf", str(arcv.absolute())], cwd=self.config.scratch_path.absolute())
                self.progbar.update()

        self.post_extract()

        groups = [p for p in self.config.scratch_path.iterdir() if p.is_dir()]
        with tqdm.tqdm(total=len(groups), desc="Creating CSVs") as bar:
            self.progbar = bar

            for p in groups:
                self.progbar.set_postfix_str(f"{p.relative_to(Path.cwd())}")
                self.process_parameter_group(p)
                self.progbar.update()

    def process_parameter_group(self, path: Path):
        pname = path.name
        records = [
            self.load_one(f)
            for f in path.glob("*index.json")
        ]
        if len(records) == 0:
            self.progbar.write(f"no samples found for {path.relative_to(Path.cwd())}")
            return
        df = pd.DataFrame.from_records(records, index=self.config.index_field)
        df.sort_index(inplace=True)
        csv = path.parent / f"{pname}.csv"
        df.to_csv(csv)

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
        p.add_argument("-f", "--force", action="store_true")
        return p.parse_args().force

def force_nonnegative_int(s) -> int:
    if not s:
        return -1
    try:
        return int(s)
    except ValueError:
        pass
    return round(float(s))
