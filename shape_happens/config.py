import json
import yaml
import hashlib
import itertools
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod


class Runner(ABC):
    def __init__(self, global_config_path=None, grid_config_path=None, local_config_path=None, config_path=None):
        if config_path:
            config = self.load_yaml(config_path)
            self.global_args = config.get('global', {})
            self.grid_args = config.get('grid', {})
            self.local_args = config.get('local', [])
        elif global_config_path or grid_config_path or local_config_path:
            self.global_args = self.load_yaml(global_config_path) if global_config_path and Path(global_config_path).exists() else {}
            self.grid_args = self.load_yaml(grid_config_path) if grid_config_path and Path(grid_config_path).exists() else {}
            self.local_args = self.load_yaml(local_config_path) if local_config_path and Path(local_config_path).exists() else []
        else:
            raise ValueError("At least one of global_config_path, grid_config_path, local_config_path, or config_path must be provided.")

    def load_yaml(self, file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    def hash_args(self, setting: dict) -> str:
        setting_str = json.dumps(setting, sort_keys=True)
        return hashlib.md5(setting_str.encode()).hexdigest()

    def merge_args(self, global_args, grid_args, local_args):
        keys, values = zip(*grid_args.items()) if grid_args else ([], [])
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if keys else [{}]

        all_args = []
        for grid_combo in grid_combinations:
            for local_cfg in local_args:
                merged = deepcopy(global_args)
                merged.update(grid_combo)
                merged.update(local_cfg)

                if self.validate_args(merged):
                    all_args.append(merged)
        return all_args

    def run_all(self, multiprocessing=True):
        full_argsets = self.merge_args(self.global_args, self.grid_args, self.local_args)

        for args in full_argsets:
            if self.results_exist(args):
                print(f"Skipping existing experiment: {args}")
                continue
            if multiprocessing:
                p = mp.Process(target=self.run_experiment, args=(args,))
                p.start()
                p.join()
            else:
                self.run_experiment(args)

        self.combine_results(full_argsets)

    def validate_args(self, args):
        """Validate the arguments for the experiment"""
        return True

    @abstractmethod
    def combine_results(self, results_args):
        """Combine results from multiple experiments (e.g., aggregate metrics)"""
        pass

    @abstractmethod
    def results_exist(self, args):
        """Return True if the experiment result for `args` already exists"""
        pass

    @abstractmethod
    def run_experiment(self, args):
        """Run the actual experiment with the given arguments"""
        pass
