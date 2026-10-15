"""Version-one implementations of the 20 baseline tasks, not a benchmark-wide fork."""

from dexverse.benchmark import V1_CONFIGS

from dexverse.tasks.utils.registration import register_env

for _task, (_module, _class) in V1_CONFIGS.items():
    register_env(f"{__name__}.config", _task, _module, _class)
