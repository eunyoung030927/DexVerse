"""Load only what the Diffusion Policy reads: LeRobot otherwise queries every observation.* column of the dataset."""
import lerobot.datasets.factory as factory
import lerobot.scripts.lerobot_train as trainer

INDEX_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
TABLE_KEYS = ("observation.state", "action")


def install_worker_settings() -> None:
    """Keep the workers and their open video decoders across epochs (LeRobot restarts them every epoch), and queue one
    batch per worker: a 480x640 batch takes ~1 GB of /dev/shm (32 GB here) and two trainings share it."""
    data_loader = trainer.torch.utils.data.DataLoader
    init = data_loader.__init__

    def persistent_init(self, *args, **kwargs):
        workers = kwargs.get("num_workers", 0) > 0
        init(self, *args, **{**kwargs, "persistent_workers": workers, "prefetch_factor": 1 if workers else None})

    data_loader.__init__ = persistent_init


def install_needed_keys() -> None:
    resolve_delta_timestamps = factory.resolve_delta_timestamps
    make_dataset = trainer.make_dataset

    def needed_delta_timestamps(cfg, ds_meta):
        keys = set(ds_meta.video_keys) | set(TABLE_KEYS)
        return {key: value for key, value in resolve_delta_timestamps(cfg, ds_meta).items() if key in keys}

    def needed_dataset(cfg):
        dataset = make_dataset(cfg)
        dataset.hf_dataset = dataset.hf_dataset.select_columns([*TABLE_KEYS, *INDEX_COLUMNS])
        return dataset

    factory.resolve_delta_timestamps = needed_delta_timestamps
    trainer.make_dataset = needed_dataset
