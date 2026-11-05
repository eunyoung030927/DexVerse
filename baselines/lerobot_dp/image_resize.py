"""240x320 input for the LeRobot Diffusion Policy (the datasets are 480x640).

Training resizes the decoded video frames in the data loader workers (a 480x640 float32 batch is ~1 GB on its way to
the GPU; resized it is a quarter). Evaluation feeds 480x640 camera images, so the encoders also resize whatever still
arrives at full size. The checkpoint's crop_shape tells which input it was trained on, so training and evaluation agree.
"""
import torch.nn.functional as F

from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder

FULL_CROP = (432, 576)  # of the dataset's 480x640
RESIZED_SHAPE = (240, 320)
RESIZED_CROP = (216, 288)


def resize(x):
    return F.interpolate(x, size=RESIZED_SHAPE, mode="bilinear", antialias=True)


def install_resize() -> None:
    """Encoder side: resize inputs that are not 240x320 yet (evaluation; training before 2026-11-01 14:00 KST)."""
    forward = DiffusionRgbEncoder.forward

    def resized_forward(self, x):
        return forward(self, x if tuple(x.shape[-2:]) == RESIZED_SHAPE else resize(x))

    DiffusionRgbEncoder.forward = resized_forward


def install_dataset_resize() -> None:
    """Data loader side: frames leave the video decoder at 240x320 (lerobot env only)."""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    decode = lerobot_dataset.decode_video_frames

    def decode_resized(*args, **kwargs):
        return resize(decode(*args, **kwargs))

    lerobot_dataset.decode_video_frames = decode_resized


def install_resize_for(config) -> bool:
    resized = tuple(config.crop_shape) == RESIZED_CROP
    if resized:
        install_resize()
    return resized
