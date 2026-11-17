"""handmask (261115): hand segmentation helpers shared by the demo mask renderer (hm_render_masks.py) and the online
eval masking (hm_eval_masked.py), so training and evaluation masks are made the same way.
Mask = pixels of the camera's instance_id_segmentation_fast whose prim path lies under the env Robot prim (every link
of the floating hand, any hand model), optionally dilated by `dilate` px; masking sets those RGB pixels to 0.
"""
import torch

SEG = "instance_id_segmentation_fast"
ROBOT_TOKEN = "/Robot/"


def add_segmentation(cam_cfg) -> None:
    """Add the instance-id annotator to a TiledCameraCfg (rgb stays the first data type, values are raw ids)."""
    if SEG not in cam_cfg.data_types:
        cam_cfg.data_types = list(cam_cfg.data_types) + [SEG]
    cam_cfg.colorize_instance_id_segmentation = False


def is_hand_label(label: str) -> bool:
    label = str(label)
    return ROBOT_TOKEN in label or label.endswith("/Robot")


def hand_mask_from_camera(cam, env_index: int = 0):
    """Return (bool mask HxW on the camera device, list of hand prim labels present)."""
    seg = cam.data.output[SEG][env_index, ..., 0]
    info = cam.data.info.get(SEG) if isinstance(cam.data.info, dict) else cam.data.info[env_index].get(SEG)
    id2lab = (info or {}).get("idToLabels", {})
    hand_items = [(int(i), str(lab)) for i, lab in id2lab.items() if is_hand_label(lab)]
    ids = [i for i, _ in hand_items]
    labels = [lab for _, lab in hand_items]
    if not ids:
        return torch.zeros(seg.shape, dtype=torch.bool, device=seg.device), labels
    hid = torch.tensor(ids, device=seg.device, dtype=seg.dtype)
    return torch.isin(seg, hid), labels


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Square dilation of a (..., H, W) bool mask by px pixels."""
    if px <= 0:
        return mask
    m = mask.float().reshape(-1, 1, *mask.shape[-2:])
    m = torch.nn.functional.max_pool2d(m, 2 * px + 1, stride=1, padding=px)
    return m.reshape(mask.shape) > 0.5


def apply_mask(rgb, mask):
    """rgb (..., H, W, 3) uint8 numpy or tensor; mask (..., H, W) bool -> copy with masked pixels = 0."""
    out = rgb.copy() if hasattr(rgb, "copy") else rgb.clone()
    out[mask] = 0
    return out
