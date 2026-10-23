"""
Dataset loader for FDGAN: images with one bounding box each, read from a JSON file of the same stem.

Replaces guided_diffusion.image_datasets. Differences:
- no mpi4py; sharding uses torch.distributed rank and world size when initialised
- box JSON with keys {"x","y","w","h"}, read as
    * (x, y, width, height) when bb_is_xywh=True (Traffic Signs)
    * (x1, y1, x2, y2) when bb_is_xywh=False (Cityscapes, COCO); bb_xyxy_end_inclusive=True adds 1 to x2, y2
- the box is mirrored together with the image on a random horizontal flip

Each item is (image in [-1, 1], {"image_filenames": str, "flip": bool, "bb": {"x", "y", "w", "h"}}); the default
collation turns the box values into tensors, which is what the training loop expects.
"""

import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# Optional: blobfile is nice for recursive listing / remote FS.
try:
    import blobfile as bf  # type: ignore
except Exception:  # pragma: no cover
    bf = None

IMG_EXTS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}


def _rank_world() -> Tuple[int, int]:
    """Return (rank, world_size) using torch.distributed if initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _list_image_files_recursively(data_dir: str) -> List[str]:
    """
    Recursive file listing supporting:
      - local filesystem (os.walk)
      - blobfile if installed
    """
    results: List[str] = []
    if bf is None:
        for root, _, files in os.walk(data_dir):
            for fn in sorted(files):
                if "." in fn and fn.rsplit(".", 1)[-1].lower() in IMG_EXTS:
                    results.append(os.path.join(root, fn))
        return results

    # blobfile path
    def _recurse(d: str):
        for entry in sorted(bf.listdir(d)):
            full_path = bf.join(d, entry)
            if bf.isdir(full_path):
                _recurse(full_path)
            else:
                if "." in entry and entry.rsplit(".", 1)[-1].lower() in IMG_EXTS:
                    results.append(full_path)

    _recurse(data_dir)
    return results


def _infer_bb_dir(data_dir: str) -> Optional[str]:
    """
    Try to guess where bbox json files live.
    Common patterns:
      - <data_dir>_bb
      - <data_dir>/bb
      - sibling folder with 'bb' in name
    """
    candidates = [
        data_dir + "_bb",
        data_dir + "_bboxes",
        os.path.join(data_dir, "bb"),
        os.path.join(data_dir, "bboxes"),
        os.path.join(data_dir, "bbox"),
        os.path.join(data_dir, "annotations_bb"),

        # common naming variants: images dir ends with "_Img" / "_img"
        data_dir.replace("_Img", "_bb"),
        data_dir.replace("_img", "_bb"),
        data_dir.replace("_Img", "_bboxes"),
        data_dir.replace("_img", "_bboxes"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def load_data(
    *,
    data_dir: str,
    batch_size: int,
    image_size: int,
    class_cond: bool = False,       # unused; kept for API compatibility
    deterministic: bool = False,
    random_crop: bool = False,      # ignored; no cropping
    random_flip: bool = True,
    # --- bbox options
    bb_dir: Optional[str] = None,
    bb_is_xywh: bool = False,
    bb_xyxy_end_inclusive: bool = True,
    resize_if_needed: bool = False,
):
    """Infinite generator of (images, cond) batches from data_dir, with the boxes read from bb_dir (or the FDGAN_BB_DIR
    variable, or a box folder inferred next to data_dir). class_cond and random_crop are accepted for compatibility
    with guided-diffusion and ignored.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)
    if len(all_files) == 0:
        raise ValueError(f"No images found in {data_dir}")

    # environment-variable override of the box folder
    if bb_dir is None:
        bb_dir = os.environ.get("FDGAN_BB_DIR", None)
    if bb_dir is None:
        bb_dir = _infer_bb_dir(data_dir)

    dataset = ImageDataset(
        image_paths=all_files,
        resolution=int(image_size),
        random_flip=bool(random_flip),
        bb_dir=bb_dir,
        bb_is_xywh=bool(bb_is_xywh),
        bb_xyxy_end_inclusive=bool(bb_xyxy_end_inclusive),
        resize_if_needed=bool(resize_if_needed),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )

    while True:
        yield from loader


class ImageDataset(Dataset):
    """Images with one bounding box each. The box comes from <bb_dir>/<stem>.json (or a JSON file next to the image), is
    converted to exclusive corner coordinates and is mirrored with the image on a random flip. The file list is
    sharded by rank for distributed runs.
    """
    def __init__(
        self,
        image_paths: List[str],
        resolution: int,
        random_flip: bool,
        bb_dir: Optional[str],
        bb_is_xywh: bool,
        bb_xyxy_end_inclusive: bool,
        resize_if_needed: bool,
    ):
        super().__init__()
        self.resolution = int(resolution)
        self.random_flip = bool(random_flip)
        self.bb_dir = bb_dir
        self.bb_is_xywh = bool(bb_is_xywh)
        self.bb_xyxy_end_inclusive = bool(bb_xyxy_end_inclusive)
        self.resize_if_needed = bool(resize_if_needed)

        rank, world = _rank_world()
        # Shard the file list by rank for distributed runs
        self.local_images = image_paths[rank::world]

        if self.bb_dir is None:
            raise ValueError(
                "bb_dir could not be inferred. Either:\n"
                "  1) set env var FDGAN_BB_DIR=/path/to/bb_jsons, or\n"
                "  2) call load_data(..., bb_dir=...).\n"
                f"data_dir was: {image_paths[0] if image_paths else '??'}"
            )

    def __len__(self) -> int:
        return len(self.local_images)

    def _open_image(self, path: str) -> Image.Image:
        """Open an image as RGB, through blobfile for remote paths."""
        if bf is not None and (path.startswith("gs://") or path.startswith("az://") or path.startswith("s3://")):
            with bf.BlobFile(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        return img

    def _load_bb_json(self, stem: str, img_path: str) -> Dict[str, Any]:
        # Preferred: <bb_dir>/<stem>.json
        """Read the box of an image from its JSON file. Accepts the key sets x/y/w/h, x1/y1/x2/y2 and xmin/ymin/xmax/ymax
        and returns {'x', 'y', 'w', 'h'} as stored.
        """
        bb_path = os.path.join(self.bb_dir, stem + ".json")
        if not os.path.exists(bb_path):
            # Fallback: alongside the image
            sidecar = os.path.splitext(img_path)[0] + ".json"
            if os.path.exists(sidecar):
                bb_path = sidecar
            else:
                raise FileNotFoundError(f"BB json not found for {stem}: tried {bb_path} and {sidecar}")

        with open(bb_path, "r") as jf:
            bb = json.load(jf)

        # Normalize possible list wrapper
        if isinstance(bb, list):
            if len(bb) == 0:
                raise ValueError(f"Empty bbox list in {bb_path}")
            if len(bb) == 1 and isinstance(bb[0], dict):
                bb = bb[0]
            else:
                bb = bb[0]

        if not isinstance(bb, dict):
            raise ValueError(f"Unexpected bbox json type in {bb_path}: {type(bb)}")

        # Accept a few common key variants
        def pick(*keys, default=None):
            for k in keys:
                if k in bb:
                    return bb[k]
            return default

        x = pick("x", "x1", "xmin")
        y = pick("y", "y1", "ymin")
        w = pick("w", "x2", "xmax", "width")
        h = pick("h", "y2", "ymax", "height")
        if x is None or y is None or w is None or h is None:
            raise KeyError(f"bbox json missing required keys in {bb_path}. Got keys: {list(bb.keys())}")

        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    def __getitem__(self, idx: int):
        """The image as a float32 CHW array in [-1, 1] and the dictionary {'image_filenames', 'flip', 'bb'}. The box is
        clamped to the image, converted to the format selected by bb_is_xywh and mirrored if the image was flipped.
        """
        path = self.local_images[idx]
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]

        # ---- load image ----
        img = self._open_image(path)
        W, H = img.size

        # Resize if requested; otherwise enforce exact resolution
        if (W != self.resolution) or (H != self.resolution):
            if self.resize_if_needed:
                img = img.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
                W, H = img.size
            else:
                raise ValueError(
                    f"Image {filename} is {W}x{H}, expected {self.resolution}x{self.resolution}. "
                    f"Set resize_if_needed=True if you want auto-resize."
                )

        # ---- load BB ----
        bb = self._load_bb_json(stem, path)
        x = int(bb["x"]); y = int(bb["y"]); w = int(bb["w"]); h = int(bb["h"])

        # Interpret bb + convert to a consistent output format.
        # We return in the SAME representation as requested by bb_is_xywh:
        #   - bb_is_xywh=True:  {"x":x1, "y":y1, "w":width, "h":height}
        #   - bb_is_xywh=False: {"x":x1, "y":y1, "w":x2, "h":y2} where x2/y2 are EXCLUSIVE ends.
        if self.bb_is_xywh:
            x1, y1 = x, y
            width, height = w, h
            # clamp width/height to image
            width = max(1, min(width, W))
            height = max(1, min(height, H))
            # clamp start
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
        else:
            x1, y1 = x, y
            x2, y2 = w, h
            # If your JSON stores end coords inclusive (common when derived from pixel indices),
            # convert to exclusive.
            if self.bb_xyxy_end_inclusive:
                x2 += 1
                y2 += 1
            # clamp
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            # order
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
            # ensure at least 1px extent
            if x2 == x1: x2 = min(W, x1 + 1)
            if y2 == y1: y2 = min(H, y1 + 1)

        # ---- optional horizontal flip ----
        flip = False
        if self.random_flip and random.random() < 0.5:
            img = ImageOps.mirror(img)
            flip = True

            if self.bb_is_xywh:
                # x' = W - (x + width)
                x1 = W - (x1 + width)
                x1 = max(0, min(x1, W - 1))
            else:
                # For exclusive ends: (x1,x2) -> (W-x2, W-x1)
                x1, x2 = (W - x2), (W - x1)
                x1 = max(0, min(x1, W - 1))
                x2 = max(0, min(x2, W))
                if x2 < x1: x1, x2 = x2, x1
                if x2 == x1: x2 = min(W, x1 + 1)

        # ---- package bb output ----
        if self.bb_is_xywh:
            bb_out = {"x": int(x1), "y": int(y1), "w": int(width), "h": int(height)}
        else:
            bb_out = {"x": int(x1), "y": int(y1), "w": int(x2), "h": int(y2)}

        # ---- to CHW float32 in [-1, 1] ----
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        arr = np.transpose(arr, [2, 0, 1])  # CHW

        out_dict = {
            "image_filenames": filename,
            "flip": flip,
            "bb": bb_out,
        }
        return arr, out_dict
