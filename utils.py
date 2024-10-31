import copy
import glob
import re
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps


def _copy_image_meta(src: Image.Image, dest: Image.Image):
    preserve_metadata_keys = ["info", "icc_profile", "exif", "dpi", "applist", "format"]
    for key in preserve_metadata_keys:
        if hasattr(src, key):
            setattr(dest, key, copy.deepcopy(getattr(src, key)))
    return dest


def safe_load_image(image: Union[bytes, str], return_numpy=True) -> np.ndarray:
    """Load an image from bytes or a file path, and ensure the orientation is correct."""
    # make sure image is bytes or a valid file path
    if isinstance(image, str):
        with open(image, "rb") as f:
            image = f.read()
    elif not isinstance(image, bytes):
        raise TypeError(f"image must be bytes or a file path, not {type(image)}")
    pil_image = Image.open(BytesIO(image))

    # Make sure the orientation is correct
    if hasattr(pil_image, "_getexif") and pil_image._getexif() is not None:
        new_pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = _copy_image_meta(pil_image, new_pil_image)

    if return_numpy:
        return np.array(pil_image)

    return pil_image
