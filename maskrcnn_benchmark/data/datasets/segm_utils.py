import numpy as np


def mask_to_bbox(mask, img_size):
    """Function from ns-vqa"""
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]

    x0, y0, x1, y1 = clip_xyxy_to_image(x0, y0, x1, y1, img_size[0], img_size[1])

    return np.array((x0, y0, x1, y1), dtype=np.float32)


def clip_xyxy_to_image(x0, y0, x1, y1, height, width):
    """Clip coordinates to an image with the given height and width."""
    x0 = np.minimum(width - 1., np.maximum(0., x0))
    y0 = np.minimum(height - 1., np.maximum(0., y0))
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    return x0, y0, x1, y1
