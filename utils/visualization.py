import numpy as np
import matplotlib.cm as cm

def apply_colormap(image, cmap_name='coolwarm'):
    cmap = cm.get_cmap(cmap_name)
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
    colored = cmap(normalized)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return colored_rgb

