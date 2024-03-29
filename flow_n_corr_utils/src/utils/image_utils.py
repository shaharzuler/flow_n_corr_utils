import numpy as np


def min_max_norm(img):
    min_ = img.min()
    max_ = img.max()
    img_norm = (img-min_)/(max_-min_) 
    return img_norm

def min_max_norm_det_vals(img, min_, max_):
    img_norm = (img-min_)/(max_-min_) 
    return img_norm

def get_norm_img(img_path, output_shape):
    if img_path is not None:
        orig_img = np.load(img_path)
        img_norm = min_max_norm(orig_img)
    else:
        img_norm = np.zeros(shape=output_shape[:-1], dtype=float) + 0.5
    return img_norm

def t3xy_to_xy3(img:np.ndarray) -> np.ndarray:
    return np.transpose(img, (1,2,0))
