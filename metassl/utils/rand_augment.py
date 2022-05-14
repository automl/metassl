# Code based on:
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# Code adapted to work also for segmentation
import random

import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps

####################################################################################################
# IDENTITY
####################################################################################################


def Identity(data, _, __):
    return data


####################################################################################################
# COLOR OPS
####################################################################################################


def AutoContrast(data, v, is_segmentation):
    if is_segmentation:
        return PIL.ImageOps.autocontrast(data[0], v), data[1]
    else:
        return PIL.ImageOps.autocontrast(data, v)


def Invert(data, _, is_segmentation):
    if is_segmentation:
        return PIL.ImageOps.invert(data[0]), data[1]
    else:
        return PIL.ImageOps.invert(data)


def Equalize(data, _, is_segmentation):
    if is_segmentation:
        return PIL.ImageOps.equalize(data[0]), data[1]
    else:
        return PIL.ImageOps.equalize(data)


def Solarize(data, v, is_segmentation):  # [0, 256]
    assert 0 <= v <= 256
    if is_segmentation:
        return PIL.ImageOps.solarize(data[0], v), data[1]
    else:
        return PIL.ImageOps.solarize(data, v)


def Posterize(data, v, is_segmentation):  # [4, 8]
    v = int(v)
    v = max(1, v)
    if is_segmentation:
        return PIL.ImageOps.posterize(data[0], v), data[1]
    else:
        return PIL.ImageOps.posterize(data, v)


def Contrast(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Contrast(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Contrast(data).enhance(v)


def Color(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Color(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Color(data).enhance(v)


def Brightness(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Brightness(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Brightness(data).enhance(v)


def Sharpness(data, v, is_segmentation):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    if is_segmentation:
        return PIL.ImageEnhance.Sharpness(data[0]).enhance(v), data[1]
    else:
        return PIL.ImageEnhance.Sharpness(data).enhance(v)


####################################################################################################
# GEOMETRIC OPS
####################################################################################################


def ShearX(data, v, is_segmentation):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), PIL.Image.NEAREST
        )
        return image, mask
    else:
        return data.transform(data.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), PIL.Image.BILINEAR)


def ShearY(data, v, is_segmentation):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), PIL.Image.NEAREST
        )
        return image, mask
    else:
        return data.transform(data.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), PIL.Image.BILINEAR)


def TranslateX(data, v, is_segmentation):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        v_0 = v * data[0].size[0]
        v_1 = v * data[1].size[0]
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, 0, v_0, 0, 1, 0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, 0, v_1, 0, 1, 0), PIL.Image.NEAREST
        )
        return image, mask
    else:
        v = v * data.size[0]
        return data.transform(data.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), PIL.Image.BILINEAR)


def TranslateY(data, v, is_segmentation):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        v_0 = v * data[0].size[0]
        v_1 = v * data[1].size[0]
        image = data[0].transform(
            data[0].size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v_0), PIL.Image.BILINEAR
        )
        mask = data[1].transform(
            data[1].size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v_1), PIL.Image.NEAREST
        )
        return image, mask
    else:
        v = v * data.size[0]
        return data.transform(data.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), PIL.Image.BILINEAR)


def Rotate(data, v, is_segmentation):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    if is_segmentation:
        return data[0].rotate(v), data[1].rotate(v)
    else:
        return data.rotate(v)


def RandomResizeCrop(data, v, is_segmentation):
    if is_segmentation:
        raise NotImplementedError

    # Get data size (32x32 for CIFAR10)
    data_width = data.size[0]
    data_height = data.size[1]

    # Resize data (if v is between 0.2 and 1.0: -> 'zoom in')
    data = data.resize((int(data_width + data_width * v), int(data_height + data_height * v)))
    data_resized_width = data.size[0]
    data_resized_height = data.size[1]

    # Crop to get original data size back (32x32 for CIFAR10)
    left = random.randrange(0, data_resized_width - data_width)
    top = random.randrange(0, data_resized_height - data_height)
    right = left + data_width
    bottom = top + data_height
    data = data.crop((left, top, right, bottom))

    return data


####################################################################################################


def augment_list():  # default opterations used in RandAugment paper
    augment_list = [
        (Identity, 0, 1),
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (Rotate, 0, 30),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (TranslateX, 0.0, 0.33),
        (TranslateY, 0.0, 0.33),
    ]

    return augment_list


class RandAugment:
    def __init__(self, neps_hyperparameters=None, is_segmentation=False):
        self.num_ops = neps_hyperparameters["num_ops"] if neps_hyperparameters is not None else 3
        self.magnitude = (
            neps_hyperparameters["magnitude"] if neps_hyperparameters is not None else 15
        )
        self.augment_list = augment_list()
        self.is_segmentation = is_segmentation

    def __call__(self, data):
        basic_op = [(RandomResizeCrop, 0.2, 1.0)]  # Important for SimSiam
        ops = random.choices(self.augment_list, k=self.num_ops)
        ops = basic_op + ops
        for op, minval, maxval in ops:
            magnitude_val = (float(self.magnitude) / 30) * float(maxval - minval) + minval
            data = op(data, magnitude_val, self.is_segmentation)
        return data


class TrivialAugment:
    def __init__(self, is_segmentation=False):
        self.augment_list = augment_list()
        self.is_segmentation = is_segmentation

    def __call__(self, data):
        basic_op = [(RandomResizeCrop, 0.2, 1.0)]  # Important for SimSiam
        ops = random.choices(self.augment_list, k=1)
        ops = basic_op + ops
        magnitude = random.randint(0, 30)
        for op, minval, maxval in ops:
            magnitude_val = (float(magnitude) / 30) * float(maxval - minval) + minval
            data = op(data, magnitude_val, self.is_segmentation)
        return data
