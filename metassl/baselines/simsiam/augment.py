# Code based on:  https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# Code adapted to work also for segmentation
import random
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps

########################################################################################################################
# IDENTITY
########################################################################################################################

def Identity(data, _, __):
    return data


########################################################################################################################
# COLOR OPS
########################################################################################################################

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


########################################################################################################################
# GEOMETRIC OPS
########################################################################################################################

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

########################################################################################################################

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

def col_augment_list_with_type():
    augment_list = [
        (AutoContrast, 0, 1, "col"),
        (Equalize, 0, 1, "col"),
        (Posterize, 0, 4, "col"),
        (Solarize, 0, 256, "col"),
        (Color, 0.1, 1.9, "col"),
        (Contrast, 0.1, 1.9, "col"),
        (Brightness, 0.1, 1.9, "col"),
        (Sharpness, 0.1, 1.9, "col"),
    ]

    return augment_list


def geo_augment_list_with_type():
    augment_list = [
        (Rotate, 0, 30, "geo"),
        (ShearX, 0.0, 0.3, "geo"),
        (ShearY, 0.0, 0.3, "geo"),
        (TranslateX, 0.0, 0.33, "geo"),
        (TranslateY, 0.0, 0.33, "geo"),
    ]

    return augment_list


def col_augment_list():
    color_augment_list = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
    ]

    return color_augment_list


def geo_augment_list():
    geometric_augment_list = [
        (Rotate, 0, 30),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (TranslateX, 0.0, 0.33),
        (TranslateY, 0.0, 0.33),
    ]

    return geometric_augment_list

def get_annealing_p(epoch_start, epoch_end, p_start, p_end, current_epoch):
    epoch_end = epoch_end - 1
    current_p = (current_epoch - epoch_start) * (p_end - p_start) / (
        epoch_end - epoch_start
    ) + p_start
    return current_p

########################################################################################################################

class RandAugment:
    def __init__(self, num_ops, magnitude, is_segmentation=False):
        self.num_ops = num_ops  # TODO: optimize with BOHB
        self.magnitude = magnitude  # TODO: optimize with BOHB
        self.augment_list = augment_list()
        self.is_segmentation = is_segmentation

    def __call__(self, data):
        ops = random.choices(self.augment_list, k=self.num_ops)
        for op, minval, maxval in ops:
            magnitude_val = (float(self.magnitude) / 30) * float(maxval - minval) + minval
            data = op(data, magnitude_val, self.is_segmentation)

        return data


class TrivialAugment:
    def __init__(self, augmentation_ops_mode, is_segmentation=False):
        self.augmentation_ops_mode = augmentation_ops_mode
        self.augment_list = augment_list()
        self.color_augment_list = col_augment_list()
        self.geo_augment_list = geo_augment_list()
        self.is_segmentation = is_segmentation

    def __call__(self, data):
        # Select augmentation list to sample from
        if self.augmentation_ops_mode.startswith('default'):
            ops = random.choices(self.augment_list, k=1)
        elif self.augmentation_ops_mode.startswith('color-only'):
            ops = random.choices(self.color_augment_list, k=1)
        elif self.augmentation_ops_mode.startswith('geometric-only'):
            ops = random.choices(self.geo_augment_list, k=1)
        else:
            raise NotImplementedError
        # Choose whether to first apply RandomResizeCrop or not
        # If RandomResizeCrop is applied, it is applied before the other ops get applied (as it is done in SimSiam)
        if self.augmentation_ops_mode.endswith('RandomResizeCrop'):
            basic_op = [(RandomResizeCrop, 0.2, 1.)]
            ops = basic_op + ops

        magnitude = random.randint(0, 30)
        for op, minval, maxval in ops:
            magnitude_val = (float(magnitude) / 30) * float(maxval - minval) + minval
            data = op(data, magnitude_val, self.is_segmentation)
        return data


class SmartSamplingAugment:
    def __init__(self, max_epochs, current_epoch, is_segmentation=False):
        self.col_augment_list = col_augment_list()
        self.geo_augment_list = geo_augment_list()
        self.epoch_end = max_epochs
        self.current_epoch = current_epoch
        self.bohb_hyperparameters = None
        self.is_weighted_ops = True
        self.num_ops = 2
        self.is_segmentation = is_segmentation

    def __call__(self, data):
        # ANNEALING P
        # --------------------------------------------------------------------------------
        epoch_start = 0
        epoch_end = self.epoch_end
        p_start = 0
        p_end = 1
        apply_ops_prob = get_annealing_p(
            epoch_start, epoch_end, p_start, p_end, self.current_epoch
        )

        random_value = random.uniform(0, 1)  # random value between 0 and 1
        if random_value > apply_ops_prob:
            return data

        if self.is_weighted_ops:
            # SAMPLED OPS - WEIGHTED
            # ----------------------------------------------------------------------------
            # weights from randaugment paper:
            # rotate: 1.3, shear-x: 0.9, sherar-y: 0.9, translate-x: 0.4,
            # translate-y: 0.4, autocontrast: 0.1, sharpness: 0.1,
            # identity: 0.1, but we use annealing p instead
            # Sum of weights: 4.1 (without identity)

            sampled_ops = []
            for op in range(self.num_ops):
                random_value = random.uniform(0, 1)
                if random_value > 0 and random_value <= (1.3 / 4.1):
                    sampled_ops.extend([geo_augment_list_with_type()[0]])

                elif random_value > 1.3 / 4.1 and random_value <= (0.9 / 4.1 + 1.3 / 4.1):
                    sampled_ops.extend([geo_augment_list_with_type()[1]])

                elif random_value > (1.3 / 4.1 + 0.9 / 4.1) and random_value <= (
                    2 * 0.9 / 4.1 + 1.3 / 4.1
                ):
                    sampled_ops.extend([geo_augment_list_with_type()[2]])

                elif random_value > (1.3 / 4.1 + 2 * 0.9 / 4.1) and random_value <= (
                    0.4 / 4.1 + 2 * 0.9 / 4.1 + 1.3 / 4.1
                ):
                    sampled_ops.extend([geo_augment_list_with_type()[3]])

                elif random_value > (
                    0.4 / 4.1 + 2 * 0.9 / 4.1 + 1.3 / 4.1
                ) and random_value <= (2 * 0.4 / 4.1 + 2 * 0.9 / 4.1 + 1.3 / 4.1):
                    sampled_ops.extend([geo_augment_list_with_type()[4]])

                elif random_value > (
                    2 * 0.4 / 4.1 + 2 * 0.9 / 4.1 + 1.3 / 4.1
                ) and random_value <= (0.1 / 4.1 + 0.4 / 4.1 + 2 * 0.9 / 4.1 + 1.3 / 4.1):
                    sampled_ops.extend([col_augment_list_with_type()[0]])

                elif (
                    random_value > (0.1 / 4.1 + 0.4 / 4.1 + 2 * 0.9 / 4.1 + 1.3 / 4.1)
                    and random_value <= 1.0
                ):
                    sampled_ops.extend([col_augment_list_with_type()[7]])

                else:
                    raise ValueError("Uncorrect random_value")

            # MAGNITUDE
            # ----------------------------------------------------------------------------
            col_magnitude = random.randint(5, 30)
            geo_magnitude = random.randint(5, 30)

        else:
            # NUM OPS
            # ----------------------------------------------------------------------------
            num_col_ops_up=3
            num_geo_ops_up=3
            num_col_ops_upper_range = (
                num_col_ops_up
                if self.bohb_hyperparameters is None
                else self.bohb_hyperparameters["num_col_ops_up"]
            )
            num_geo_ops_upper_range = (
                num_geo_ops_up
                if self.bohb_hyperparameters is None
                else self.bohb_hyperparameters["num_geo_ops_up"]
            )
            num_col_ops = random.randint(0, num_col_ops_upper_range)
            num_geo_ops = random.randint(0, num_geo_ops_upper_range)

            # SAMPLED OPS
            # ----------------------------------------------------------------------------
            sampled_col_ops = random.sample(col_augment_list_with_type(), k=num_col_ops)
            sampled_geo_ops = random.sample(geo_augment_list_with_type(), k=num_geo_ops)
            sampled_ops = sampled_col_ops + sampled_geo_ops

            # MAGNITUDE
            # ----------------------------------------------------------------------------
            col_magnitude_low = 5
            geo_magnitude_low = 5
            col_magnitude_lower_range = (
                col_magnitude_low
                if self.bohb_hyperparameters is None
                else self.bohb_hyperparameters["col_magnitude_low"]
            )
            col_magnitude = random.randint(
                col_magnitude_lower_range, col_magnitude_lower_range + 15
            )
            geo_magnitude_lower_range = (
                geo_magnitude_low
                if self.bohb_hyperparameters is None
                else self.bohb_hyperparameters["geo_magnitude_low"]
            )
            geo_magnitude = random.randint(
                geo_magnitude_lower_range, geo_magnitude_lower_range + 15
            )

        # SMARTSAMPLING
        # --------------------------------------------------------------------------------
        for op, minval, maxval, type in sampled_ops:
            magnitude = col_magnitude if type == "col" else geo_magnitude
            magnitude_val = (float(magnitude) / 30) * float(maxval - minval) + minval
            data = op(data, magnitude_val, self.is_segmentation)

        return data

