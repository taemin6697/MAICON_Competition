# import matplotlib.pyplot as plt
import random
import numpy as np
import random, shutil, os
import os
import random
import numpy as np
import random
import csv
from PIL import Image
from PIL import ImageFilter
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


class DataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf = random_color_tf

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [[TF.to_pil_image(img[:,:img.shape[1]//2,]), TF.to_pil_image(img[:,img.shape[1]//2:,])] for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0][0].size != (self.img_size, self.img_size):
                imgs = [[TF.resize(img1, [self.img_size, self.img_size],
                        interpolation=TF.InterpolationMode.BICUBIC),
                         TF.resize(img2, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.BICUBIC)]
                        for [img1, img2] in imgs]
        else:
            self.img_size = imgs[0][0].size[0]

        labels = [[TF.to_pil_image(img[:,:img.shape[1]//2,]), TF.to_pil_image(img[:,img.shape[1]//2:,])] for img in labels]

        if len(labels) != 0:
            if labels[0][0].size != (self.img_size, self.img_size):
                labels = [[
                    TF.resize(img1, [self.img_size, self.img_size], 
                        interpolation=TF.InterpolationMode.NEAREST
                    ),
                    TF.resize(img2, [self.img_size, self.img_size], 
                        interpolation=TF.InterpolationMode.NEAREST)]
                    for [img1, img2] in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [[TF.hflip(img1), TF.hflip(img2)] for [img1, img2] in imgs]
            labels = [[TF.hflip(img1), TF.hflip(img2)] for [img1, img2] in labels]

        if self.with_random_vflip and random.random() > 0.5:

            imgs = [[TF.vflip(img1), TF.vflip(img2)] for [img1, img2] in imgs]
            labels = [[TF.vflip(img1), TF.vflip(img2)] for [img1, img2] in labels]

        # if self.with_random_rot and random.random() > random_base:
        if self.with_random_rot:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]

            imgs = [[TF.rotate(img1, angle), TF.rotate(img2, angle)] for [img1, img2] in imgs]
            labels = [[TF.rotate(img1, angle), TF.rotate(img2, angle)] for [img1, img2] in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0][0], scale=(0.8, 1.2), ratio=(1., 1.))

            imgs = [[TF.resized_crop(img1, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC),
                    TF.resized_crop(img2, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)]
                    for [img1, img2] in imgs]

            labels = [[TF.resized_crop(img1, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST),
                       TF.resized_crop(img2, i, j, h, w,
                                       size=(self.img_size, self.img_size),
                                       interpolation=Image.NEAREST)
                       ]
                      for [img1, img2] in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [[pil_rescale(img1, target_scale, order=3), pil_rescale(img2, target_scale, order=3)] for [img1, img2] in imgs]
            labels = [[pil_rescale(img1, target_scale, order=0), pil_rescale(img2, target_scale, order=0)] for [img1, img2] in labels]
            # crop
            imgsize = imgs[0][0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)

            imgs = [[pil_crop(img1, box, cropsize=self.img_size, default_value=0),
                     pil_crop(img2, box, cropsize=self.img_size, default_value=0)]
                    for [img1, img2] in imgs]
            labels = [[pil_crop(img1, box, cropsize=self.img_size, default_value=255),
                       pil_crop(img2, box, cropsize=self.img_size, default_value=255)]
                      for [img1, img2] in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()

            imgs = [[img1.filter(ImageFilter.GaussianBlur(radius=radius)),
                     img2.filter(ImageFilter.GaussianBlur(radius=radius))]
                    for [img1, img2] in imgs]

        if self.random_color_tf:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            imgs_tf = []

            for [img1, img2] in imgs:
                tf = transforms.ColorJitter(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue)
                imgs_tf.append([tf(img1), tf(img2)])
            imgs = imgs_tf

        if to_tensor:
            imgs = [[TF.to_tensor(img1), TF.to_tensor(img2)] for [img1, img2] in imgs]
            labels = [[torch.from_numpy(np.array(img1, np.uint8)).unsqueeze(dim=0),
                       torch.from_numpy(np.array(img2, np.uint8)).unsqueeze(dim=0)]
                      for [img1, img2] in labels]


        imgs = [torch.cat([img1, img2], dim=2) for [img1, img2] in imgs]
        labels = [torch.cat([img1, img2], dim=2) for [img1, img2] in labels]

        return imgs, labels

def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)

prj_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    dataaug = DataAugmentation(img_size=754,
            with_random_hflip=True,  # Horizontally flip
            with_random_vflip=True,  # Vertically flip
            with_random_rot=True,  # rotation
            with_random_crop=False,  # transforms.RandomResizedCrop
            with_scale_random_crop=False,  # rescale & crop
            with_random_blur=True,  # GaussianBlur
            random_color_tf=True)  # colorjitter

    x_paths = []
    with open(os.path.join(prj_dir, 'class1.csv'), newline='') as f:  # cls 라벨 정보
        reader = csv.reader(f)
        for r in reader:
            x_paths.append(os.path.join(prj_dir, '..', 'data','train','x', r[0]))  # 원본 데이터 경로

    with open(os.path.join(prj_dir, 'class2.csv'), newline='') as f:
        reader = csv.reader(f)
        for r in reader:
            x_paths.append(os.path.join(prj_dir, '..', 'data','train','x', r[0]))

    x_paths = list(set(x_paths))
    print(f"Upsampling data length : {len(x_paths)}")

    save_path = os.path.join(prj_dir, '..', 'data', 'up')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'x'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'y'), exist_ok=True)

    for x in x_paths:
        image = Image.open(x)
        label = Image.open(x.replace('x', 'y'))

        image = np.asarray(np.expand_dims(image, axis=0))
        label = np.asarray(np.expand_dims(label, axis=0)).transpose(1, 2, 0)
        label = np.asarray(np.expand_dims(label, axis=0))

        # Augment an image
        transform = T.ToPILImage()
        trans_imgs, trans_labels = dataaug.transform(image, label)
        x_img = transform(trans_imgs[0])
        y_img = transform(trans_labels[0])

        x_img.save(os.path.join(save_path, 'x', os.path.basename(x)))
        y_img.save(os.path.join(save_path, 'y', os.path.basename(x.replace('x', 'y'))))
