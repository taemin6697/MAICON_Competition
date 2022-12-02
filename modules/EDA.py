import cv2
from glob import glob
import os
from PIL import Image
import numpy as np
from utils import load_yaml
import matplotlib.pyplot as plt

prj_dir = os.path.dirname(os.path.abspath(__file__))
train_dirs = os.path.join(prj_dir, '..', 'data', 'train')
train_img_paths = glob(os.path.join(train_dirs, 'y', '*.png'))


config_path = os.path.join(prj_dir, '..', 'config', 'train.yaml')
config = load_yaml(config_path)

class_info = {i: 0 for i in range(4)}

for i, img_path in enumerate(train_img_paths):

    img = Image.open(img_path)
    img = np.array(img)

    for j in range(4):
        if j in img:
            class_info[j] += 1
            continue
    if i % 100 == 0:
        print(i, class_info)

x = class_info.keys()
y = class_info.values()


plt.bar(x, y)
plt.xlabel('class')
plt.ylabel('count')
plt.show()

print(class_info)