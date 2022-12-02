from glob import glob
import os
from PIL import Image
import numpy as np
import csv


# 클래스 별로 데이터 경로 csv 파일로 저장
prj_dir = os.path.dirname(os.path.abspath(__file__))
train_dirs = os.path.join(prj_dir, '..', 'data', 'train')
train_img_paths = glob(os.path.join(train_dirs, 'y', '*.png'))
# train_img_paths = list(map(lambda x:x.split("/")[-1], train_img_paths))

file_dir = os.path.join(prj_dir, '..', 'preprocess')
f1 = open(os.path.join(file_dir, 'class1.csv'),'w', newline='\n')
f2 = open(os.path.join(file_dir, 'class2.csv'),'w', newline='\n')

wr1 = csv.writer(f1)
wr2 = csv.writer(f2)

wr = ['', wr1, wr2]

for i, img_path in enumerate(train_img_paths):

    img = Image.open(img_path)
    img = np.array(img)

    for j in range(1, 3):
        if j in img:
            wr[j].writerow([os.path.basename(img_path)])
            continue

    if i % 100 == 0:
        print(i)

f1.close()
f2.close()
