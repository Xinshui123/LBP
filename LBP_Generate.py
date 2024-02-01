from os import listdir
from os.path import isfile, join
import os

import cv2

from tqdm import tqdm, trange

import LBP

Rec_path = "LBP_Reco/"
Img_path = "LBP_Image/"
targetFile = "D:/dataset/cifar-10/datasets/combine"
# imgFile = "../ImaNet/trainingset"

rec_files = [f for f in listdir(targetFile) if isfile(join(targetFile, f))]
# img_files = [f for f in listdir(imgFile) if isfile(join(imgFile, f))]

# len(rec_files)
for n in trange(10):
    source_rec = cv2.imread(join(targetFile, rec_files[n]))
    # source_img = cv2.imread(join(imgFile,img_files[n]))
    # transform_img = LBP.ULBP(source_img)
    transform_Rec = LBP.ULBP(source_rec)
    cv2.imwrite(os.path.join(Rec_path, "b{}".format(rec_files[n])), transform_Rec)
    # cv2.imwrite(os.path.join(Img_path , 'Img.{}.jpg'.format(n)),transform_img)
