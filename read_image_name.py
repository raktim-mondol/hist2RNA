
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt




file_name=[]
path='./tcga-gc/'
for image_file in next(os.walk(path))[1]:
    new_name=image_file.split('-')[0]+'-'+image_file.split('-')[1]+'-'+image_file.split('-')[2]+'-'+'01'
    file_name.append(new_name)
    
    
   