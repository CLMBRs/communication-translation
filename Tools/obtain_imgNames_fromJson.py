import os
import sys
import json
from tkinter import image_names

dir = sys.argv[1]
file_name = sys.argv[2]
img_file_name = sys.argv[3]

meta_data = []

with open(os.path.join(dir, file_name), 'r') as f:
    for i in f:
        meta_data.append(json.loads(i))

image_names = [i['file_name'] for i in meta_data] 

with open(os.path.join(dir, img_file_name), 'w+') as fout:
    for img in image_names:
        img = img + '\n'
        fout.write(img)