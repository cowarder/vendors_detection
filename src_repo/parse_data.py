import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil as sh
import xml.etree.cElementTree as ET

data_path = "/home/data/51"

img_file = []
xml_file = []

# build log, models folder
if not os.path.exists('/project/train/log'):
    os.makedirs('/project/train/log')

if not os.path.exists('/project/train/models'):
    os.makedirs('/project/train/models')

for item in os.listdir(data_path):
    if item.endswith(".jpg"):
        img_file.append(item)
    else:
        xml_file.append(item)

print(len(img_file), len(xml_file))

class_name = ['vendors']
cls2num = {'vendors': 0,}


res = []
for file in xml_file:
    path = data_path + "/" + file
    tree = ET.parse(path)
    objects = []
    size = tree.find('size')
    img_h = size.find('height').text
    img_w = size.find('width').text
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = file
        obj_struct['class'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
        obj_struct['img_h'] = img_h
        obj_struct['img_w'] = img_w
        #obj_struct['class'] = bbox.find('class')
        objects.append(obj_struct)
    res.append(objects)

df = pd.DataFrame(columns=['image_id', 'x', 'y', 'w', 'h', 'img_w', 'img_h', 'classes'])
img_ids =[]
x, y, w, h =[], [], [], []
img_w, img_h = [], []
classes = []

for img in res:
    for obj in img:
        if obj['class'] not in cls2num.keys():
            print(obj['class'])
            continue
        img_ids.append(obj['name'][:-4])
        x.append(float(obj['bbox'][0]))
        y.append(float(obj['bbox'][1]))
        w.append(float(obj['bbox'][2]) - float(obj['bbox'][0]))
        h.append(float(obj['bbox'][3]) - float(obj['bbox'][1]))
        img_w.append(float(obj['img_w']))
        img_h.append(float(obj['img_h']))
        classes.append(cls2num[obj['class']])

df['image_id'] = img_ids
df['x'] = x
df['y'] = y
df['w'] = w
df['h'] = h
df['img_w'] = img_w
df['img_h'] = img_h
df['classes'] = classes


df['x_center'] = df['x'] + df['w']/2
df['y_center'] = df['y'] + df['h']/2

index = list(set(df.image_id))

if True:
    for fold in [0]:
        val_index = index[len(index)*fold//5:len(index)*(fold+1)//5]
        for name,mini in tqdm(df.groupby('image_id')):
            if name in val_index:
                path2save = 'val2017/'
            else:
                path2save = 'train2017/'
            if not os.path.exists('/home/data/convertor/fold{}/labels/'.format(fold)+path2save):
                os.makedirs('/home/data/convertor/fold{}/labels/'.format(fold)+path2save)
            with open('/home/data/convertor/fold{}/labels/'.format(fold)+path2save+name+".txt", 'w+') as f:
                row = mini[['classes','x_center','y_center','w','h', 'img_w', 'img_h']].astype(float).values
                wh = mini[['img_w', 'img_h']]
                for i in range(len(row)):
                    row[i][0] = int(row[i][0])
                    row[i][1] = row[i][1]/row[i][5]
                    row[i][2] = row[i][2]/row[i][6]
                    row[i][3] = row[i][3]/row[i][5]
                    row[i][4] = row[i][4]/row[i][6]
                    #row[i][1] = min(0.9999, row[i][1])
                    #row[i][1] = max(0.0001, row[i][1])
                    #row[i][2] = min(0.9999, row[i][2])
                    #row[i][2] = max(0.0001, row[i][2])
                    #row[i][3] = min(0.9999, row[i][3])
                    #row[i][3] = max(0.0001, row[i][3])
                    #row[i][4] = min(0.9999, row[i][4])
                    #row[i][4] = max(0.0001, row[i][4])
                    
                    
                #row[0] = mini['classes']
                # row = row.astype(str)
                for j in range(len(row)):
                    
                    if row[j][1] <= 0 or row[j][1]>= 1:
                        continue
                    if row[j][2] <= 0 or row[j][2]>= 1:
                        continue
                    if row[j][3] <= 0 or row[j][3]>= 1:
                        continue
                    if row[j][4] <= 0 or row[j][4]>= 1:
                        continue
                    
                    text = ' '.join(row[j].astype(str)[:5])
                    f.write(text)
                    f.write("\n")
            if not os.path.exists('/home/data/convertor/fold{}/images/{}'.format(fold,path2save)):
                os.makedirs('/home/data/convertor/fold{}/images/{}'.format(fold,path2save))
            sh.copy("{}/{}.jpg".format(data_path,name),'/home/data/convertor/fold{}/images/{}/{}.jpg'.format(fold,path2save,name))