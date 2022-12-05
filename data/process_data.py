import os 
import csv
from tqdm import tqdm
import random
import pandas as pd

# process TU Berlin dataset
def process_tuber():
    path = "/edward-slow-vol/Sketch2Model/png/"
    sketches = {}
    classes = set()

    header = ['path', 'class']
    with open('tuberlin.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
            sub = dirpath.split("/")[-1]
            for item in dirnames:
                classes.add(item)
            if sub:
                for name in filenames:
                    writer.writerow([dirpath + "/" + name, sub])
    return classes

def process_sketchy():
    path = "/edward-slow-vol/Sketch2Model/256x256/photo/"
    photo = {}
    classes = set() 
    header = ['path', 'class']
    with open('sketchy_photo.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
            sub = dirpath.split("/")[-1]
            for item in dirnames:
                classes.add(item)
            if sub:
                for name in filenames:
                    writer.writerow([dirpath + "/" + name, sub])
    path = "/edward-slow-vol/Sketch2Model/256x256/sketch/"
    sketch = {}
    classes = set() 
    header = ['path', 'class']
    with open('sketchy_sketch.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
            sub = dirpath.split("/")[-1]
            for item in dirnames:
                classes.add(item)
            if sub:
                for name in filenames:
                    writer.writerow([dirpath + "/" + name, sub])
    return classes 

def overlap_dataset(overlaps, count = 0):
    if count>0 and count < len(overlaps):
        temp = set()
        for i, val in enumerate(random.sample(overlaps, count)):
            temp.add(val)
        overlaps = temp

    with open('sketchy_photo.csv') as f_open:
        csv_reader = csv.reader(f_open)
        with open('overlap_photo.csv', 'w') as f:
            line_count = 0
            writer = csv.writer(f)
            for row in tqdm(csv_reader):
                if line_count == 0:
                    writer.writerow(row)
                    line_count += 1
                else:
                    for over in overlaps:
                        if over in row:
                            writer.writerow(row)
                            break
                    line_count += 1
    dataset = set()   
    with open('sketchy_sketch.csv') as f_open:
        csv_reader = csv.reader(f_open)
        line_count = 0
        for row in tqdm(csv_reader):
            if line_count == 0:
                line_count += 1
            else:
                for over in overlaps:
                    if over in row:
                        dataset.add(tuple(row))
                        break
                line_count += 1
    with open('tuberlin.csv') as f_open:
        csv_reader = csv.reader(f_open)
        line_count = 0
        for row in tqdm(csv_reader):
            if line_count == 0:
                line_count += 1
            else:
                for over in overlaps:
                    if over in row:
                        dataset.add(tuple(row))
                        break
                line_count += 1
    header = ['path', 'class']
    with open('overlap_sketch.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in dataset:
            writer.writerow(list(item))

def filter_imagenet(translations):
    path = "/edward-slow-vol/Sketch2Model"
    items = set()
    for key, value in translations.items():
        for v in value:
            items.add(v)
    for filename in tqdm(os.listdir(path)):
        if ".tar" not in filename:
            continue
        delete = True
        for item in items:
            if item in filename:
                delete=False
                break
        if delete:
            if os.path.exists(filename):
                os.remove(filename)
                print("removed", filename)
            else:
                print("cant remove", filename)


def aggregate_imagenet(translations):
    path = "/edward-slow-vol/Sketch2Model/ImageSubNet"
    header = ['path', 'class']
    with open('chair.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for filename in tqdm(os.listdir(path)):
            for key, value in translations.items():
                for v in value:
                    if v in filename:
                        writer.writerow(["/edward-slow-vol/Sketch2Model/ImageSubNet/" + filename, key])

def aggregate_sketch(translations):
    sketch = {}
    header = ['path', 'class']
    with open('chair_sketch.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        path = "/edward-slow-vol/Sketch2Model/256x256/sketch/"
        for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
            sub = dirpath.split("/")[-1]
            if sub and sub in translations:
                for name in filenames:
                    writer.writerow([dirpath + "/" + name, sub])
        path = "/edward-slow-vol/Sketch2Model/png/"
        for (dirpath, dirnames, filenames) in tqdm(os.walk(path)):
            sub = dirpath.split("/")[-1]
            if sub and sub in translations:
                for name in filenames:
                    writer.writerow([dirpath + "/" + name, sub])


def combine_csv(csvs):
    combined_csv = pd.concat([pd.read_csv(f) for f in csvs ])
    combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    # tuber_classes = process_tuber()
    # sketchy_classes = process_sketchy()
    # overlaps = tuber_classes.intersection(sketchy_classes)
    choice = {'bench', 'chair', 'airplane', 'table', 'mushroom', 'fish', 'pineapple',
                'banana', 'cat', 'tank'}

    translations= {'bench': ['n03891251'],
        'chair': ['n03376595', 'n04099969'],
        'airplane': ['n04552348', 'n02690373'],
        'table': ['n03201208'],
        'mushroom': ['n07734744'],
        'fish': ['n01443537', 'n02607072'],
        'pineapple': ['n07753275'],
        'banana': ['n07753592'],
        'cat': ['n02123394', 'n02124075', 'n02123597', 'n02123045'],
        'tank': ['n04389033']
    }

    translations = {'chair'}
    # overlap_dataset(choice)
    # aggregate_imagenet(translations)
    aggregate_sketch(translations)
    # combine_csv(['overlap_photo_imagenet.csv', 'overlap_photo.csv'])