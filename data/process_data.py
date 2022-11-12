import os 
import csv
from tqdm import tqdm

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

def overlap_dataset(overlaps):
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

if __name__ == "__main__":
    tuber_classes = process_tuber()
    sketchy_classes = process_sketchy()
    overlaps = tuber_classes.intersection(sketchy_classes)
    overlap_dataset(overlaps)