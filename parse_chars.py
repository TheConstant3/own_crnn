import json
import os

path = '/home/kostya/VideoAnalyticProjects/NomerDet/eu_dataset/val/ann'

all_labels = set()
for filename in os.listdir(path):
    with open(f'{path}/{filename}') as f:
        label = json.load(f)['description']
        all_labels.update(set(label))

all_labels = list(all_labels)
all_labels.sort()
print(all_labels)
print(len(all_labels))
