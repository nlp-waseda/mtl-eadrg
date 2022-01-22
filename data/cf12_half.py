import csv
import os
import random
import shutil
import sys

emotions = [
    'anger',
    'boredom',
    # 'empty',
    'enthusiasm',
    'fun',
    'happiness',
    'hate',
    'love',
    'neutral',
    'relief',
    'sadness',
    'surprise',
    'worry'
]

input_path, output_dir = sys.argv[1:]

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

pairs = []
with open(input_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['sentiment'] != 'empty':
            label = emotions.index(row['sentiment'])
            pairs.append((row['content'], label))

random.seed(0)
random.shuffle(pairs)

pairs = pairs[:len(pairs)//2]

val_size = len(pairs) // 10
train_size = len(pairs) - 2 * val_size

list_of_pairs = [
    pairs[:train_size],
    pairs[train_size:train_size+val_size],
    pairs[train_size+val_size:]
]

for i, split in enumerate(['train', 'val', 'test']):
    for j, lang in enumerate(['source', 'target']):
        output_path = os.path.join(output_dir, split + '.' + lang)
        with open(output_path, 'w') as f:
            for k in range(len(list_of_pairs[i])):
                f.write(str(list_of_pairs[i][k][j]) + '\n')
