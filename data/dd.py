import os
import shutil
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

for split in ['train', 'validation', 'test']:
    path = os.path.join(in_dir, split, 'dialogues_' + split + '.txt')

    pairs = []
    with open(path) as f:
        for line in f:
            texts = line.split('__eou__')[:-1]

            for i in range(len(texts) - 1):
                pairs.append([texts[i].strip(), texts[i+1].strip()])
    
    if split == 'validation':
        split = 'val'

    for i, lang in enumerate(['source', 'target']):
        out_path = os.path.join(out_dir, split + '.' + lang)
        with open(out_path, 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
