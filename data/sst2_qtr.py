import os
import shutil
import sys

if os.path.isdir(sys.argv[2]):
    shutil.rmtree(sys.argv[2])
os.mkdir(sys.argv[2])

in_path = os.path.join(sys.argv[1], '{}.tsv')
out_path = os.path.join(sys.argv[2], '{}.{}')

with open(in_path.format('test')) as f:
    test_size = len(f.readlines())

pairs_dict = {}

for split in ['train', 'dev']:
    pairs = []

    with open(in_path.format(split)) as f:
        header = next(f)
        for line in f:
            pairs.append(line.rstrip().split('\t'))
    
    if split == 'train':
        # pairs_dict['train'] = pairs[:-test_size]
        pairs_dict['train'] = pairs[:len(pairs)//4]
        pairs_dict['test'] = pairs[-test_size:]
    else:
        pairs_dict['val'] = pairs

for split, pairs in pairs_dict.items():
    for i, lang in enumerate(['source', 'target']):
        with open(out_path.format(split, lang), 'w') as f:
            for j in range(len(pairs)):
                f.write(pairs[j][i] + '\n')
