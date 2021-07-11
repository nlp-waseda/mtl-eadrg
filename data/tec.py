import os
import re
import sys

emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

mention = re.compile(r'^@\w+\s+', re.ASCII)
hashtag = re.compile(r'\s+#\w+$', re.ASCII)

in_path = sys.argv[1]
out_path = os.path.join(sys.argv[2], '{}.{}')

pairs = []
with open(in_path) as f:
    for line in f:
        text, emo = line.split('\t', maxsplit=1)[1].rsplit('\t', maxsplit=1)

        # clean text
        while re.search(mention, text):
            text = re.sub(mention, '', text)
        while re.search(hashtag, text):
            text = re.sub(hashtag, '', text)

        pairs.append((text, emotions.index(emo[3:].rstrip())))

val_size = len(pairs) // 10
train_size = len(pairs) - 2 * val_size

list_of_pairs = [
    pairs[:train_size],
    pairs[train_size:train_size+val_size],
    pairs[train_size+val_size:]
]

for i, split in enumerate(['train', 'val', 'test']):
    for j, lang in enumerate(['source', 'target']):
        with open(out_path.format(split, lang), 'w') as f:
            for k in range(len(list_of_pairs[i])):
                f.write(str(list_of_pairs[i][k][j]) + '\n')
