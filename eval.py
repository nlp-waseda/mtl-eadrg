import csv
import os
import sys
from tqdm.auto import tqdm

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams


def corpus_dist(corpus, n):
    '''https://www.aclweb.org/anthology/N16-1014/'''
    ngrams_set = set()
    num_tokens = 0

    for tokens in corpus:
        num_tokens += len(tokens)
        ngrams_set |= set(ngrams(tokens, n))
            
    return len(ngrams_set) / num_tokens


data_dir = sys.argv[1]
output_dirs = sys.argv[2:]

list_of_references = []
with open(os.path.join(data_dir, 'dd', 'test.target')) as f:
    for line in f:
        references = [word_tokenize(line)]
        list_of_references.append(references)

rows = [['model', 'bleu', 'dist-1', 'dist-2', 'avg len']]

for output_dir in tqdm(output_dirs):
    model = os.path.basename(os.path.dirname(output_dir))
    row = [model]

    hypotheses = []
    with open(os.path.join(output_dir, 'pred.txt')) as f:
        for line in f:
            hypothesis = word_tokenize(line)
            hypotheses.append(hypothesis)
    
    # BLEU
    bleu = corpus_bleu(list_of_references, hypotheses)
    row.append(bleu)

    # distinct
    for n in [1, 2]:
        dist = corpus_dist(hypotheses, n)
        row.append(dist)

    # average length
    avg_len = sum(len(tokens) for tokens in hypotheses) / len(hypotheses)
    row.append(avg_len)

    rows.append(row)

with open('eval.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)
