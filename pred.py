import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from multitask_bart import BartForMultitaskLearning
from dataset import MultitaskDataset


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    model = BartForMultitaskLearning.from_pretrained(args.output_dir).to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    dataset = MultitaskDataset(['response'], tokenizer, args.data_dir, 'test', 64)
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    outputs = []

    for batch in tqdm(loader):
        outs = model.generate(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device), 
            max_length=256,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            task=batch['task'][0]
        )

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        outputs.extend(dec)

    with open(os.path.join(args.output_dir, 'pred.txt'), 'w') as f:
        for output in outputs:
            f.write(output + '\n')


if __name__ == '__main__':
    main()
