import os
import sys
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from transformers import BartTokenizer

from multitask_bart import BartForMultitaskLearning
from dataset import MultitaskDataset

data_dir, output_dir = sys.argv[1:]

model = BartForMultitaskLearning.from_pretrained(output_dir).cuda()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

dataset = MultitaskDataset(['response'], tokenizer, data_dir, 'test', 64)
loader = DataLoader(dataset, batch_size=32)

model.eval()
outputs = []

for batch in tqdm(loader):
    outs = model.generate(
        input_ids=batch['source_ids'].cuda(),
        attention_mask=batch['source_mask'].cuda(), 
        max_length=256,
        num_beams=5,
        no_repeat_ngram_size=3,
        early_stopping=True,
        task=batch['task'][0]
    )

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    outputs.extend(dec)

with open(os.path.join(output_dir, 'pred.txt'), 'w') as f:
    for output in outputs:
        f.write(output + '\n')
        