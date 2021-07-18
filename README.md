# Emotion-Aware Dialogue Response Generation by Multi-Task Learning

Codes for the paper of [Multi-Task Learning of Generation and Classification for Emotion-Aware Dialogue Response Generation](https://aclanthology.org/2021.naacl-srw.15/) at NAACL SRW 2021.

## Usage

### Preprocess

    python data/dd.py path/to/ijcnlp_dailydialog/ data/dd/
    
    python data/tec.py path/to/Jan9-2012-tweets-clean.txt data/tec/
    python data/sst2_qtr.py path/to/SST-2/ data/sst2_qtr/
    python data/cf12_half.py path/to/text_emotion.csv data/cf12_half/

### Training

    python train.py \
        $DATA_DIR \
        $OUTPUT_DIR \
        --max_seq_length 64 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --warmup_steps 500 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --num_train_epochs 64 \
        --gradient_accumulation_steps 4 \
        --max_grad_norm 0.1 \
        --tasks cfemotion,emotion,response,sentiment \
        --loss_weights 0.3333,0.3333,1.0,0.3333


### Prediction

    python pred.py $DATA_DIR $OUTPUT_DIR

### Evaluation

    python eval.py $DATA_DIR $OUTPUT_DIR1 $OUTPUT_DIR2 ...

## Datasets

| Dataset | Task | # Labels | URL |
| --- | --- | --- | --- |
| DailyDialog | Response Generation | - | [Yanran's Attic](http://yanran.li/dailydialog) |
| Twitter Emotion Corpus | Emotion Recognition | 6 | [Saif \| Emotion and Sentiment Data](http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html) |
| SST-2 | Coarse-Grained Emotion Recognition | 2 | [GLUE Benchmark](https://gluebenchmark.com/tasks) |
| CrowdFlower | Fine-Grained Emotion Recognition | 12 | [Sentiment Analysis in Text - dataset by crowdflower \| data.world](https://data.world/crowdflower/sentiment-analysis-in-text) |

The datasets are selected from https://github.com/sarnthil/unify-emotion-datasets.

## References

- [Bart — transformers 3.0.2 documentation](https://huggingface.co/transformers/v3.0.2/model_doc/bart.html)
- [exploring-T5/t5_fine_tuning.ipynb at master · patil-suraj/exploring-T5](https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb)
- [fairseq/README.summarization.md at master · pytorch/fairseq](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md)
- [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
- [Adversarial Multi-task Learning for Text Classification - ACL Anthology](https://www.aclweb.org/anthology/P17-1001/)
- [DCGAN Tutorial — PyTorch Tutorials 1.6.0 documentation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## Citation

    @inproceedings{ide-kawahara-2021-multi,
        title = "Multi-Task Learning of Generation and Classification for Emotion-Aware Dialogue Response Generation",
        author = "Ide, Tatsuya  and
        Kawahara, Daisuke",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Student Research Workshop",
        month = jun,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.naacl-srw.15",
        doi = "10.18653/v1/2021.naacl-srw.15",
        pages = "119--125",
        abstract = "For a computer to naturally interact with a human, it needs to be human-like. In this paper, we propose a neural response generation model with multi-task learning of generation and classification, focusing on emotion. Our model based on BART (Lewis et al., 2020), a pre-trained transformer encoder-decoder model, is trained to generate responses and recognize emotions simultaneously. Furthermore, we weight the losses for the tasks to control the update of parameters. Automatic evaluations and crowdsourced manual evaluations show that the proposed model makes generated responses more emotionally aware.",
    }
