# Cutoff: A Simple Data Augmentation Approach for Natural Language 

This repository contains source code necessary to reproduce the results presented in the following paper:
* [*A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation*](https://arxiv.org/abs/2009.13818)

This project is maintained by [Dinghan Shen](https://sites.google.com/view/dinghanshen). Feel free to contact dishen@microsoft.com for any relevant issues.

## Natural Language Undertanding (e.g. GLUE tasks, etc.) 
### Prerequisite: 
* CUDA, cudnn
* Python 3.7
* PyTorch 1.4.0

### Run
1. Install Huggingface Transformers according to the instructions here: https://github.com/huggingface/transformers.

2. Download the datasets from the GLUE benchmark:
```python
python download_glue_data.py --data_dir glue_data --tasks all
```

3. Fine-tune the RoBERTa-base or RoBERTa-large model with the *Cutoff* data augmentation strategies:
```python
>>> chmod +x run_glue.sh
>>> ./run_glue.sh
```
  Options: different settings and hyperparameters can be selected and specified in the `run_glue.sh` script: 
- `do_aug`: whether augmented examples are used for training.
- `aug_type`: the specific strategy to synthesize *Cutoff* samples, which can be chosen from: *'span_cutoff'*, *'token_cutoff'* and *'dim_cutoff'*.
- `aug_cutoff_ratio`: the ratio corresponding to the span length, token number or number of dimensions to be cut.
- `aug_ce_loss`: the coefficient for the cross-entropy loss over the cutoff examples.
- `aug_js_loss`: the coefficient for the Jensen-Shannon (JS) Divergence consistency loss over the cutoff examples.
- `TASK_NAME`: the downstream GLUE task for fine-tuning.
- `model_name_or_path`: the pre-trained for initialization (both RoBERTa-base or RoBERTa-large models are supported).
- `output_dir`: the folder results being saved to.

## Natural Language Generation (e.g. Translation, etc.)

Please refer to [Neural Machine Translation with Data Augmentation](https://github.com/stevezheng23/fairseq_extension/tree/master/examples/translation/augmentation) for more details

### IWSLT'14 German to English (Transformers)
| Task          | Setting           | Approach   | BLEU     |
|---------------|-------------------|------------|----------|
| iwslt14 de-en | transformer-small | w/o cutoff | 36.2     |
| iwslt14 de-en | transformer-small | w/ cutoff  | **37.6** |

### WMT'14 English to German (Transformers)

| Task          | Setting           | Approach   | BLEU     |
|---------------|-------------------|------------|----------|
| wmt14 en-de   | transformer-base  | w/o cutoff | 28.6     |
| wmt14 en-de   | transformer-base  | w/ cutoff  | **29.1** |
| wmt14 en-de   | transformer-big   | w/o cutoff | 29.5     |
| wmt14 en-de   | transformer-big   | w/ cutoff  | **30.3** |

## Citation 
Please cite our paper in your publications if it helps your research:

```latex
@article{shen2020simple,
  title={A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation},
  author={Shen, Dinghan and Zheng, Mingzhi and Shen, Yelong and Qu, Yanru and Chen, Weizhu},
  journal={arXiv preprint arXiv:2009.13818},
  year={2020}
}
```
