# Machine Translation
Implementation of original Encoder-Decoder transformer architecture for English-German Translation
<p align="center">
  <img src="assets/transformer_architecture.webp" width="300">
</p>

## Introduction
This project aims to utilize the classical Encoder-Decoder architecture to perform machine translation on the [IWSLT14 dataset](https://www.kaggle.com/datasets/qcriwgao/iwslt14).

## Project Structure
```text
Machine Translation/
├── artifacts/        
    ├── vocabs                # Extracted dictionaries
    ├── checkpoints          
        ├── train0            # Training session saving: model weights, training and validation loss
        ├── train1
        ├── ...
├── assets/                   # Visual aids for README
├── datasets/                 # IWSLT14
├── src/
    ├── dataset_processing/   # Dataset loading and vocab extraction
    ├── evaluation/           # Perform BLEU scoring
    ├── training/             # Model training utilities
    ├── transformer/          # Transformer modules and models architecture
    ├── config.py             # Predefined constants for settings of dataset, model and training
├── .gitattributes
├── .gitignore
├── README.md
````

## Model Architecture
Here is the model settings used in this project. The following configuration is contained in config.py.
```python
# Model settings
D_MODEL = 256
N_HEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
```
## Dataset
The **IWSLT14 dataset** contains three splits including **train**, **evaluation** and **test**. Each splits consists of two parallel text files of **Byte Pair Encoding (BPE)** tokenized sentences, aligned in line-by-line manner. Table 1 summarizes the number of samples for each categories.

<div align="center">

**Table 1. Statistics of the IWSLT14 German–English dataset**

| Split | Samples |
|:-----|:--------|
| Train | 160,239 |
| Validation | 7,283 |
| Test | 6,750 |

</div>

