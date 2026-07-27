# nanoGPT Pantun Experiment

This repository is a fork of Andrej Karpathy's `ng-video-lecture` repository. The original project demonstrates a small character-level GPT model trained on Tiny Shakespeare. This fork adapts the same simple GPT training code to learn from an Indonesian pantun dataset.

## Project Overview

The goal of this experiment is to train a small character-level Transformer language model on pantun text.

The model learns to predict the next character given a sequence of previous characters. For example, given a text sequence, the model is trained to predict each next character one step at a time.

This experiment keeps the original educational structure of Karpathy's `gpt.py`, but replaces the original Tiny Shakespeare dataset with a cleaned pantun text file.

## Dataset

The pantun dataset used in this experiment is derived from the public `ir-nlp-csui/sampiran` repository:

https://github.com/ir-nlp-csui/sampiran

The original dataset contains pantun examples in labelled sequence formats. For this experiment, the labelled files were converted into a plain text format similar to Karpathy's Tiny Shakespeare `input.txt`.

The cleaning process removes special sequence labels such as:

```text
<BOS>
<EOS>
<CLS>
<CONTENT>
<DELIVERANCE>
</DELIVERANCE>
</CONTENT>
<PREF>
<SYL:...>
<RHYME:...>
```

Each pantun is then formatted as four plain-text lines separated by a blank line.

Example format:

```text
karena terlalu lama berdiri
para tamu pada ngeliatin
selamat hari raya idul fitri
mohon maaf lahir dan batin

tanaman diserang hama
kata adik sepupu ibuku
waktu jadi terasa sangat lama
bila dirimu tak ada disampingku
```

The cleaned text file is used as the language modelling corpus.

## Files

```text
gpt.py              Main character-level GPT training script
input_pantun.txt    Cleaned pantun dataset
more.txt            Optional generated text output
README.md           Project description
```

If `gpt.py` expects the dataset file to be called `input.txt`, copy or rename the pantun file:

```bash
cp input_pantun.txt input.txt
```

## Training Setup

The model follows Karpathy's original character-level language modelling setup.

The dataset is treated as one long sequence of characters. It is then split into training and validation portions:

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

This means:

```text
First 90% of characters  -> training data
Last 10% of characters   -> validation data
```

The model is trained on random chunks from the training data and evaluated on random chunks from the validation data.

## Important Hyperparameters

Some important hyperparameters in `gpt.py` are:

```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

### `block_size`

`block_size` controls how many previous characters the model can use as context.

For example:

```python
block_size = 128
```

means the model can look at up to 128 previous characters when predicting the next character.

A larger `block_size` gives the model a longer memory, which is useful for learning pantun structure.

### `batch_size`

`batch_size` controls how many random text chunks are processed at the same time during training.

For example:

```python
batch_size = 32
block_size = 128
```

means each training step uses 32 chunks, and each chunk contains 128 characters.

## Training Environment

The model was trained using a Kaggle notebook with GPU acceleration enabled.

Kaggle provided the following GPU configuration:

```text
GPU: NVIDIA Tesla T4 x 2
```

Although two T4 GPUs were available in the Kaggle environment, this experiment used only one GPU. The original `gpt.py` script selects a single CUDA device using:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

This means the model runs on one available GPU by default and does not use distributed or multi-GPU training.

During training, only one T4 GPU was actively used, while the second GPU remained mostly idle. This is expected because the training script is a simple single-process PyTorch implementation.

A multi-GPU setup would require additional changes, such as using PyTorch Distributed Data Parallel, but this was not necessary for this small character-level pantun experiment.


## Running the Model

Install the required Python packages:

```bash
pip install torch
```

Run training:

```bash
python gpt.py
```

During training, the script prints the training and validation loss:

```text
step 0: train loss 3.9353, val loss 3.9340
step 500: train loss 1.7887, val loss 1.8424
step 1000: train loss 1.3187, val loss 1.4043
```

The training loss measures how well the model predicts characters from the training data.

The validation loss measures how well the model predicts characters from unseen validation text.

A good run should show both training and validation loss decreasing. If the training loss keeps decreasing but validation loss starts increasing, the model is beginning to overfit.

## Text Generation

At the end of training, `gpt.py` generates text from the trained model.

The line:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

prints 500 newly generated characters.

The optional line:

```python
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
```

generates 10,000 new characters and saves them to:

```text
more.txt
```

For this pantun experiment, `more.txt` contains generated pantun-like text.

## Example Result

A successful training run may look like this:

```text
step 0: train loss 3.9353, val loss 3.9340
step 500: train loss 1.7887, val loss 1.8424
step 1000: train loss 1.3187, val loss 1.4043
step 1500: train loss 1.1363, val loss 1.2697
step 2000: train loss 1.0193, val loss 1.2134
step 2500: train loss 0.9138, val loss 1.1811
step 3000: train loss 0.8203, val loss 1.1750
step 3500: train loss 0.7228, val loss 1.1728
step 4000: train loss 0.6444, val loss 1.1919
step 4500: train loss 0.5651, val loss 1.2049
```

In this example, the validation loss improves until around step 3500, then starts to increase. This suggests that the model learns useful pantun character patterns up to around step 3500, after which mild overfitting begins.

## Notes

This is a small educational experiment rather than a production-level language model.

The model is trained at character level, not word level or subword level. This means every character is treated as a token. The model learns spelling, line breaks, rhyme-like patterns, and local text structure directly from characters.

The purpose of this project is to understand how a small GPT-style model learns from a custom text corpus.
