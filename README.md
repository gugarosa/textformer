# Textformer: Transformer-Based Text Operators

[![Latest release](https://img.shields.io/github/release/gugarosa/textformer.svg)](https://github.com/gugarosa/textformer/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/textformer.svg)](https://github.com/gugarosa/textformer/issues)
[![License](https://img.shields.io/github/license/gugarosa/textformer.svg)](https://github.com/gugarosa/textformer/blob/master/LICENSE)

## Welcome to Textformer.

Did you ever want to transform text? Are you tired of re-implementing and defining state-of-the-art architectures? If yes, Textformer is the way-to-go! This package provides a straightforward implementation of sequence-to-sequence and transformer-based architectures, fostering all research related to text generation and translation.

Use Textformer if you need a library or wish to:

* Create your network;
* Design or use pre-loaded state-of-the-art architectures;
* Mix-and-match encoder and decoders to solve your problem;
* Because it is fun to transform text.

Read the docs at [textformer.readthedocs.io](https://textformer.readthedocs.io).

Textformer is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**, call us.

---

## Getting started: 60 seconds with Textformer

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, chose your subpackage, and follow the example. We have high-level examples for most tasks we could think of.

Alternatively, if you wish to learn even more, please take a minute:

Textformer is based on the following structure, and you should pay attention to its tree:

```
- textformer
    - core
        - model
    - datasets
        - generative
        - translation
    - models
        - decoders
            - bi_gru
            - conv
            - gru
            - lstm
            - self_attention
        - encoders
            - bi_gru
            - conv
            - gru
            - lstm
            - self_attention
        - layers
            - attention
            - multi_head_attention
            - position_wide_forward
            - residual_attention
        - att_seq2seq
        - conv_seq2seq
        - joint_seq2seq
        - seq2seq
        - transformer
    - utils
        - constants
        - exception
        - logging
        - visualization
```

### Core

The core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Datasets

Because we need data, right? Datasets are composed of classes and methods that allow preparing data for further transformers.

### Models

Each neural network architecture is defined in this package. From Seq2Seq to Transformers, you can use whatever suits your needs.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use it as you wish than re-implementing the same thing over and over again.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Textformer will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever)!:

```Python
pip install textformer
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
