# Introduction

Voice conversion is a technique that converts the perceived identity of a speaker's utterance (source) into another speaker's utterance (target). In this project, inspired from the paper titled "Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder" by Chin-Cheng Hsu, our team would focus on using **spectral conversion (SC) techniques**.

Our project is divided into 2 seperate portions:
                    1. **Voice treatment**
                    2. **Training of VAE model**


# Prerequisites

- **numpy** 👉 the basic scientific library of Python with built-in math functions and easy array handling
- **librosa** 👉 package used for music and audio analysis
- **pyworld** 👉 open sourced software for high quality speech analysis, manipulation and systhesis
- **tensorflow** 👉 machine learning package that can train/run deep neural networks (NN)
- **matplotlib** 👉  for graph plotting

# Data Provided

We are using the utterances provided from the VCC2016 dataset. This dataset consists of both a training set (150 utterances), validation set (12 utterances) and a testing set (54 utterances). Source speakers include 3 females and 2 males while target speakers includes 2 females and 3 males.

# Voice Treatment
## Parameters used
```
EPSILON = 1e-10
FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1 
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1  # [sp, ap, f0, en, s] 
RECORD_BYTES = FEAT_DIM * 4 
```
