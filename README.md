# Introduction

Voice conversion is a technique that converts the perceived identity of a speaker's utterance (source) into another speaker's utterance (target). In this project, inspired from the paper titled "Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder" by Chin-Cheng Hsu, our team would focus on using **spectral conversion (SC) techniques**.

Our project is divided into 2 seperate portions:
                    1. **Voice treatment**
                    2. **Training of VAE model**

The aim of this git, is to allow users to have a comprehensive view towards how VAE can be used not only to treat MNIST data, but also phonetic content from voice files as well. Hence, **please refer to the sideComparisonVAE.ipynb** to have an overview of a side-by-side comparison on how we approach both MNIST data as well as phonetic content (spectral envelope) of voice files.

# Architecture

Below is the architecture of our project:
- 📁 modules
  - 📑 voiceVaeModel.py
  - 📑 voicetreatment.py

- 📑 README.md
- 📑 mnistVAE.ipynb
- 📑 sideComparisonVAE.ipynb
- 📑 voiceTreatment.ipynb


# Prerequisites

- **numpy** 👉 the basic scientific library of Python with built-in math functions and easy array handling
- **librosa** 👉 package used for music and audio analysis
- **pyworld** 👉 open sourced software for high quality speech analysis, manipulation and systhesis
- **tensorflow** 👉 machine learning package that can train/run deep neural networks (NN)
- **matplotlib** 👉  for graph plotting
- **pandas** 👉 to manipulate dataframes, a Python object that comes in handy when we manipulate large datasets

# Data Provided

We are using the utterances provided from the VCC2016 dataset. This dataset consists of both a training set (150 utterances), validation set (12 utterances) and a testing set (54 utterances). Source speakers include 3 females and 2 males while target speakers includes 2 females and 3 males.

# Voice Treatment
## Parameters used

| epsilon | FFT size | Spectral envelope dimension | features dimension   | f0 ceiling |
|:-------:|:--------:|-----------------------------|----------------------|------------|
|    2    |   1024   |             513             | 513+513+1+1+1 = 1029 | 500 Hz     |

The selected FFT size (defined = 1024) would determine the resolution of the resulting spectra. The number of spectral lines is **half** the FFT size. Hence, the spectral envelope has 512 spectral lines. The resolution of each spectral lines is = sampling rate/FFT_size = 16000//1024 = approximately 15 Hz. 
Larger FFT size would hence provide higher resolution, but would take a longer time to compute.

## wav2pw(...) function
```
def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    _f0, timeaxis = pw.dio(x, fs, f0_ceil = f0_ceil) # _f0 = Raw pitch
    f0 = pw.stonemask(x, _f0, timeaxis, fs)  # f0 = Refined pitch
    sp = pw.cheaptrick(x, f0, timeaxis, fs, fft_size=fft_size) # sp = spectogram spectral smoothing
    ap = pw.d4c(x, f0, timeaxis, fs, fft_size=fft_size) # extract aperiodicity
    return {
      'f0': f0,
      'sp': sp,
      'ap': ap,
    }
    
```
The pyworld package and its subpackages (dio, stonemask, cheaptrick and d4c) would return the spectral envelope (SP), aperiodocity (AP),  fundamental frequency (f0) of the specified {}.wav file. The f0 ceiling is set such that we only filter the lower frequencies.  There would be 513 instances of SP and 513 instances of AP.

## analysis(...) function
```
def analysis(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction ''' 
    fs = 16000
    x, _ = librosa.load(filename, sr=fs, mono=True, dtype=np.float64) #audio time series, sampling rate
    features = wav2pw(x, fs=16000, fft_size=fft_size)
    ap = features['ap']
    f0 = features['f0'].reshape([-1, 1]) #rows = unknown, columns = 1
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True) # Normalizing Factor
    sp_r = np.log10(sp / en) # Refined Spectrogram Normalization
    target = np.concatenate([sp_r, ap, f0, en], axis=1).astype(dtype)
    return target 
```
In this function, we use the *librosa* package to acquire the amplitude and sampling rate of the audio file. The amplitude is then being used by the *wav2pw()* module. We add in a normalizing factor (summation of SP + epsilon), and refined the spectrogram of the spectral envelope on a logarithm scale.

Each single audio file would have a varying length of fundamental frequency depending on the length of the audio. For example: For SF1/100001.wav, we have 704 instances of f0. While for SF1/100002.wav, we have 216 instances of f0. This is because the audio duration for SF1/100001.wav is longer than SF1/100001.wav. However, the featured dimensions remain the same: For each instance of f0, we still have 513 instances of SP and 513 instances of AP.

For example, in SF1/100002.wav, for a f0 = 212.431344 Hz, the spectral envelope is as such:
![alt text](https://github.com/lukecyb8687/Voice-Conversion-VAE/blob/master/raw/SPexample.png)
## extract_and_save_bin_to(...) function

Due to the large amount of data to process, and to allow our VAE model to run faster, we converted the featured dimensions of each {}.wav file into a {}.bin binary file.

# VAE Model
The proposed method in the reference paper inspires the reference of the anagolous work of generating hand-written digits (MNIST dataset). This reference work attempts to extract writing style and digit identity from a vast image of handwriting and to re-synthesis such image. Drawing parallels, the 2 casual latent variables are **identity** and **variation**:

|                    | Variation          | Identity        |
|--------------------|--------------------|-----------------|
| Hand-written digit | Hand-writing style | Nominal number  |
| Speech Frame       | Phonetic content   | Speaking source |

The architecture of the encoder, decoder and VAE model can be doung in the *sideComparisonVAE.ipynb* file. 

## Training Procedure

- Training a VAE involves sampling from a distribution of a latent variable *z*. Therefore, we introduce a re-parametrization trick to introduce stochasticity into this latent variable. 
- Training a VAE is pointwise. Both the *souce* and *target* spectral frames will not be segregated as input and output. Both would be viewed as an input.
- The input into the encoder is the concantenation of both the spectral frames and the speaker identity. The encoder recieves frames from both the source and the target, hence it has the ability of speaker-independent encoding.
- VAE parameters:

| Number of hidden layers | Number of nodes per hidden layer | Latent space size | Size of mini-batch |
|:-----------------------:|:--------------------------------:|-------------------|--------------------|
|            2            |                512               |         128        |         128        |

## Inference and learning
We aim to conduct Maximum likelihood learning, by maximizing the log likelihood of data in our model: **max** log *p*<sub>theta</sub> (*x*), where *p*<sub>θ</sub> (*x*) is known as the marginal likelihood of observation *x*, and θ is the model parameter.

Computing this marginal likelihood of observation *x* is difficult as the joint likelihood model is given by: *p*<sub>θ</sub> (*x,z*) = *p*<sub>θ</sub> (*x|z*) p(*z*), where the term of the left is known as the latent representation.

Since directly optimizing log *p*<sub>θ</sub> (*x*) is infeasible, we will choose to optimize a lower bound of it (by splitting it into a reconstruction loss term and a KL-divergence loss term). This lower bound is called the Evidence Lower Bound (ELBO). To fit the keras model, instead of maximizing the ELBO, we would minimize the NELBO (Negative ELBO).

# VII. Acknowledgements
- Inspired by https://github.com/JeremyCCHsu/vae-npvc
- Guided by https://blog.keras.io/building-autoencoders-in-keras.html

# VIII. License

_This project is licensed under the terms of the CentraleSupélec license_.

Reproduction and modifications are allowed as long as there is a **mention** of either of the contributors or of this repository.

# IX. Contributors

- **Yun Bin Choh** - _Student @ CentraleSupélec_ - [lukecyb8687](https://github.com/lukecyb8687)
- **Si Jie Tang**  - _Student @ CentraleSupélec_ -[lamartang123](https://github.com/lamartang123)
