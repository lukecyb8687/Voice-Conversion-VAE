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
- **pandas** 

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
f0_ceil = 500
```
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

## extract_and_save_bin_to(...) function

Due to the large amount of data to process, and to allow our VAE model to run faster, we converted the featured dimensions of each {}.wav file into a {}.bin binary file:
```
#dir_to_wav = '/content/drive/My Drive/dataset/vcc2016/wav/'
#dir_to_bin = '/content/drive/My Drive/dataset/vcc2016/bin/'
#dir_to_source = dir_to_wav
def extract_and_save_bin_to(dir_to_bin, dir_to_source):
    sets = [s for s in os.listdir(dir_to_source) if s in SETS] # create 2 directories: ['Training Set', 'Testing Set']
    for d in sets:
            path = join(dir_to_source, d) # create path for the 2 directories
            speakers = [s for s in os.listdir(path) if s in SPEAKERS] # The respective speakers in the training and testing set
 
            for s in speakers:
                        path = join(dir_to_source, d, s)
                        #print(path) #eg: /content/drive/My Drive/dataset/vcc2016/wav/Training Set/SF1
                       
                        output_dir = join(dir_to_bin, d, s)
                        if not tf.gfile.Exists(output_dir): # create bin file if non-existent in the first place
                            tf.gfile.MakeDirs(output_dir)
                        for f in os.listdir(path): # for each entry in the path
                            filename = join(path, f) # create path for EVERY SINGLE WAV FILES IN THE WAV DIRECTORY
                            #print(filename)
                            if not os.path.isdir(filename): # if False, then conduct analysis
                                                features = analysis(filename)
                                                labels = SPEAKERS.index(s) * np.ones(
                                                    [features.shape[0], 1],
                                                    np.float32,
                                                )
                                                b = os.path.splitext(f)[0] # the ROOT of the file eg. soundfile NUMBER
                                                features = np.concatenate([features, labels], 1) #The WORLD vocdoer features and the speaker label are stored in binary format.
                                                                                                # Format: [[s1, s2, ..., s513, a1, ..., a513, f0, en, spk],
                                                                                                          #[s1, s2, ..., s513, a1, ..., a513, f0, en, spk],
                                                                                                          #...,
                                                                                                          #[s1, s2, ..., s513, a1, ..., a513, f0, en, spk]]
                                                with open(join(output_dir, '{}.bin'.format(b)), 'wb') as fp:
                                                    fp.write(features.tostring())
```
