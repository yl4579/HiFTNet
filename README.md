# HiFTNet: A Fast High-Quality Neural Vocoder with Harmonic-plus-Noise Filter and Inverse Short Time Fourier Transform

### Yinghao Aaron Li, Cong Han, Xilin Jiang, Nima Mesgarani

> Recent advancements in speech synthesis have leveraged GAN-based networks like HiFi-GAN and BigVGAN to produce high-fidelity waveforms from mel-spectrograms. However, these networks are computationally expensive and parameter-heavy. iSTFTNet addresses these limitations by integrating inverse short-time Fourier transform (iSTFT) into the network, achieving both speed and parameter efficiency. In this paper, we introduce an extension to iSTFTNet, termed HiFTNet, which incorporates a harmonic-plus-noise source filter in the time-frequency domain that uses a sinusoidal source from the fundamental frequency (F0) inferred via a pre-trained F0 estimation network for fast inference speed. Subjective evaluations on LJSpeech show that our model significantly outperforms both iSTFTNet and HiFi-GAN, achieving ground-truth-level performance. HiFTNet also outperforms BigVGAN-base on LibriTTS for unseen speakers and achieves comparable performance to BigVGAN while being four times faster with only 1/6 of the parameters. Our work sets a new benchmark for efficient, high-quality neural vocoding, paving the way for real-time applications that demand high quality speech synthesis.

Paper: [https://arxiv.org/abs/2309.09493](https://arxiv.org/abs/2309.09493)

Audio samples: [https://hiftnet.github.io/](https://hiftnet.github.io/)

**Check our TTS work that uses HiFTNet as speech decoder for human-level speech synthesis here: https://github.com/yl4579/StyleTTS2**

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/HiFTNet.git
cd HiFTNet
```
3. Install python requirements: 
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --config config_v1.json --[args]
```
For the F0 model training, please refer to [yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor). This repo includes a pre-trained F0 model on LibriTTS. Still, you may want to train your own F0 model for the best performance, particularly for noisy or non-speech data, as we found that F0 estimation accuracy is essential for the vocoder performance. 

## Inference
Please refer to the notebook [inference.ipynb](https://github.com/yl4579/HiFTNet/blob/main/inference.ipynb) for details.
### Pre-Trained Models
You can download the pre-trained LJSpeech model [here](https://huggingface.co/yl4579/HiFTNet/blob/main/LJSpeech/cp_hifigan.zip) and the pre-trained LibriTTS model [here](https://huggingface.co/yl4579/HiFTNet/blob/main/LibriTTS/cp_hifigan.zip). The pre-trained models contain parameters of the optimizers and discriminators that can be used for fine-tuning.  

## References
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)
