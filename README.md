> [!NOTE] 
> For proper equation rendering, please view this documentation in day mode instead of night mode. 

# Author

An-Hsien KAO [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/an-hsien-kao-a63543342) , National Chengchi University, 116 Taipei, Taiwan. 

# Advisors 

[Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), National Chengchi University, 116 Taipei, Taiwan. 

[Yueh-Cheng KUO](https://sites.google.com/view/yckuo/home), National Chengchi University, 116 Taipei, Taiwan. 

# Introduction 

This is an audio deconvolution program, which is an alternative solution to the [Helsinki Speech Challenge 2024](https://blogs.helsinki.fi/helsinki-speech-challenge/), without involving any machine learning algorithm. 
This solution improves the solution [arXiv:2501.01650](https://arxiv.org/abs/2501.01650), which proposed by [Pu-Yun KOW](https://puyun321.github.io/) and [Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), who won the [second place](https://blogs.helsinki.fi/helsinki-speech-challenge/results/) in the challenge. 

## Task 1 (filter experiments) 

We first compute the log-FFT transform of both clean and polluted data. 
More precisely, we transform both clean and polluted data using [fast fourier transformc (FFT)](https://numpy.org/doc/stable/reference/routines.fft.html) in `numpy`, and then taking the logarithm of the magnitude of the transformed data. 
We compute the average difference ![\mu](https://latex.codecogs.com/png.image?\dpi{110}\mu) over all training samples. 
For each test sample, we estimate the clean spectrum by adding the average difference to the log-FFT of the recorded signal. 
The corrected spectrum is then transformed back to the time domain using the inverse FFT. 
This approach restores high-frequency content and improves clarity in muffled recordings. 

<div align="center">
<img src="plot/original-fft.png" alt="Original spectrum" width="45%" style="margin-right: 10px;" /> <img src="plot/adjusted-fft.png" alt="Adjusted spectrum" width="45%" />
</div>

## Task 2 (reverberation experiments) 

As mentioned in [arXiv:2501.01650](https://arxiv.org/abs/2501.01650), it is difficult to train the phase. We assume that the reverberation can be modeled as the convolution of a clean signal with a decaying impulse response:
<div align="center">
  
![y[n]=x[n]*h[n]](https://latex.codecogs.com/png.image?\dpi{110}y[n]=x[n]*h[n]) 
</div>

where  ![h(t)=e^{-\lambda t}](https://latex.codecogs.com/png.image?\dpi{110}h(t)=e^{-\lambda%20t}). We estimate the decay factor ![\lambda](https://latex.codecogs.com/png.image?\dpi{110}\lambda) using the following relation: 
<div align="center">
  
![\lambda \cdot T \approx 6.9](https://latex.codecogs.com/png.image?\dpi{110}\lambda\cdot%20T\approx%206.9)
</div>

where ![T](https://latex.codecogs.com/png.image?\dpi{110}T) is the reverberation time we sampled from training data.
Then, we perform deconvolution by simple division:
<div align="center">
  
![\mathcal{F}(x[n])\approx\frac{\mathcal{F}(y[n])}{\mathcal{F}(h[n])}](https://latex.codecogs.com/png.image?\dpi{110}\mathcal{F}(x[n])\approx\frac{\mathcal{F}(y[n])}{\mathcal{F}(h[n])})
</div>

where ![\mathcal{F}](https://latex.codecogs.com/png.image?\dpi{110}\mathcal{F}) is the [fast fourier transformc (FFT)](https://numpy.org/doc/stable/reference/routines.fft.html) in `numpy` mentioned above. 

## Task 3 (hybrid experiments) 

We first apply the method in [Task 2 (reverberation experiments)](#task-2-reverberation-experiments) above, and then apply the method in [Task 1 (filter experiments)](#task-1-filter-experiments) above

# Requirements 

As no neural network algorithms are involved, our program has no stringent requirements. A `requirements.txt` file is still provided, just in case. 

> [!IMPORTANT] 
> The programs only handle 16-bit 16kHz audio files, in `.wav` format.

# Results 

We compare our results to the top entries in the [Helsinki Speech Challenge 2024](https://blogs.helsinki.fi/helsinki-speech-challenge/). 
To evaluate the quality of speech restoration, we use a metric in speech recognition called the Character Error Rate (CER). 
A lower CER indicates better intelligibility and restoration quality. 
We use a pretrained [DeepSpeech](https://github.com/mozilla/DeepSpeech) model to transcribe both the original and restored audio signals. 
The transcriptions are then compared to the ground truth text to compute the CER. 
The top-performing teams primarily used deep learning models for speech enhancement. 
ur method is based entirely on mathematical modeling and signal processing, using the Fourier domain and a simple statistical estimation from training data. Thus, the method itself is lightweight with extremely low real-time factors (RTF).
In some cases, [DeepSpeech](https://github.com/mozilla/DeepSpeech) fails to capture the actual improvement, but human listeners report a clear increase in intelligibility after processing.

<div align="center">
<img src="plot/our_CER.png" alt="Our results" width="45%" style="margin-right: 10px;" /> <img src="plot/HSC2024results.png" alt="HSC2024results" width="45%" />
</div>

## Task 1 

https://github.com/user-attachments/assets/8122e762-92a6-444f-9984-33eaf47ffc25

https://github.com/user-attachments/assets/b410ca4a-debd-4ee3-b8c2-8ac155e65f65

https://github.com/user-attachments/assets/93a58c1d-d18b-4ac4-a7f6-9441d97c195a

## Task 2 

https://github.com/user-attachments/assets/190b3b5b-6ec4-46fb-a464-e843aaaafc69

https://github.com/user-attachments/assets/3e70fe97-7931-4c2a-84dd-fb1fe9150877

https://github.com/user-attachments/assets/7787900c-d074-4329-9d10-41890a437d51

## Task 3 

https://github.com/user-attachments/assets/daefcb18-1972-47d0-9995-b4df0ebf3fcb

https://github.com/user-attachments/assets/cf85b2ab-eb10-4b73-bf0d-3263d965545a

https://github.com/user-attachments/assets/fcd69d10-09e8-4fdb-ab27-33dbe4dba1a4


[comment]: <> (ffmpeg -f lavfi -i color=c=black:s=100x100:d=10 -i input.wav -shortest -c:v libx264 -c:a aac output.mp4)

[comment]: <> (https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
