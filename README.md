# Author

An-Hsien KAO, National Chengchi University, 116 Taipei, Taiwan. 

# Advisors 

[Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), National Chengchi University, 116 Taipei, Taiwan. 

[Yueh-Cheng KUO](https://sites.google.com/view/yckuo/home), National Chengchi University, 116 Taipei, Taiwan. 

# Introduction 

This is an audio deconvolution program, which is an alternative solution to the [Helsinki Speech Challenge 2024](https://blogs.helsinki.fi/helsinki-speech-challenge/), without involving any machine learning algorithm. 
This solution improves the solution [arXiv:2501.01650](https://arxiv.org/abs/2501.01650), which proposed by [Pu-Yun KOW](https://puyun321.github.io/) and [Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), who won the [second place](https://blogs.helsinki.fi/helsinki-speech-challenge/results/) in the challenge. 

## Task 1 (filter experiment) 

We first compute the log-FFT transform of both clean and polluted data. 
More precisely, we transform both clean and polluted data using [fast fourier transformc (FFT)](https://numpy.org/doc/stable/reference/routines.fft.html) in `numpy`, and then taking the logarithm of the magnitude of the transformed data. 
We compute the average difference $\mu$ over all training samples. 
For each test sample, we estimate the clean spectrum by adding the average difference to the log-FFT of the recorded signal. 





TBA 

# Installation instructions, including any requirements 
- [ ] add requirement.txt 
- [ ] which Python version needed in each `.py` file

# Usage instructions 

TBA 

> [!IMPORTANT] 
> The programs only handle 16-bit 16kHz audio files, in `.wav` format.

# An illustration of some example results 

[comment]: <> (https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
