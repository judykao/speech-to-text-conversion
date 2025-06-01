import os
import numpy as np
from scipy.io.wavfile import write

from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

data_folders=['Task_1_Level_1', 'Task_1_Level_2', 'Task_1_Level_3', 'Task_1_Level_4', 'Task_1_Level_5', 'Task_1_Level_6', 'Task_1_Level_7']

for i, data_folder in enumerate(data_folders):
    # Load the data from the .npy file
    input_data = np.load(r'/working/CNN/Dataset/%s/new_input.npy'%data_folder)
    output_data = np.load(r'/working/CNN/Dataset/%s/new_output.npy'%data_folder)

    # Perform FFT along the second axis (23000-dimensional)
    input_fft = np.fft.fft(input_data, axis=1)
    output_fft = np.fft.fft(output_data, axis=1)

    # Get the magnitude (absolute value) of the FFT
    input_fft_magnitude = np.abs(input_fft)
    output_fft_magnitude = np.abs(output_fft)
    log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
    log_output_fft_magnitude = np.log10(output_fft_magnitude+1)

    log_fft_magnitude_diff = log_output_fft_magnitude - log_input_fft_magnitude
    column_means = np.mean(log_fft_magnitude_diff[:30], axis=0)
    smoothed_column_means = gaussian_filter1d(column_means, sigma=2400)
    print(column_means.shape)
    adjusted_log_input_fft = log_input_fft_magnitude + smoothed_column_means

    # sample = 10

    # plt.figure()
    # plt.plot(column_means, label='mean')
    # plt.plot(smoothed_column_means, label='smoothed_mean')
    # plt.xlabel('frequency')
    # plt.ylabel('log magnitude')
    # plt.title('Average Difference')
    # plt.legend()
    # plt.savefig('plot/mean-fft.png')
    # plt.clf()

    # plt.figure()
    # plt.plot(log_output_fft_magnitude[sample, :], label='clean', color='blue')
    # plt.plot(log_input_fft_magnitude[sample, :], label='recorded', color='red')
    # plt.xlabel('frequency')
    # plt.ylabel('log magnitude')
    # plt.title('Original')
    # plt.legend()
    # plt.savefig('plot/original-fft.png')
    # plt.clf()
    
    # plt.figure()
    # plt.plot(log_output_fft_magnitude[sample, :], label='clean', color='blue')
    # plt.plot(adjusted_log_input_fft[sample, :], label='adjusted', color='red')
    # plt.xlabel('frequency')
    # # plt.ylabel('log magnitude')
    # plt.title('Adjusted')
    # plt.legend()
    # plt.savefig('plot/adjusted-fft.png')
    # plt.clf()

    adjusted_input_fft_magnitude = 10 ** adjusted_log_input_fft - 1
    input_fft_phase = np.angle(input_fft)
    adjusted_input_fft = adjusted_input_fft_magnitude * np.exp(1j * input_fft_phase)

    pred_data = np.fft.ifft(adjusted_input_fft, axis=1).real
    pred_data /= np.max(np.abs(pred_data), axis=1, keepdims=True)

    wav_folder = r'/working/CNN/Dataset/%s/pred_audio_smoothed'%data_folder
    
    os.makedirs(wav_folder, exist_ok=True)

    sample_rate = 16000
    for i, audio in enumerate(pred_data):
        wav_path = os.path.join(wav_folder, f"{data_folder.lower()}_recorded_{(i+1):03}.wav")
        print(f'Saving audio: {wav_path}')
        write(wav_path, sample_rate, (audio * 32767).astype(np.int16))

