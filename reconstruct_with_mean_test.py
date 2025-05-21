#%%
import os
import numpy as np
from scipy.io.wavfile import write

data_folders=['Task_1_Level_1', 'Task_1_Level_2', 'Task_1_Level_3', 'Task_1_Level_4', 'Task_1_Level_5', 'Task_1_Level_6', 'Task_1_Level_7']
# data_folders=['Task_2_Level_1','Task_2_Level_2']

for i, data_folder in enumerate(data_folders):
    # Load the data from the .npy file
    input_data = np.load(r'/working/CNN/Dataset/%s/new_input.npy'%data_folder)
    output_data = np.load(r'/working/CNN/Dataset/%s/new_output.npy'%data_folder)

    # print(input_data.shape, output_data.shape)

    # Perform FFT along the second axis (23000-dimensional)
    input_fft = np.fft.fft(input_data, axis=1)
    output_fft = np.fft.fft(output_data, axis=1)

    # input_fft = input_fft[:100]
    # output_fft = output_fft[:100]

    # Get the magnitude (absolute value) of the FFT
    input_fft_magnitude = np.abs(input_fft)
    output_fft_magnitude = np.abs(output_fft)
    log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
    log_output_fft_magnitude = np.log10(output_fft_magnitude+1)

    #%%
    log_fft_magnitude_diff = log_output_fft_magnitude - log_input_fft_magnitude
    # print(log_fft_magnitude_diff.shape)
    column_means = np.mean(log_fft_magnitude_diff[:30], axis=0)
    # column_means = np.mean(log_fft_magnitude_diff, axis=0)
    print(column_means.shape)

    #==================test_data==================
    
    data_folder = data_folder.replace('Task', 'Test')
    indices = np.load(r'/working/CNN/Dataset/%s/indices.npy'%data_folder)
    input_data = np.load(r'/working/CNN/Dataset/%s/new_input.npy'%data_folder)
    input_fft = np.fft.fft(input_data, axis=1)

    # Get the magnitude (absolute value) of the FFT (test)
    input_fft_magnitude = np.abs(input_fft)
    log_input_fft_magnitude = np.log10(input_fft_magnitude+1)

    adjusted_log_input_fft = log_input_fft_magnitude + column_means

    adjusted_input_fft_magnitude = 10 ** adjusted_log_input_fft - 1
    input_fft_phase = np.angle(input_fft)
    adjusted_input_fft = adjusted_input_fft_magnitude * np.exp(1j * input_fft_phase)

    pred_data = np.fft.ifft(adjusted_input_fft, axis=1).real
    pred_data /= np.max(np.abs(pred_data), axis=1, keepdims=True)

    wav_folder = r'/working/CNN/Dataset/%s/test_audio_30'%data_folder
    os.makedirs(wav_folder, exist_ok=True)

    sample_rate = 16000
    for j, audio in enumerate(pred_data):
        # wav_path = os.path.join(wav_folder, f"{data_folder.lower()}_recorded_{(i+1):03}.wav")
        wav_path = os.path.join(wav_folder, indices[j])
        print(f'Saving audio: {wav_path}')
        write(wav_path, sample_rate, (audio * 32767).astype(np.int16))

