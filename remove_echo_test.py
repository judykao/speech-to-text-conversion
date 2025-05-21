import os
import numpy as np
import librosa
from scipy.io.wavfile import write
from scipy.fft import fft, ifft

from scipy.ndimage import gaussian_filter1d

def find_echo_duration(clean, echo):
    _, clean_trim = librosa.effects.trim(clean, top_db=30)
    q3 = np.percentile(echo[:clean_trim[0]], 99.73)
    echo_modified = echo - q3
    end_of_clean = clean_trim[1]
    trim_threshold = np.abs(echo_modified[end_of_clean-1])*0.001
    k = end_of_clean
    for i in range(len(echo_modified[k:])):
        window = echo_modified[k+i:k+i+1600]  ## 1600 samples = 0.1 seconds
        max_in_window = np.max(window)
        if max_in_window < trim_threshold:
            break
        
    trim_index = k+i+1600
    duration = trim_index - end_of_clean
    # print(f'duration: {duration} / {duration/16000} seconds.')

    return duration

########################################################################################################################

sr = 16000
task_relation = {'Task_3_Level_1': ('Task_1_Level_2', 'Task_2_Level_2'), 'Task_3_Level_2': ('Task_1_Level_4', 'Task_2_Level_3')}

for data_folder in task_relation:
    test_data_folder = data_folder.replace('Task', 'Test')
    #==================rm_echo==================
    echo_data_folder = task_relation[data_folder][1]
    input_data = np.load(r'./Dataset/%s/new_input.npy'%echo_data_folder)
    output_data = np.load(r'./Dataset/%s/new_output.npy'%echo_data_folder)
    
    tmp = input_data
    input_data = output_data
    output_data = tmp
    
    print(input_data.shape, output_data.shape) ## clean, echo

    input_fft = np.fft.fft(input_data, axis=1)
    output_fft = np.fft.fft(output_data, axis=1)
    
    duration = []
    for clean, echo in zip(input_data, output_data):
        T = find_echo_duration(clean, echo)
        duration.append(T)
    duration = np.array(duration)
    
    upper_bound, lower_bound = round(np.percentile(duration, 75)), round(np.percentile(duration, 25))
    filtered_duration = duration[(duration >= lower_bound) & (duration <= upper_bound)]
    
    T = np.mean(filtered_duration)/16000
    lam_in_time = 6.9/T
    lam = lam_in_time/16000
    print(f'lambda: {lam}')
    
    def weight(t):
        return np.exp(-lam * t)

    len_clean = len(clean)
    t = np.array(range(len_clean))
    w = np.zeros(len_clean)
    w[0] = 1
    for i in t:
        if i < 1:
            continue
        else:
            w[i] = weight(i)

    indices = np.load(r'./Dataset/%s/indices.npy'%test_data_folder)
    data = np.load(r'./Dataset/%s/new_input.npy'%test_data_folder)

    if len(data[0]) > len_clean:
        data = data[:, :len_clean]
    else:
        w = w[:len(data[0])]
        
    Y_f = np.fft.fft(data, axis=1)

    W_f = fft(w)
    print(Y_f.shape, W_f.shape)
    
    X_f = Y_f/W_f
    pred_data = np.fft.ifft(X_f, axis=1).real

    #==================enhanced==================
    enhanced_data_folder = task_relation[data_folder][0]
    input_data = np.load(r'./Dataset/%s/new_input.npy'%enhanced_data_folder)
    output_data = np.load(r'./Dataset/%s/new_output.npy'%enhanced_data_folder)

    input_fft = np.fft.fft(input_data, axis=1)
    output_fft = np.fft.fft(output_data, axis=1)

    input_fft_magnitude = np.abs(input_fft)
    output_fft_magnitude = np.abs(output_fft)
    log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
    log_output_fft_magnitude = np.log10(output_fft_magnitude+1)

    log_fft_magnitude_diff = log_output_fft_magnitude - log_input_fft_magnitude
    column_means = np.mean(log_fft_magnitude_diff, axis=0)
    smoothed_column_means = gaussian_filter1d(column_means, sigma=4000)
    
    print(f'column_means gets in {enhanced_data_folder}.')

    # indices = np.load(r'./Dataset/%s/indices.npy'%data_folder)
    input_data = pred_data
    input_data = input_data[:, :len(output_data[0])]
    
    input_fft = np.fft.fft(input_data, axis=1)

    input_fft_magnitude = np.abs(input_fft)
    log_input_fft_magnitude = np.log10(input_fft_magnitude+1)

    adjusted_log_input_fft = log_input_fft_magnitude + smoothed_column_means

    adjusted_input_fft_magnitude = 10 ** adjusted_log_input_fft - 1
    input_fft_phase = np.angle(input_fft)
    adjusted_input_fft = adjusted_input_fft_magnitude * np.exp(1j * input_fft_phase)

    pred_data_2 = np.fft.ifft(adjusted_input_fft, axis=1).real
    np.save('./Dataset/%s/rm_echo_enhanced.npy'%data_folder, pred_data_2)
    
    pred_data_2 /= np.max(np.abs(pred_data_2), axis=1, keepdims=True)

    wav_folder = r'./Dataset/%s/rm_echo_enhanced_audio'%test_data_folder
    os.makedirs(wav_folder, exist_ok=True)

    sample_rate = 16000
    for j, audio in enumerate(pred_data_2):
        wav_path = os.path.join(wav_folder, indices[j])
        print(f'Saving audio: {wav_path}')
        write(wav_path, sample_rate, (audio * 32767).astype(np.int16))
