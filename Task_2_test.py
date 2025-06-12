#%%
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

data_folders=['Task_2_Level_1','Task_2_Level_2','Task_2_Level_3']

for data_folder in data_folders:
    test_folder = data_folder.replace('Task', 'Test')
    input_data = np.load(r'./Dataset/%s/new_input.npy'%data_folder) 
    output_data = np.load(r'./Dataset/%s/new_output.npy'%data_folder)
    test_data =  np.load(r'./Dataset/%s/new_input.npy'%test_folder)
    
    tmp = input_data
    input_data = output_data
    output_data = tmp
    
    print(input_data.shape, output_data.shape) ## clean, echo

    input_fft = np.fft.fft(input_data, axis=1)
    output_fft = np.fft.fft(output_data, axis=1)
    test_fft = np.fft.fft(test_data, axis=1)
    
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

    W_f = fft(w)
    Y_f = output_fft
    print(Y_f.shape, W_f.shape)
    
    X_f = Y_f/W_f

    test_f = test_fft/W_f
    pred_data = np.fft.ifft(test_f, axis=1).real

    np.save('./Dataset/%s/pred_data_enhanced.npy'%test_folder, pred_data)
    pred_data_enhanced /= np.max(np.abs(pred_data), axis=1, keepdims=True)

    indices = np.load(r'./Dataset/%s/indices.npy'%test_folder)

    wav_folder = r'./Dataset/%s/pred_audio'%test_folder
    os.makedirs(wav_folder, exist_ok=True)

    for i, audio in enumerate(pred_data):
        wav_path = os.path.join(wav_folder, indices[i])
        print(f'Saving audio: {wav_path}')
        write(wav_path, sr, (audio * 32767).astype(np.int16))

