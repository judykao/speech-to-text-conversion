import librosa
import numpy as np
import os

def process(audio_file):
    # Load and resample the audio file to 16kHz
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    # Convert the audio signal from floating point to 16-bit PCM
    audio_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)
    return audio_int16 


def sampling(audio_dir):
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    audio_int=[]
    for audio_file in audio_files:
        full_path = os.path.join(audio_dir, audio_file)
        audio_int.append(process(full_path))
    
    return audio_files,audio_int

def fill_zeros(list1, max_len):
    filled = []
    for sublist in list1:
        if len(sublist)>max_len:
            filled.append(sublist[:max_len])
        else:
            filled.append(np.pad(sublist, (0, max_len - len(sublist)), 'constant'))
    # list1 = [np.pad(sublist, (0, max_len - len(sublist)), 'constant') for sublist in list1]
    
    return filled



def split_data(audio_int1, audio_int2):
    splitted_audio_int1 = []
    splitted_audio_int2 = []
    chunk_length = 16000  # 1 second

    def split_chunks(data):
        chunks = []
        num_full_chunks = len(data) // chunk_length
        remainder = len(data) % chunk_length

        # Create full chunks
        for i in range(num_full_chunks):
            chunks.append(data[i * chunk_length:(i + 1) * chunk_length])

        # Handle the remaining data
        if remainder > 0:
            last_chunk = np.zeros(chunk_length, dtype=data.dtype)
            last_chunk[:remainder] = data[num_full_chunks * chunk_length:]
            chunks.append(last_chunk)

        return chunks
    
    # splitted_audio1=[]; splitted_audio2=[]
    for index in range(len(audio_int1)):
        audio1=[]; audio2=[]
        # Split both audio inputs into chunks
        audio1.append(split_chunks(audio_int1[index]))
        audio2.append(split_chunks(audio_int2[index]))
    
        # Determine the minimum number of full chunks
        print(len(audio1[0]),len(audio2[0]))
        min_num_chunks = min(len(audio1[0]), len(audio2[0])) 
    
        # Truncate the longer list to match the length of the shorter one
        audio1 = audio1[0][:min_num_chunks]
        audio2 = audio2[0][:min_num_chunks]

        splitted_audio_int1.append(audio1)
        splitted_audio_int2.append(audio2)
    return splitted_audio_int1, splitted_audio_int2


def adjust_signal(signal_2, signal_1):
    # Compute the cross-correlation between the two signals
    correlation = np.correlate(signal_1 - np.mean(signal_1), signal_2 - np.mean(signal_2), mode='full')
    
    # Find the lag (shift) corresponding to the maximum cross-correlation
    lag = np.argmax(correlation) - (len(signal_1) - 1)
    
    # Shift the second signal based on the lag
    if lag > 0:
        aligned_signal_2 = np.roll(signal_2, lag)
        aligned_signal_2[:lag] = 0  # Set shifted portion to 0 to avoid misalignment
        aligned_signal_1 = signal_1
    else:
        aligned_signal_2 = np.roll(signal_2, lag)
        aligned_signal_2[lag:] = 0  # Set shifted portion to 0 for negative lag
        aligned_signal_1 = signal_1
    
    # Trim both signals to the same length after shifting
    min_length = min(len(aligned_signal_2), len(aligned_signal_1))
    aligned_signal_1 = aligned_signal_1[:min_length]
    aligned_signal_2 = aligned_signal_2[:min_length]

    return aligned_signal_2, aligned_signal_1

def get_intersect(noise_list,clean_list):
    list1 = noise_list; list2= clean_list
    # Remove the word 'recorded' from each string
    list1 = [name.replace("recorded_", "") for name in list1]
    list2 = [name.replace("clean_", "") for name in list2]

    if len(list2)<len(list1):
        # Convert list2 to a set for faster lookup
        set_list2 = set(list2)
        
        # Get intersection with indices from list1 and list2
        intersection_with_indices = [idx1 for idx1, item in enumerate(list1) if item not in set_list2]
    else:
        set_list1 = set(list1)
        
        # Get intersection with indices from list1 and list2
        intersection_with_indices = [idx2 for idx2, item in enumerate(list2) if item in set_list1]

    return intersection_with_indices 

max_len = [233340, 240252, 223740, 250236, 220668, 238716, 246012]
raw_folders=['Task_1_Level_1','Task_1_Level_2','Task_1_Level_3','Task_1_Level_4','Task_1_Level_5','Task_1_Level_6','Task_1_Level_7']

for i in range(len(raw_folders)):
    noisy_dir=r'/working/CNN/Test/%s/Recorded'%raw_folders[i]
    noisy_filename,noisy_audio_int=sampling(noisy_dir)

    clean_dir=r'/working/CNN/Test/%s/Clean'%raw_folders[i]
    clean_filename,clean_audio_int=sampling(clean_dir)


    if len(noisy_audio_int) != len(clean_audio_int):
        print(f'Mismatch length: {raw_folders[i]}.')
        # continue
        break
    else:
        n = len(noisy_audio_int)
        print(n)
    
    print(f'Processing {n} indices in {raw_folders[i]}.')
    
    noisy_audio_int=fill_zeros(noisy_audio_int, max_len[i])
    indices = np.array(noisy_filename)

    print(len(noisy_audio_int[0]))
    print(len(noisy_audio_int[29]))

    raw_folder = raw_folders[i].replace('Task', 'Test')
    
    os.makedirs('/working/CNN/Dataset', exist_ok=True)
    os.makedirs(r'/working/CNN/Dataset/%s'%raw_folder, exist_ok=True)
    np.save(r'/working/CNN/Dataset/%s/indices.npy'%raw_folder, indices)
    np.save(r'/working/CNN/Dataset/%s/new_input.npy'%raw_folder, noisy_audio_int)