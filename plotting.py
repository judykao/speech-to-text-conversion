import matplotlib.pyplot as plt
import numpy as np


tasks = [
    "T1L1", "T1L2", "T1L3", "T1L4", "T1L5", "T1L6", "T1L7",
    "T2L1", "T2L2", "T2L3",
    "T3L1", "T3L2"
]

interrupted_signal_cer = [0.04339, 0.0767828, 0.3426279, 0.7299539, 0.90753529, 0.9731081, 0.9720190, 0.12230028701820446, 0.47202084392825583, 0.5556121065297427, 0.9248744531503083, 0.9974532742389884]
reconstruct_with_mean = [0.008648355728511917, 0.008539568516291673, 0.02380708403251787, 0.24724661674639323, 0.31540938910659777, 0.471734430720319, 0.5545294377942882, 0.14165002346578545, 0.44443606877378317, 0.5116943923333722, 0.6944396704620923, 0.8751964264729608]

# interrupted_signal_cer = [0.04339, 0.0767828, 0.3426279, 0.7299539, 0.90753529, 0.9731081, 0.9720190]
# origin = [0.008648355728511917, 0.008539568516291673, 0.02380708403251787, 0.24724661674639323, 0.31540938910659777, 0.471734430720319, 0.5545294377942882]
# smooth = [0.029571754171277147, 0.05297700984277315, 0.07719492154447023, 0.35229462686627744, 0.38853997102808985, 0.5339748261935876, 0.5825480365426614]

plt.figure(figsize=(10, 6))
plt.axhline(0.3, linestyle='--', color='grey')
plt.plot(tasks, reconstruct_with_mean, label='Adjusted', color='orange', marker='o', linestyle='-.', markersize=6)
# plt.plot(tasks, smooth, label='smoothed', color='blue', marker='o', linestyle='-.', markersize=6)
plt.plot(tasks, interrupted_signal_cer, label='Recorded', color='black', marker='o', linestyle='-', markersize=6)
plt.ylabel('Mean CER', fontsize=12)
plt.title('Comparison of Baseline and Our Result', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plot/our_CER.png')


# import librosa
# import librosa.display

# def plot_spec(wav_path, label, n=None):
#     y, sr = librosa.load(wav_path)

#     if n:
#         y  = y[:n]

#     print(len(y))
#     D = librosa.amplitude_to_db(abs(librosa.stft(y)), ref=np.max)

#     plt.figure(figsize=(10, 5))
#     librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
#     # plt.colorbar(format='%+2.0f dB')
#     # plt.title('Spectrogram')
#     plt.tight_layout()
#     plt.savefig(f'plot/{label}-spec.png')
#     plt.clf()

# plot_spec('Tests/Task_2_Level_3/Clean/audio_test_10_053.wav', 'T2_clean')
# plot_spec('Tests/Task_2_Level_3/Recorded/audio_test_10_053.wav', 'T2_recorded')
# # plot_spec('Dataset/Task_2_Level_3/pred_audio/task_2_level_3_recorded_001.wav', 'T2_adjusted', n=211680)
# plot_spec('Dataset/Test_2_Level_3/pred_audio/audio_test_10_053.wav', 'T2_adjusted', n=177812)
# plot_spec('Dataset/Test_2_Level_3/pred_audio_enhanced/audio_test_10_053.wav', 'T2_enh', n=177812)

# # plot_spec('Tasks/Task_3_Level_1/Clean/task_3_level_1_clean_001.wav', 'T3_clean')
# # plot_spec('Tests/Task_3_Level_1/Recorded/audio_test_9_001.wav', 'T3_recorded')
# # plot_spec('Dataset/Test_3_Level_1/rm_echo_enhanced_audio/audio_test_9_001.wav', 'T3_adjusted', n=172167)
