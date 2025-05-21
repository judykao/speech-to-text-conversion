import matplotlib.pyplot as plt
import numpy as np

# Data
# tasks = [
#     "Task 1 Level 1", "Task 1 Level 2", "Task 1 Level 3", "Task 1 Level 4", 
#     "Task 1 Level 5", "Task 1 Level 6", "Task 1 Level 7"
# ]

tasks = [
    "T1L1", "T1L2", "T1L3", "T1L4", "T1L5", "T1L6", "T1L7",
    "T2L1", "T2L2", "T2L3",
    "T3L1", "T3L2"
]

# reconstruct_with_mean_30 = [0.009272442654941683, 0.007087956593600201, 0.01958308496452265, 0.19529477812269883, 0.3059141474568784, 0.4618230726400286, 0.5534406052530579]
# reconstruct_with_mean = [0.008429, 0.007440, 0.0210849, 0.192401, 0.300059, 0.4399001, 0.534203]
interrupted_signal_cer = [0.04339, 0.0767828, 0.3426279, 0.7299539, 0.90753529, 0.9731081, 0.9720190, 0.12230028701820446, 0.47202084392825583, 0.5556121065297427, 0.9248744531503083, 0.9974532742389884]

# reconstruct_with_mean_30 = [0.009430355162279769, 0.007626354745935047, 0.02190027882328946, 0.259270068012538, 0.3143793179409694, 0.4873466544830556, 0.5650369293916457]
reconstruct_with_mean = [0.008648355728511917, 0.008539568516291673, 0.02380708403251787, 0.24724661674639323, 0.31540938910659777, 0.471734430720319, 0.5545294377942882, 0.14172480130624104, 0.4382961080458223, 0.5219076883190213, 0.6966106824159349, 0.8426108513279796]

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plotting the data
# plt.plot(tasks, reconstruct_with_mean_30, label='reconstruct_with_mean_30', color='red', marker='o', linestyle='-', markersize=6)
plt.axhline(0.3, linestyle='--', color='grey')
plt.plot(tasks, reconstruct_with_mean, label='Adjusted', color='orange', marker='o', linestyle='-.', markersize=6)
plt.plot(tasks, interrupted_signal_cer, label='Recorded', color='black', marker='o', linestyle='-', markersize=6)


# Add labels and title
# plt.xlabel('Task Levels', fontsize=12)
plt.ylabel('Mean CER', fontsize=12)
plt.title('Comparison of Baseline and Our Result', fontsize=12)

# Rotate x-axis labels for readability

plt.xticks(rotation=45, ha='right')

# Add a grid
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('plot/test.png')

# plt.show()
