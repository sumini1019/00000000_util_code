import matplotlib.pyplot as plt
import numpy as np

# Dataset for the new experiment
labels = ['Rand Augment (3)', 'Rand Augment (5)', 'Rand Augment (7)', 'Control Group']
val_iou_new = [0.755, 0.7478, 0.7313, 0.7171]
test_iou_new = [0.8572, 0.8809, 0.8593, 0.8271]

# Calculate the percentage change compared to the control group
val_iou_change_new = [(iou - val_iou_new[-1]) / val_iou_new[-1] * 100 for iou in val_iou_new]
test_iou_change_new = [(iou - test_iou_new[-1]) / test_iou_new[-1] * 100 for iou in test_iou_new]

# Generate the bar plot
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5))

rects1_new = ax.bar(x - width/2, val_iou_change_new, width, label='IoU (Val)', color='b')
rects2_new = ax.bar(x + width/2, test_iou_change_new, width, label='IoU (Test)', color='g')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance Change (%)')
ax.set_title('Performance Change compared to Control Group (by Num_RandAug)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Label with specially formatted floats
ax.bar_label(rects1_new, fmt='%.2f')
ax.bar_label(rects2_new, fmt='%.2f')

plt.show()