import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

ASPECTS_csv = pd.read_csv('C:/Users/gyuha/Desktop/cASPECTS/for_bland_altman3.csv')

ASPECTS_human = ASPECTS_csv['ASPECT_Human'].tolist()
ASPECTS_rapid = ASPECTS_csv['ASPECTS_RAPID'].tolist()
ASPECTS_heuron = ASPECTS_csv['ASPECTS_Heuron'].tolist()

fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(16,5))
sm.graphics.mean_diff_plot(np.array(ASPECTS_human),
                           np.array(ASPECTS_rapid), ax = ax0)
sm.graphics.mean_diff_plot(np.array(ASPECTS_human),
                           np.array(ASPECTS_heuron), ax = ax1)
sm.graphics.mean_diff_plot(np.array(ASPECTS_rapid),
                           np.array(ASPECTS_heuron), ax = ax2)
ax0.set_title('Human vs RAPID')
ax1.set_title('Human vs Heuron')
ax2.set_title('RAPID vs Heuron')
plt.show()