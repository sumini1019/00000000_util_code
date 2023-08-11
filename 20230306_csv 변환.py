import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import re
import matplotlib.pyplot as plt

path_csv = r'D:\OneDrive\00000000_Code\20221102_cSTROKE\log_memory_HeuronStroke - 복사본.csv'

df = pd.read_csv(path_csv)

list_str = list(df['log'])

data = {'cnt': [], 'Qkind': [], 'memory': []}

for line in list_str:
    if 'Current memory usage' in line:
        # cnt
        cnt = int(line.split('cnt ')[1].split(')')[0])
        # kind
        kind = line.split('- ')[1].split(' : ')[0]
        # memory
        memory = float(line.split(' : ')[1].split(' MB')[0])

        data['cnt'].append(cnt)
        data['kind'].append(kind)
        data['memory'].append(memory)

# df = pd.DataFrame(data)
df['kind'] = data['kind']
df['memory'] = data['memory']


# df.to_csv('memory_usage.csv', index=False)

# kind 별로 groupby 하기
grouped = df.groupby(['kind'])

# 각 kind 별로 그래프 그리기
for kind, group in grouped:
    plt.plot(group['cnt'], group['memory'])
    plt.title(f'Memory usage by cnt for {kind}')
    plt.xlabel('Cnt')
    plt.ylabel('Memory Usage (MB)')
    plt.show()


# kind 4개의 값을 1개 그래프로 그리기 --------------------------------------------------------------
# cnt를 기준으로 그룹화하여 각 cnt에서 kind의 평균값을 구합니다.
df_mean = df.groupby(['cnt', 'kind'], as_index=False)['memory'].mean()

# kind별 색상 설정
colors = {'Init': 'red', 'After_ICH': 'blue', 'After_ELVO': 'green', 'After_ASPECTS': 'purple'}

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 5))

for kind, color in colors.items():
    df_temp = df_mean[df_mean['kind'] == kind]
    ax.plot(df_temp['cnt'], df_temp['memory'], label=kind, color=color)

ax.legend()
ax.set_xlabel('Cnt')
ax.set_ylabel('Mean Memory Usage (MB)')
ax.set_title('Memory Usage by Cnt and Kind')

plt.show()


# kind 4개의 평균값을 cnt별로 그래프 그리기   ----------------------------------------------------------
# Get mean memory usage for each count
df_mean = df.groupby('cnt').mean()

# Plot mean memory usage for each count
plt.plot(df_mean['memory'], marker='x', markersize=3)

# Set plot labels
plt.xlabel('Cnt')
plt.ylabel('Mean Memory Usage (MB)')
plt.title('Mean Memory Usage by Cnt')

# Show plot
plt.show()