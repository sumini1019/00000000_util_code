import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# x축 데이터
x = ['2023.02.28', '2023.05.26', '2023.08.25', '2023.12.05', '2024.04.20']

# y축 데이터
cumulative_count = [1423, 3000, 4441, 6661, 8882] # 누적 건수
progress_rate = [0, 0.337, 0.5, 0.75, 1] # 진행율

# x축 데이터를 datetime 타입으로 변환
x = [datetime.strptime(date, '%Y.%m.%d').date() for date in x]

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(12,6))

# 진행율 그래프 설정
ax1.set_xlabel('Date')
ax1.set_ylabel('Progress Rate', color='red')
ax1.plot(x, progress_rate, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# x축 tick 간격 및 위치 설정
x_ticks = [x[0], x[1], x[2], x[3], x[4]]
plt.xticks(x_ticks, rotation=45, ha='right')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))

# 진행 건수 표시
for i in range(len(x)):
    ax1.text(x[i], progress_rate[i], str(cumulative_count[i]), ha='center', va='bottom')

plt.title('TimeLine')
plt.show()
