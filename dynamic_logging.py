#!/usr/bin/python3
from matplotlib import pyplot as plt
from matplotlib import animation
FILE_NAME = "error.log"

fig = plt.figure()

# fig.suptitle(FILE_NAME, fontsize=14, fontweight='bold') # Заголовок графика

ax = plt.axes(xlim=(0, 100), ylim=(0, 0.8))
ax.set_title('') # Еще заголовок
# ax.set_ylabel('Cost function') # Подпись к оси ординат

ax.patch.set_facecolor('#0f2c2c') # Цвет фона
line, = ax.plot([], [], lw=2, color="white") # Цвет линии графика

def init():
    line.set_data([], [])
    return line,

def animate(i):
    f = open(FILE_NAME, "r")
    data_from_file = f.read().splitlines()
    f.close()
    y = [float(x) for x in data_from_file]
    x = range(1, len(y) + 1) # t = [1.. len(f)]
    line.set_data(x, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=500, blit=True)

plt.show()