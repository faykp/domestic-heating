import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

# .dat containing data from MIDAS Open -- Met Office
data = np.loadtxt("temperature.dat", unpack=True)

max_air = data[0]           # maximum air temperature am
min_air = data[1]           # minimum air temperature am
min_grass = data[2]         # minimum grass temperature am
avg_air = data[3]           # average air temperature
soil = data[4]              # soil (1 metre depth) temperature
max_air_pm = data[5]        # maximum air temperature pm
min_air_pm = data[6]        # minimum air temperature pm
avg_air_pm = data[7]        # average air temperature pm

diff = [avg_air[i] - soil[i] for i in range(365)]


#############################
# plotting functions
#############################


def construct_plots(ax):
    # plot graphs against soil temperatures
    ax.plot(dates,soil, label="$T_{soil}$", color = "orange")
    ax.set_xticks(dates[::20], labels=dates[::20], rotation=30)
    ax.set_ylabel("$Temperature, \\degree$C")

def legend(ax):
    # show legend
    ax.legend()


#############################
# main
#############################


start = dt.date(2023,1,1)
dates = []

for day in range(365):
    # set dates for tick marks
    dates.append(start.isoformat())
    start += dt.timedelta(days=1)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

axes = [ax1,ax2,ax3,ax4]

# plot air & grass temperatures
ax1.plot(dates,max_air,label="$T_{air,max}$", color = "skyblue")
ax1.plot(dates,min_air,label="$T_{air,min}$", color = "steelblue")
ax2.plot(dates,avg_air,label="$T_{air,avg}$")
ax3.plot(dates,min_grass, label = "$T_{grass}$", color = "green")
ax4.plot(dates, diff, label="$\\Delta T = T_{air,avg} - T_{soil}$", color = "orange")

for i in range(4):
    construct_plots(axes[i])
    if i == 3:
        axes[i].lines[-1].remove()
    legend(axes[i])

plt.show()
