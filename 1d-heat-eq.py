import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# parameters
n_x = 51                # number of spatial points
dx = 1.0 / (n_x - 1)    # spatial step

t_final = 1             # duration of heating in seconds
dt = 0.5 * dx**2        # time step governed by stability condition

'''
l                       # length of rod
alpha                   # thermal diffusivity of rod medium
t_i                     # initial/boundary temperature of rod
t_b                     # boundary temperature of rod
q                       # heat flux of heat emitter
k                       # thermal conductivity of rod medium
'''

# stability condition: fourier number, F_0 <= 0.5
# F_0 = alpha * dt / dx**2


def heat_model_dmsnless():
    # 1d heat equation dimensionless solution

    # initial temperature distribution
    u_i = np.zeros(n_x)
    u_i[0] = 1.0
    u_i[-1] = 1.0

    u = [u_i.copy()]
    f = [u_i.copy()]

    n_t = int(t_final / dt) + 1

    x = np.linspace(0,1,n_x)

    # time evolution
    for t in range(n_t-1):
        u_new = u_i.copy()
        for i in range(1, n_x-1):
            # finite difference method
            u_new[i] = u_i[i] + dt / dx**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_i = u_new.copy()
        u.append(u_new)

        # fourier approximation
        f.append([fourier_series((t + 1) * dt, x_i) for x_i in x])

    plot_graphs(u, f, n_x, n_t)


def heat_model(l, t_b, t_i, alpha = 0.14e-6, t_final = 3600):
    # 1d heat equation dimensional solution

    # default alpha: thermal diffusivity of water
    # default t_final: 1 hour

    dx_d = dx * l
    dt_d = 0.5 * dx_d**2 / alpha

    # initial temperature distribution
    u_i = np.zeros(n_x)
    u_i[0] = 1.0
    u_i[-1] = 1.0
    u = [u_i.copy()]

    t_1 = t_i * np.ones(n_x)
    t_1[0] = t_b
    t_1[-1] = t_b
    t = [t_1.copy()]
    f = [t_1.copy()]

    n_t = int(t_final / dt_d) + 1

    x = np.linspace(0,l,n_x)

    # time evolution
    for n in range(n_t - 1):
        u_new = u_i.copy()
        for i in range(1, n_x - 1):
            #finite difference method (FDM)
            u_new[i] = u_i[i] + alpha * dt_d/dx_d**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_i = u_new.copy()
        u.append(u_new)

        # dimensionalise u from FDM
        t.append([t_i + (t_b - t_i) * u_new[i] for i in range(n_x)])

        # fourier approximation
        f.append([t_i + (t_b - t_i) * fourier_series((n + 1) * dt_d * alpha / l**2, x_i / l) for x_i in x])


    plot_graphs(t, f, n_x, n_t, False, l, t_i, t_b, t_final)


#############################
# heat emitter functions
#############################

def heat_emitter_dmsnless():
    # 1d heat equation dimensionless solution
    # boundary walls at fixed temp

    # initial temperature distribution
    u_i = np.zeros(n_x)
    u_i[0] = 1.0

    u = [u_i.copy()]
    f = [u_i.copy()]

    n_t = int(t_final / dt) + 1

    x = np.linspace(0,1,n_x)

    # time evolution
    for t in range(n_t-1):
        u_new = u_i.copy()
        for i in range(1, n_x-1):
            # finite difference method
            u_new[i] = u_i[i] + dt / dx**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_i = u_new.copy()
        u.append(u_new)

        # fourier approximation
        f.append([fourier_series_emitter((t + 1) * dt, x_i) for x_i in x])

    plot_graphs(u, f, n_x, n_t)

def heat_emitter_dmsnless_t():
    #1d heat equation dimensionless solution for heat emitter
    # boundary wall with flux

    # initial temperature distribution
    u_i = 0.5 * np.ones(n_x)
    u_i[0] = 1.0

    u = [u_i.copy()]
    f = [u_i.copy()]

    n_t = int(t_final / dt) + 1

    x = np.linspace(0,1,n_x)

    # time evolution
    for t in range(n_t-1):
        u_new = u_i.copy()

        for i in range(1, n_x-1):
            # finite difference method
            u_new[i] = u_i[i] + dt / dx**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_new[-1] = u_new[-2] - dx
        u_i = u_new.copy()

        u.append(u_new)

        # fourier approximation
        f.append([fourier_series_emitter_t((t + 1) * dt, 0.5, x_i) for x_i in x])
    
    plot_graphs(u, f, n_x, n_t)

def heat_emitter_dmsnless_p():
    #1d heat equation dimensionless solution for heat emitter
    # boundary wall with zero flux

    # initial temperature distribution
    u_i = np.zeros(n_x)
    u_i[0] = 1.0

    u = [u_i.copy()]
    f = [u_i.copy()]

    n_t = int(t_final / dt) + 1

    x = np.linspace(0,1,n_x)

    # time evolution
    for t in range(n_t-1):
        u_new = u_i.copy()

        for i in range(1, n_x-1):
            # finite difference method
            u_new[i] = u_i[i] + dt / dx**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_new[-1] = u_new[-2]
        u_i = u_new.copy()

        u.append(u_new)

        # fourier approximation
        f.append([fourier_series_emitter_p((t + 1) * dt, x_i) for x_i in x])
    
    plot_graphs(u, f, n_x, n_t)

def heat_emitter_dmsnless_f():
    # 1d heat equation dimensionless solution for heat emitter
    # emitter boundary wall with flux

    # initial temperature distribution
    u_i = np.zeros(n_x)

    u = [u_i.copy()]
    f = [u_i.copy()]

    n_t = int(t_final / dt) + 1

    x = np.linspace(0,1,n_x)

    # time evolution
    for t in range(n_t-1):
        u_i[0] = u_i[1] + dx
        u_new = u_i.copy()

        for i in range(1, n_x-1):
            # finite difference method
            u_new[i] = u_i[i] + dt / dx**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_i = u_new.copy()
        u.append(u_new)

        # fourier approximation
        f.append([fourier_series_emitter_f((t + 1) * dt, x_i) for x_i in x])
    
    plot_graphs(u, f, n_x, n_t)


def heat_emitter_model_f(l, t_i, q, k = 0.026, alpha = 22e-6, t_final = 3600):
    # 1d heat equation dimensional solution for heat emitter

    # default k: thermal conductivity of air
    # default alpha: thermal diffusivity of air
    # default t_final: 1 hour

    dx_d = dx * l
    dt_d = 0.5 * dx_d**2 / alpha

    # initial temperature distribution
    u_i = np.zeros(n_x)
    u_i[0] = 1.0
    u = [u_i.copy()]

    t_1 = t_i * np.ones(n_x)
    t_1[0] = t_b
    t_1[-1] = t_b
    t = [t_1.copy()]
    f = [t_1.copy()]

    n_t = int(t_final / dt_d) + 1

    x = np.linspace(0,l,n_x)

    # time evolution
    for n in range(n_t - 1):
        u_i[0] = u_i[1] + dx * q
        u_new = u_i.copy()

        for i in range(1, n_x - 1):
            #finite difference method (FDM)
            u_new[i] = u_i[i] + alpha * dt_d/dx_d**2 * (u_i[i+1] - 2*u_i[i] + u_i[i-1])

        u_i = u_new.copy()
        u.append(u_new)

        # dimensionalise u from FDM
        t.append([t_i + (q * l / k) * u_new[i] for i in range(n_x)])

        # fourier approximation
        f.append([t_i + (q * l / k) * fourier_series_emitter_f((n + 1) * dt_d * alpha / l**2, x_i / l) for x_i in x])

    t_b = t_i + q*l/k
    plot_graphs(t, f, n_x, n_t, False, l, t_i, t_b, t_final)


#############################
# fourier functions
#############################


def fourier_series(t, x, i = 100): 
    # fourier series for boundary heating

    theta = 1 + sum((-4/(n * np.pi)) * np.exp(-n**2 * np.pi**2 * t) 
                    * np.sin(n * np.pi * x) for n in range(1, i, 2))
    return theta

def fourier_series_emitter(t, x, i = 100):
    # fourier series for heat emitter at fixed temp
    # right boundary at fixed temp

    theta = 1 - x + sum(-2/(n*np.pi)
                * np.exp((-(n*np.pi)**2)*t) * np.sin((n*np.pi*x)) for n in range(1,i))
    return theta

def fourier_series_emitter_t(t, t_i, x, i = 100):
    # fourier series for heat emitter at fixed temp
    # right boundary with flux

    theta = 1 - x + sum(((8*(-1)**n)/(((2*n+1)*(np.pi))**2) + 4*(t_i-1)/((2*n+1)*np.pi))
                * np.exp(-(((2*n+1)*np.pi/2)**2)*t) * np.sin((2*n+1)*np.pi*x/2) for n in range(i))
    return theta

def fourier_series_emitter_p(t, x, i = 100):
    # fourier series for heat emitter at fixed temp
    # right boundary with zero flux

    theta = 1 + sum((4*(-1)/((2*n+1)*np.pi))
                * np.exp(-(((2*n+1)*np.pi/2)**2)*t) * np.sin((2*n+1)*np.pi*x/2) for n in range(i))
    return theta

def fourier_series_emitter_f(t, x, i = 100):
    # fourier series for heat emitter with flux

    theta = 1 - x + sum(-8/(((2*n+1)*(np.pi))**2)
                * np.exp(-(((2*n+1)*np.pi/2)**2)*t) * np.cos((2*n+1)*np.pi*x/2) for n in range(i))
    return theta


#############################
# plotting functions
#############################


def plot_graphs(u_fdm, u_fou, n_x, n_t, dmnsl=True, l = 1, t_i = 0, t_b = 1, t_final = 1):
    # plot evolution of temperature distribution

    # default dmsl: to correct axis labels
    # if false, specific values of l, t_i, t_b passed in

    x = np.linspace(0, l, n_x)
    t = np.linspace(0, t_final, n_t)
    u_fdm = np.array(u_fdm)
    u_fou = np.array(u_fou)

    fig1, ax1 = plt.subplots()        # fdm - x
    fig2, ax2 = plt.subplots()        # fou - x
    fig3, ax3 = plt.subplots()        # fdm - t
    fig4, ax4 = plt.subplots()        # fou - t
    fig5, ax5 = plt.subplots()        # ani - x
    fig6, ax6 = plt.subplots(subplot_kw={"projection": "3d"})

    construct_plot(fig1, ax1, l, t_i, t_b, dmnsl)
    construct_plot(fig2, ax2, l, t_i, t_b, dmnsl)
    construct_plot(fig3, ax3, l, t_i, t_b, dmnsl, t_final, True)
    construct_plot(fig4, ax4, l, t_i, t_b, dmnsl, t_final, True)
    construct_plot(fig5, ax5, l, t_i, t_b, dmnsl)
    construct_plot(fig6, ax6, l, t_i, t_b, dmnsl, t_final, False, True)

    for i in range(0,n_t,int(n_t**0.5)):
        ax1.plot(x, u_fdm[i])
        ax2.plot(x, u_fou[i])
        
    ax3.plot(t, u_fdm.T[int(n_x/2)]) # centre of rod
    ax4.plot(t, u_fou.T[int(n_x/2)]) # centre of rod

    animate_plot(fig5, ax5, u_fdm, u_fou, x, n_t, dmnsl)

    X, T = np.meshgrid(x, t)
    ax6.plot_surface(T, X, u_fou, cmap='coolwarm')
 
    plt.show()


def construct_plot(fig, ax, l = 1, t_i = 0, t_b = 1, dmnsl=True, t_final = 0, time = False, surface = False):
    # label axis depending on plot

    ax.set_xlim(0,l)
    if (time):
        ax.set_xlim(0,t_final)

    ax.set_ylim(t_i,t_b)
    if (surface):
        ax.set_xlim(0,t_final)
        ax.set_ylim(0,l)
        ax.set_zlim(t_i,t_b)

        x_ticks = np.linspace(0, t_final, 6)
        y_ticks = np.linspace(0, l, 6)
        z_ticks = np.linspace(t_i, t_b, 6)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)

        ax.invert_zaxis()
        ax.view_init(-170,-50)

    if (dmnsl):
        ax.set_xlabel("$\\xi$")
        if (time):
            ax.set_xlabel("$\\tau$")
        ax.set_ylabel("$\\Theta$")
        if (surface):
            ax.set_xlabel("$\\tau$")
            ax.set_ylabel("$\\xi$")
            ax.set_zlabel("$\\Theta$($\\xi$,$\\tau$)")
    else:
        ax.set_xlabel("x ($m$)")
        if (time):
            ax.set_xlabel("t ($s$)")

        ax.set_ylabel("u ($\\degree C$)")
        if (surface):
            ax.set_xlabel("t ($s$)")
            ax.set_ylabel("x ($m$)")
            ax.set_zlabel("u(x,t) ($\\degree C$)")


def animate_plot(fig, ax, u_fdm, u_fou, x, n_t, dmnsl=True):
    # animating temperature distribution time evolution
    
    fdm, = ax.plot([], [], '-', color = 'orange', label="Finite Difference Method")
    fou, = ax.plot([], [], '--', color = 'green', label="Fourier Approximation")
    ax.legend()

    def update(frame):
        fdm.set_data(x, u_fdm[frame])
        fou.set_data(x, u_fou[frame])
        return fdm, fou

    anim = ani.FuncAnimation(fig, update, frames=n_t, interval=20, blit=True)
    if (dmnsl):
        anim.save('1d-dmnsl.gif')

    else:
        anim.save('1d-fdm-fou.gif')
