# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:54:17 2023

@author: Alexandre
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

print('Running neurocomputers.py')
plt.close('all')

#%% number & fun. definitions

one = np.array(((-1, -1, 1, 1, -1, -1),
                (-1, 1, 1, 1, -1, -1),
                (1, 1, 1, 1, -1, -1),
                (-1, -1, 1, 1, -1, -1),
                (-1, -1, 1, 1, -1, -1),
                (-1, -1, 1, 1, -1, -1),
                (-1, -1, 1, 1, -1, -1),
                (-1, -1, 1, 1, -1, -1),
                (-1, -1, 1, 1, -1, -1),
                (1, 1, 1, 1, 1, 1)))

two = np.array(((-1, 1, 1, 1, 1, -1),
                (1, 1, 1, 1, 1, 1),
                (1, 1, -1, -1, 1, 1),
                (-1, -1, -1, -1, 1, 1),
                (-1, -1, -1, -1, 1, 1),
                (-1, -1, -1, 1, 1, -1),
                (-1, -1, 1, 1, -1, -1),
                (-1, 1, 1, -1, -1, -1),
                (1, 1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1)))

zero = np.array(((-1, 1, 1, 1, 1, -1),
                 (-1, 1, -1, -1, 1, -1),
                 (1, 1, -1, -1, 1, 1),
                 (1, 1, -1, -1, 1, 1),
                 (1, -1, -1, -1, -1, 1),
                 (1, -1, -1, -1, -1, 1),
                 (1, 1, -1, -1, 1, 1),
                 (1, 1, -1, -1, 1, 1),
                 (-1, 1, -1, -1, 1, -1),
                 (-1, 1, 1, 1, 1, -1)))

smile = np.array(((-1, -1, -1, -1, -1, -1),
                  (-1, 1, -1, -1, 1, -1),
                  (-1, 1, -1, -1, 1, -1),
                  (-1, 1, -1, -1, 1, -1),
                  (-1, -1, -1, -1, -1, -1),
                  (-1, 1, 1, 1, 1, -1),
                  (1, 1, -1, -1, 1, 1),
                  (1, 1, -1, -1, 1, 1),
                  (1, 1, -1, -1, 1, 1),
                  (1, 1, 1, 1, 1, 1)))

fig, ax = plt.subplots(nrows=2, ncols=3)
ax = ax.flatten()
noisy_one = np.clip(one + np.random.choice((-2, 0, 2), size=one.shape,
                                           p=(0.08, 0.84, 0.08)), -1, 1)
noisy_two = np.clip(two + np.random.choice((-2, 0, 2), size=one.shape,
                                           p=(0.08, 0.84, 0.08)), -1, 1)
noisy_zero = np.clip(zero + np.random.choice((-2, 0, 2), size=one.shape,
                                           p=(0.05, 0.9, 0.05)), -1, 1)
noisy_smile = np.clip(smile + np.random.choice((-2, 0, 2), size=one.shape,
                                               p=(0.05, 0.9, 0.05)), -1, 1)
labels = ['Zero', 'One', 'Two', 'Zero + Noise', 'One + Noise', 'Two + Noise']
for ia, a in enumerate(ax):
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(labels[ia])
ax[1].imshow(one, cmap='gist_gray_r')
ax[4].imshow(noisy_one, cmap='gist_gray_r')
ax[2].imshow(two, cmap='gist_gray_r')
ax[5].imshow(noisy_two, cmap='gist_gray_r')
ax[0].imshow(zero, cmap='gist_gray_r')
ax[3].imshow(noisy_zero, cmap='gist_gray_r')
# ax[3].imshow(smile, cmap='gist_gray_r')
# ax[7].imshow(noisy_smile, cmap='gist_gray_r')
def initialization(array):
    array = array.flatten()
    c = np.zeros((60, 60))
    for i in range(60):
        for j in range(60):
            c[i, j] = array[i]*array[j]
    return c


def a_t(c, omega, t, a_o):
    a = np.zeros((60))
    for i in range(60):
        for j in range(60):
            a[i] += c[i, j] * np.cos((omega[j] - omega[i])*t)
    return a + a_o


def mean_field(phi):
    return sum([np.cos(phi[j]) + 1j*np.sin(phi[j]) for j in range(len(phi))])


def mean_field_each_phi(phi):
    mf = []
    for ip, p in enumerate(phi):
        # p1 = np.zeros((len(phi)))
        # p1[ip] = p
        mf.append(mean_field(phi-p))
    return np.imag(mf)


def ph_imag(num):
    return np.arctan(np.imag(num)/np.real(num))


def get_theta(phi, omega, t):
    return omega*t + phi

# np.sin(-phi)*np.imag(mean_field(phi))


def hebbian_learning(eta_array):
    s = np.zeros((60, 60))
    for i in range(60):
        for j in range(60):
            for r in range(eta_array.shape[0]):
                s[i, j] += eta_array[r, i]*eta_array[r, j]
    s /= 60
    return s


def pattern_rec(t, p0, s):
    phi = np.copy(p0)
    for i in range(60):
        phi[i] = sum(s[i, :]*np.sin(p0 - p0[i]))
    # theta = omega + eps*a_t(s, omega, t)
    return phi


def initialization_proc(t, p0, c_init):
    phi = np.zeros(p0.shape)
    for i in range(60):
        phi[i] = sum(c_init[i, :]*np.sin(p0 - p0[i]))
    # theta = omega + eps*a_t(c_init, omega, t)
    # phi[np.abs(phi) >= np.pi*2] -= 2*np.pi*np.sign(phi[np.abs(phi) >= np.pi*2])
    return phi


def hopfield(t, s0, w):
    return np.sign(w @ s0)


def disc_hopfield(s0, w, time):
    s = np.zeros((len(s0), len(time)))
    s[:, 0] = s0
    for it, t in enumerate(time[1:], start=1):
        s[:, it] = hopfield(t, s[:, it-1], w)
    return s


def u_potential(s, phi):
    u = 0
    for i in range(len(phi)):
        u += sum(s[i, :]*np.cos(phi-phi[i]))
    return -u/2


def overlap(pattern, state):
    return sum([pattern[i]*state[i] for i in range(len(state))])/len(state)


def get_mf_from_sim(time=np.linspace(0, 10, 10001), num_sims=20):
    plt.figure()
    last_mf_val = []
    for s in range(num_sims):
        s0 = np.random.choice(np.linspace(-1, 1, 52), one.shape).flatten()
        p0 = np.arccos(s0)*(-1)**(np.random.choice([1, 2], len(s0)))
        t_span = [0, int(max(time))]
        p0 = np.arccos(np.clip(noisy_one.flatten() +
                               np.random.randn(len(noisy_one.flatten())),
                               -1, 1))
        p0 = np.arccos(s0.flatten())
    
        pat = solve_ivp(lambda t, x: pattern_rec(t, x, s_mat), t_span, p0,
                        t_eval=time)
        mf = []
        for i, y in enumerate(pat.y.T):
            mf.append(mean_field(y))
        plt.plot(np.real(mf).T[0], np.imag(mf).T[0], 'o', color='k', alpha=0.4)
        plt.plot(np.real(mf).T, np.imag(mf).T, color='k', alpha=0.4)
        for i_t, t in enumerate(time):
            if t % 1 == 0:
                plt.plot(np.real(mf).T[i_t], np.imag(mf).T[i_t],
                         'o', color='k', markersize=10/np.sqrt(t+5), alpha=0.4)
        last_mf_val.append(mf[-1])
    plt.title('Mean field M(t)')
    # plt.plot(np.real(mf).T[0], np.imag(mf).T[0], 'o', color='k')
    # plt.plot(np.real(mf).T, np.imag(mf).T, color='k')
    plt.xlabel(r'$\mathcal{Re}(M(t)$')
    plt.ylabel(r'$\mathcal{Im}(M(t)$')
    return last_mf_val

def overlap_hopf_noise(time, hopf_def, noisy_one=noisy_one, return_overlap=True):
    ov_hopf = overlap(noisy_one.flatten(), -hopf_def)
    if not return_overlap:
        try:
            ind = np.where((np.abs(ov_hopf) == 1))[0][0]
        except:
            ind = np.nan
        return ind
    else:
        return ov_hopf    
#%% initialization

c_init = initialization(noisy_one)
nums_to_learn = np.vstack((one.flatten(), two.flatten(), zero.flatten()))
# smile.flatten()
s_mat = hebbian_learning(nums_to_learn)
t_span = [0, 10]
time = np.linspace(0, 10, 21)
omega = [0 + val for val in np.linspace(0, 1000*np.pi-1e-5, 60)]
p0 = np.random.choice(omega, size=one.flatten().size)


s0 = np.random.choice(np.linspace(-1, 1, 52), one.shape).flatten()
p0 = np.arccos(s0)*(-1)**(np.random.choice([1, 2], len(s0)))

# initialization
t_span = [0, 10]
time = np.linspace(0, 10, 21)
hopf = solve_ivp(lambda t, x: hopfield(t, x, c_init), t_span,
                 s0, t_eval=time)
time = np.linspace(0, 10, 21)
hopf_def = disc_hopfield(s0=s0, w=c_init, time=time)
fig, ax = plt.subplots(ncols=6)
for ia, a in enumerate(ax):
    a.set_xticks([])
    a.set_yticks([])
v_vals = [1, 1, 1, 1, 5]
for j in range(5):
    ax[j].imshow(-hopf_def[:, j*v_vals[j]].reshape(one.shape),
                 cmap='gist_gray_r')
    ax[j].set_title(str(time[j*v_vals[j]]) + ' s')
fig.suptitle('Initialization - Hopfield model')
ax[-1].imshow(noisy_one, cmap='gist_gray_r')
ax[-1].set_title('One + Noise')


time = np.linspace(0, 10, 21)

ini = solve_ivp(lambda t, x: initialization_proc(t, x, c_init), t_span,
                p0, t_eval=time)


u_l = []
for i in range(len(time)):
    u_l.append(u_potential(c_init, -ini.y[:, i]))

plt.figure()
plt.plot(time, np.array(u_l), color='k', label='Osc.')

time = np.linspace(0, 10, 21)
en_hopf_l = []
for i in range(len(time)):
    en_hopf_l.append(u_potential(c_init, -hopf_def[:, i])*2)
plt.plot(time, np.array(en_hopf_l), color='r', label='Hopfield')
plt.legend()
plt.title('Energy')

time = np.linspace(0, 10, 21)

ov_l = []
for i in range(len(time)):
    ov_l.append(overlap(noisy_one.flatten(), -np.sign(np.cos(ini.y[:, i]))))


plt.figure()
plt.plot(time, np.array(ov_l), color='k', label='Osc.')
plt.title('Overlap')
ov_hopf_l = []
time = np.linspace(0, 10, 21)
for i in range(len(time)):
    ov_hopf_l.append(overlap(noisy_one.flatten(), -hopf_def[:, i]))
plt.plot(time, np.array(ov_hopf_l), color='r', label='Hopfield')
plt.legend()


time = np.linspace(0, 10, 21)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Time (s)')
ax.set_title(r'${S}^1$', fontsize=18)
for y in ini.y:
    al = y  # np.mod(y, 2*np.pi)
    tim = ini.t
    ax.plot3D(tim, np.cos(al), np.sin(al),
              color='k')
    # x = np.linspace(-1, 1, len(tim))33
    # X, Y = np.meshgrid(x , tim)
    # Z = np.sqrt(1 - X**2)
    # ax.contour(X, Y, Z)
fig, ax = plt.subplots(ncols=6)
v_vals = [1, 1, 1, 1, 5]
for ia, a in enumerate(ax):
    a.set_xticks([])
    a.set_yticks([])
for j in range(5):
    ax[j].imshow(-np.cos(ini.y[:, j*v_vals[j]]).reshape(one.shape),
                 cmap='gist_gray_r')
    ax[j].set_title(str(time[j*v_vals[j]]) + ' s')
fig.suptitle('Initialization - Oscillatory')
ax[-1].imshow(noisy_one, cmap='gist_gray_r')
ax[-1].set_title('One + Noise')

u_l_i = []
for i in range(len(time)):
    u_l_i.append(u_potential(c_init, ini.y[:, i]))




# %% pattern recognition
# plt.figure()
# for y in ini.y:
#     al = np.mod(y[::100], 2*np.pi)
#     plt.plot(ini.t[::100], al)

# as init_cond of S for hopfield we must use the initialized with noise, and 
# use S now
# pattern recognition
t_span = [0, 10]

time = np.linspace(0, 10, 21)

s0 = np.clip(noisy_one.flatten() + np.random.randn(len(noisy_one.flatten())),
             -1, 1)
hopf = solve_ivp(lambda t, x: hopfield(t, x, s_mat), t_span,
                 s0, t_eval=time)
time = np.linspace(0, 10, 21)
hopf_def = disc_hopfield(s0, s_mat, time)

fig, ax = plt.subplots(ncols=6)
for ia, a in enumerate(ax):
    a.set_xticks([])
    a.set_yticks([])
for j in range(5):
    ax[j].imshow(hopf_def[:, j].reshape(one.shape), cmap='gist_gray_r')
    ax[j].set_title(str(time[j]) + ' s')
fig.suptitle('Pattern recognition - Hopfield model')
ax[-1].imshow(one, cmap='gist_gray_r')
ax[-1].set_title('One')

time = np.linspace(0, 10, 21)


p0 = np.random.choice(omega, size=one.flatten().size)
p0 = np.arccos(np.clip(noisy_zero.flatten() + np.random.randn(len(noisy_one.flatten())),
                       -1, 1))*(-1)**np.random.choice(
                           [0, 1], len(s0.flatten())).astype(float)
p0 = np.arccos(s0.flatten())*(-1)**np.random.choice(
    [0, 1], len(s0.flatten())).astype(float)
# p0 = [sum(np.sin(s0-s0[i])) for i in range(len(s0))]
# plt.imshow(p0.reshape(one.shape), cmap='gist_gray_r')

pat = solve_ivp(lambda t, x: pattern_rec(t, x, s_mat), t_span, p0,
                t_eval=time)
fig, ax = plt.subplots(ncols=6)
for ia, a in enumerate(ax):
    a.set_xticks([])
    a.set_yticks([])
vals = [1, 2, 3, 4, 5]
for j in range(5):
    ax[j].imshow(np.cos(pat.y[:, j*vals[j]]).reshape(one.shape), cmap='gist_gray_r')
    ax[j].set_title(str(time[j*vals[j]]) + ' s')
fig.suptitle('Pattern recognition - Osc.')
ax[-1].imshow(one, cmap='gist_gray_r')
ax[-1].set_title('One')

# mf = []
# for i, y in enumerate(pat.y.T):
#     mf.append(get_theta(y, omega, s_mat, time[i]))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# for y in pat.y:
#     al = y  # np.mod(y, 2*np.pi)
#     ax.plot3D(pat.t, np.cos(al), np.sin(al),
#               color='k')

u_l = []
for i in range(len(time)):
    u_l.append(u_potential(s_mat, pat.y[:, i]))

plt.figure()
plt.plot(time, np.array(u_l), color='k', label='Osc.')

time = np.linspace(0, 10, 21)
en_hopf_l = []
for i in range(len(time)):
    en_hopf_l.append(u_potential(s_mat, hopf_def[:, i])*2)
plt.plot(time, np.array(en_hopf_l), color='r', label='Hopfield')
plt.legend()
plt.title('Energy')

time = np.linspace(0, 10, 21)

ov_l_rec = []
for i in range(len(time)):
    ov_l_rec.append(overlap(one.flatten(), np.sign(np.cos(pat.y[:, i]))))

plt.figure()
plt.plot(time, np.array(ov_l_rec), color='k', label='Osc.')
plt.title('Overlap')
time = np.linspace(0, 10, 21)
ov_hopf_l_rec = []
for i in range(len(time)):
    ov_hopf_l_rec.append(overlap(one.flatten(), hopf_def[:, i]))
plt.plot(time, np.array(ov_hopf_l_rec), color='r', label='Hopfield')
plt.xlabel('Time (s)')
plt.legend()


# fig = plt.figure(figsize=(15, 5))
# fig.subplots_adjust(bottom=-0.15, top=1.2)
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(2, None)
fig = plt.figure()
ax = plt.axes(projection='3d')
# pos = ax.get_position()
time = np.linspace(0, 10, 21)

# x = np.concatenate((np.linspace(-1, -0.9, 300),
#                     np.linspace(-0.89999, 0.89999, len(time)-600),
#                     np.linspace(0.9, 1, 300)))
# # ax.set_position([pos.x0, pos.y0, pos.width*2, pos.height])
# T, X = np.meshgrid(time, x)
# Y = np.sqrt(1 - X**2)
# for t in range(Y.shape[0]):
#     ax.plot3D(time, X[t,:], Y[t,:], color='gray')
#     ax.plot3D(time, X[t,:], -Y[t,:], color='gray')
for y in pat.y:
    al = y
    tim = pat.t
    ax.plot3D(tim, np.cos(al), np.sin(al),
              color='k')
ax.set_xlabel('Time (s)')
ax.set_title(r'${S}^1$', fontsize=18)


#%% overlap
fig, ax = plt.subplots(ncols=2)
time = np.linspace(0, 10, 21)
ax[0].plot(time, np.array(ov_l), color='k', label='Osc.')
time = np.linspace(0, 10, 21)
ax[0].plot(time, np.array(ov_hopf_l), '--', color='r', label='Hopfield')
ax[0].legend()
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Overlap')
ax[0].set_title('Initialization')
ax[0].set_ylim(0, 1.1)
time = np.linspace(0, 10, 21)
ax[1].plot(time, np.array(ov_l_rec), color='k', label='Osc.')
time = np.linspace(0, 10, 21)
ax[1].plot(time, np.array(ov_hopf_l_rec), '--', color='r', label='Hopfield')
ax[1].set_title('Pat. Recognition')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylim(0, 1.1)

# %% init different noise in c_init
s0 = np.random.choice(np.linspace(-1, 1, 52), one.shape).flatten()
noise_vals = np.arange(0, 7, 1e-1)
plt.figure()
colormap = pl.cm.hot(np.linspace(0, 0.6, len(noise_vals)))
seed_vals = np.arange(200)
ov_seed = np.zeros((len(seed_vals), len(noise_vals)))
for i_s, seed in enumerate(seed_vals):
    np.random.seed(seed)
    conv_l = []
    for i_n, nois in enumerate(noise_vals):
        c_init = initialization(noisy_one) + np.random.randn(60, 60)*nois
        time = np.linspace(0, 1, 1001)
        hopf_def = disc_hopfield(s0=s0, w=c_init, time=time)
        conv = overlap_hopf_noise(time, hopf_def, return_overlap=False)
        conv_l.append(conv)
    ov_seed[i_s, :] = conv_l
ov_seed[ov_seed == 0] = np.nan
mean_vals = np.nanmean(ov_seed, axis=0)
err_vals = np.nanstd(ov_seed, axis=0)
plt.errorbar(noise_vals, mean_vals, err_vals, marker='o', color='k')
plt.yscale('log')
plt.figure()
for n in range(len(seed_vals)):
    plt.plot(noise_vals, ov_seed[n, ov_seed[n, :] != 0], 'o', color='k',
             markersize=2)
plt.ylabel('Number of steps until convergence')
plt.xlabel('Noise weight')
plt.yscale('log')
plt.ylim(0, 100)

# %% init different noise in c_init osc
s0 = np.random.choice(np.linspace(-1, 1, 52), one.shape).flatten()
p0 = np.arccos(s0.flatten())*(-1)**np.random.choice(
    [0, 1], len(s0.flatten())).astype(float)
noise_vals = np.arange(0, 7, 1e-1)
plt.figure()
colormap = pl.cm.hot(np.linspace(0, 0.6, len(noise_vals)))
seed_vals = np.arange(20)
ov_seed = np.zeros((len(seed_vals), len(noise_vals)))
t_span = [0, 30]
time = np.linspace(0, 30, 1001)
for i_s, seed in enumerate(seed_vals):
    np.random.seed(seed)
    conv_l = []
    for i_n, nois in enumerate(noise_vals):
        c_init = initialization(noisy_one) + np.random.randn(60, 60)*nois
        pat = solve_ivp(lambda t, x: pattern_rec(t, x, c_init), t_span, p0,
                        t_eval=time)
        conv = overlap_hopf_noise(time, np.sign(np.cos(pat.y)),
                                  return_overlap=False)
        if np.isnan(conv):
            conv_l.append(np.nan)
        else:
            conv_l.append(time[conv])
    ov_seed[i_s, :] = conv_l
ov_seed[ov_seed == 0] = np.nan
mean_vals = np.nanmean(ov_seed, axis=0)
err_vals = np.nanstd(ov_seed, axis=0)
plt.errorbar(noise_vals, mean_vals, err_vals, marker='o', color='k')
plt.yscale('log')
plt.figure()
for n in range(len(seed_vals)):
    plt.plot(noise_vals, ov_seed[n, ov_seed[n, :] != 0], 'o', color='k',
             markersize=2)
plt.ylabel('Time (s) until convergence')
plt.xlabel('Noise weight')
plt.yscale('log')
# plt.ylim(0, 100)


