import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Zielverteilung
mu, sigma = 0.5, 0.1
n_frames = 1200

# ---------- i.i.d. Monte Carlo ----------
mc_samples = []

# ---------- MCMC (Random-Walk MH) ----------
mcmc_samples = []
proposal_sigma = 0.05
theta = 0.2

def log_target(x):
    return -0.5 * ((x - mu) / sigma) ** 2

#---------- Figure ----------
fig, axes = plt.subplots(
   2, 2, figsize=(12, 6),
   gridspec_kw={"width_ratios": [1, 3]}
)

(ax_mc_hist, ax_mc_trace), (ax_mcmc_hist, ax_mcmc_trace) = axes

def update(frame):
    global theta

    # ===== Monte Carlo =====
    mc_theta = np.random.normal(mu, sigma)
    mc_samples.append(mc_theta)

    # ===== MCMC =====
    proposal = theta + np.random.normal(0, proposal_sigma)
    if np.log(np.random.rand()) < log_target(proposal) - log_target(theta):
        theta = proposal
    mcmc_samples.append(theta)

    # Clear
    for ax in axes.flatten():
        ax.clear()

    # ----- MC Trace -----
    ax_mc_trace.plot(mc_samples,color="tomato", lw=1.2, linestyle="--", alpha=0.8)
    ax_mc_trace.set_title("Monte Carlo")
    ax_mc_trace.set_ylabel(r"$\theta$")
    ax_mc_trace.set_ylim(0, 1)
    ax_mc_trace.set_xticks([])
    #ax_mc_trace.set_yticks([])

    # ----- MC Histogram -----
    ax_mc_hist.hist(
        mc_samples, bins=30, density=False, color="tomato", alpha=0.8,
        orientation="horizontal"
    )
    ax_mc_hist.set_ylim(0, 1)
    ax_mc_hist.invert_xaxis()
    ax_mc_hist.set_xticks([])
    #ax_mc_hist.set_yticks([])

    # ----- MCMC Trace -----
    ax_mcmc_trace.plot(mcmc_samples, lw=1.2, linestyle="--", alpha=0.8, color="deepskyblue")
    ax_mcmc_trace.set_title("Markov Chain Monte Carlo")
    ax_mcmc_trace.set_xlabel("Iteration")
    ax_mcmc_trace.set_ylabel(r"$\theta$")
    ax_mcmc_trace.set_ylim(0, 1)
    ax_mcmc_trace.set_xticks([])
    #ax_mcmc_trace.set_yticks([])

    # ----- MCMC Histogram -----
    ax_mcmc_hist.hist(
        mcmc_samples, bins=30, density=False, color="deepskyblue",
        orientation="horizontal"
    )
    ax_mcmc_hist.set_ylim(0, 1)
    ax_mcmc_hist.invert_xaxis()
    ax_mcmc_hist.set_xticks([])
    #ax_mcmc_hist.set_yticks([])

plt.tight_layout()
ani = FuncAnimation(fig, update, frames=n_frames, interval=100)

ani.save(
    "mcmc_animation.mp4",
    writer="ffmpeg",
    fps=10,
    dpi=150
)

#plt.show()
