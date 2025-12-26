import numpy as np
import scipy.stats
from scipy.stats import beta
import matplotlib.pyplot as plt

# SET PARAMETER & VALUE
mu = np.linspace(0, 1, 1000)
N = 40
m = 10
a = 6
b = 88

# LIKELIHOOD FUNCTION (BINOMIAL)
def likelihood(N, m, mu):
    return scipy.special.binom(N, m)*mu**(m)*(1-mu)**(N-m)

# PRIOR FUNCTION (BETA)
def prior(a, b, mu):
    return beta.pdf(mu, a, b)

# POSTERIOR FUNCTION
def posterior(N, m, a, b, mu):
    marginal = likelihood(N, m, mu)*prior(a, b, mu)
    norm = np.trapz(marginal, mu)
    return marginal / norm

# CHECK FOR NORMALITY
print("Prior integral     =", np.trapz(prior(a, b, mu), mu))
print("Posterior integral =", np.trapz(posterior(N, m, a, b, mu), mu))

# PLOT TOGETHER
mu = np.linspace(0, 0.5, 1000)
likelihood_norm = likelihood(N, m, mu) / np.trapz(likelihood(N, m, mu), mu)
plt.plot(mu, likelihood_norm,
         color = "magenta",
         lw = 2,
         label = 'Likelihood')
plt.fill_between(mu, 0, likelihood_norm,
                 alpha = 0.2,
                 color = "magenta")
mu_max1 = mu[np.argmax(likelihood_norm)]
L_max = np.max(likelihood_norm)
plt.vlines(x=mu_max1,
           ymin=0,
           ymax=L_max,
           color='magenta',
           linestyles='--')
plt.plot(mu, prior(a, b, mu),
         color = "maroon",
         lw = 2,
         label = 'Prior')
plt.fill_between(mu, 0, prior(a, b, mu),
                 alpha = 0.2,
                 color = "maroon")
mu_max2 = mu[np.argmax(prior(a, b, mu))]
prior_max = np.max(prior(a, b, mu))
plt.vlines(x=mu_max2,
           ymin=0,
           ymax=prior_max,
           color='maroon',
           linestyles='--')
plt.plot(mu, posterior(N, m, a, b, mu),
         color = "navy",
         lw = 2,
         label = 'Posterior')
plt.fill_between(mu, 0, posterior(N, m, a, b, mu),
                 alpha = 0.2,
                 color = "navy")
mu_max3 = mu[np.argmax(posterior(N, m, a, b, mu))]
posterior_max = np.max(posterior(N, m, a, b, mu))
plt.vlines(x=mu_max3,
           ymin=0,
           ymax=posterior_max,
           color='navy',
           linestyles='--')
plt.xlabel('$\mu$')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

# PLOT LIKELIHOOD
mu = np.linspace(0, 0.5, 1000)
plt.plot(mu, likelihood(N, m, mu),
         color = "magenta",
         lw = 2,
         label = 'Likelihood')
plt.fill_between(mu, 0, likelihood(N, m, mu),
                 alpha = 0.2,
                 color = "magenta")
mu_max1 = mu[np.argmax(likelihood(N, m, mu))]
L_max = np.max(likelihood(N, m, mu))
plt.vlines(x=mu_max1,
           ymin=0,
           ymax=L_max,
           color='magenta',
           linestyles='--')
plt.xlabel('$\mu$')
plt.ylabel('Likelihood')
plt.grid()
plt.show()

# PLOT PRIOR
mu = np.linspace(0, 0.2, 1000)
plt.plot(mu, prior(a, b, mu),
         color = "maroon",
         lw = 2,
         label = 'Prior')
mu_max2 = mu[np.argmax(prior(a, b, mu))]
prior_max = np.max(prior(a, b, mu))
plt.vlines(x=mu_max2,
           ymin=0,
           ymax=prior_max,
           color='maroon',
           linestyles='--')
plt.fill_between(mu, 0, prior(a, b, mu),
                 alpha = 0.2,
                 color = "maroon")
plt.xlabel('$\mu$')
plt.ylabel('Prior')
plt.grid()
plt.show()