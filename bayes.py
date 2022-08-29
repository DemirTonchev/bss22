import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as st
import pandas as pd
import xarray as xr
rng = np.random.default_rng(seed=1337)

params = {
    # 'style':
    # 'legend.fontsize': 'x-large',
    # 'figure.figsize': (15, 5),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'legend.title_fontsize': 'x-large'}

plt.rcParams.update(params)
plt.style.use('seaborn-pastel')
# plt.style.use('default')


def plot_bern_bar(theta, p_theta, ax, **kwargs):
    ax.bar(theta, p_theta, width=min(0.02, 1/len(theta)*0.3))
    ax.set(xlim=(0-0.1, 1+0.1), ylim=(0, np.max(p_theta)*1.2),
           **kwargs)
    plt.show()
    return ax

# %%
# =============================================================================
# ### show inference with two world states
# =============================================================================


# show how data is generated
p_success = 0.3
bern_rv = st.bernoulli(p_success)
bern_rv.rvs(random_state=rng, size=1)

# %%
data = np.repeat([1, 0], [1, 2])

theta = np.array([0.3, 0.7])
prior_theta = np.array([0.1, 0.9])
# Create summary values of data
z = data.sum()  # number of 1's in data
N = len(data)  # number of flips in data
# Compute the likelihood of the data for each value of theta.
bern_likelihood = theta**z * (1 - theta)**(N - z)
# Compute product of likelihood and prior (posterior)
unstd_posterior = bern_likelihood * prior_theta
posterior = unstd_posterior / unstd_posterior.sum()

_, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
axes[0].set(xlabel=r'$\theta$')
for ax, p, tlt, ylb in zip(axes,
                           [prior_theta, bern_likelihood, posterior],
                           ['Prior', 'Likelihood', 'Posterior'],
                           [r'$P(\theta)$', r'$P(D|\theta)$', r'$P(\theta|D)$']):
    plot_bern_bar(theta, p, ax,
                  title=tlt, ylabel=ylb)

# %%
# lets def a fun
theta = np.array([0.3, 0.7])
prior_theta = np.array([0.5, 0.5])

def plot_prior_post_bern(theta, prior, data):

    # Create summary values of data
    z = data.sum()  # number of 1's in data
    N = len(data)  # number of flips in data
    # Compute the likelihood of the data for each value of theta.
    bern_likelihood = theta**z * (1 - theta)**(N - z)
    # Compute product of likelihood and prior (posterior)
    unstd_posterior = bern_likelihood * prior
    posterior = unstd_posterior / unstd_posterior.sum()

    _, axes = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
    axes[0].set(xlabel=r'$\theta$')
    for ax, p, tlt, ylb in zip(axes,
                               [prior_theta, bern_likelihood, posterior],
                               ['Prior', 'Likelihood', 'Posterior'],
                               [r'$P(\theta)$', r'$P(\theta|D)$', r'$P(\theta|D)$']):
        plot_bern_bar(theta, p, ax,
                      title=tlt, ylabel=ylb)
    return axes

# achieve same result but find P(data) as denumerator


# %%
data = np.repeat([1, 0], [7, 3])

# change eps
eps = 0.1
theta = np.arange(start=0+eps, stop=1, step=eps)
prior_theta = np.repeat(1, len(theta))

plot_prior_post_bern(theta, prior_theta, data)
# %%
# zadacha prior da e s 2 vyrha blizo do 0 i 1
# ................


# %% smooth version and conjugate priors

def plt_beta(a, b, ax=None):
    # generate fun support
    x = np.linspace(start=0, stop=1, num=200)
    # density of beta
    p_x = st.beta.pdf(x, a, b)
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, p_x)
    ax.set(xlim=(0-0.05, 1+0.05), ylim=(0, np.max(p_x[p_x < np.inf])*1.2))
    return ax


def plt_conjugate_bernbeta(data, prior_a=1, prior_b=1):
    # find opportunities for code refactoring
    # Create summary values of data

    z = data.sum()  # number of 1's in data
    N = len(data)  # number of flips in data
    post_a = prior_a+z
    post_b = prior_b+N-z
    # bernuli likelihood
    theta = np.linspace(0, 1, 200)
    bern_likelihood = theta**z * (1 - theta)**(N - z)

    fig, (ax_prior, ax_lkhd, ax_post) = plt.subplots(3, 1, sharex=True)
    fig.suptitle(f'success={z}, failure={N-z}', fontsize=16)
    # prior
    plt_beta(prior_a, prior_b, ax=ax_prior) \
        .set(title='Prior', ylabel=r'$P(\theta)$')
    # likelihood
    ax_lkhd.plot(theta, bern_likelihood)
    ax_lkhd.set(title='Likelihood', ylabel=r'$P(D|\theta)$')
    # posterior
    plt_beta(post_a, post_b, ax=ax_post) \
        .set(title='Posterior', xlabel=r'$\theta$', ylabel=r'$P(\theta|D)$')


data = np.repeat([1, 0], [2, 3])
plt_conjugate_bernbeta(data, prior_a=1, prior_b=1)


# %% same stuff with pymc

data = np.repeat([1, 0], [2, 3])

a_prior = 1
b_prior = 1
with pm.Model() as beta_bern_model:

    theta = pm.Beta('theta', alpha=a_prior, beta=b_prior)
    y = pm.Bernoulli('y', p=theta, observed=data)
    idata = pm.sample(2000)

# %% just for comparison
z = data.sum()  # number of 1's in data
N = len(data)
pm_post_theta = idata.posterior.stack(sample=['chain', 'draw'])['theta'].data
ax = plt_beta(a_prior+z, b_prior+N-z)
ax.hist(pm_post_theta, bins=100, density=True)

# %%
# bounded beliefs

with pm.Model() as beta_bern_model:

    theta = pm.Bound('theta',
                     pm.Beta.dist(alpha=a_prior, beta=b_prior),
                     lower=0.5)
    y = pm.Bernoulli('y', p=theta, observed=data)
    idata = pm.sample(2000)

pm_post_theta = idata.posterior.stack(sample=['chain', 'draw'])['theta'].data
ax = plt_beta(a_prior+z, b_prior+N-z)
ax.hist(pm_post_theta, bins=100, density=True)

# %% generate data
# =============================================================================
# lin reg
# =============================================================================


def generate_simple_reg(x, N=100, true_a=1, true_b=2, true_sd=1):
    true_mu = true_a + true_b * x
    return rng.normal(loc=true_mu, scale=true_sd, size=N)


N = 100
predictor = rng.normal(loc=0, scale=6, size=N)
outcome = generate_simple_reg(predictor, true_sd=3)

_, data_scatter = plt.subplots()
data_scatter.scatter(predictor, outcome)

# %%

prior_samples = 50
with pm.Model() as model_1:
    # prior on params
    a = pm.Normal("a", 0.0, 10.0)
    b = pm.Normal("b", 0.0, 10.0)
    # deterministic function
    _mu = a + b * predictor
    mu = pm.Deterministic('mu', _mu)
    sigma = pm.Exponential("sigma", 1.0)

    pm.Normal("obs", mu=mu, sigma=sigma, observed=outcome)
    idata = pm.sample_prior_predictive(
        samples=prior_samples, random_seed=rng)
    # idata.extend(pm.sample(1000, tune=2000, random_seed=rng_jesus))

prior = idata.prior.stack(sample=("draw", "chain")).copy()
# %%

x = np.linspace(-20, 20, 50)  # [:, None]
# y = prior["a"].data + prior["b"].data * x

# this can be vectorized
a_samples = prior['a'].data
b_samples = prior['b'].data

_, ax = plt.subplots()
for i in range(prior_samples):
    # this can be vectorized
    ax.plot(x, a_samples[i] + b_samples[i] * x, "b", alpha=0.2)
    ax.set_title("Prior predictive checks (weakly informative)")

# %% run the same with different priors :)
# .......
# %%
# train the model

with model_1:
    idata.extend(pm.sample(tune=2000, random_seed=rng))
posterior = idata.posterior.stack(sample=("draw", "chain")).copy()

# %%
a_samples = posterior['a'].data
b_samples = posterior['b'].data

for i in range(prior_samples):
    # this can be vectorized
    data_scatter.plot(x, a_samples[i] + b_samples[i] * x, alpha=0.2)
    data_scatter.set_title("Posterior samples")

# %%
# sample from posterior predictive
with model_1:
    pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=rng)

# %%
ppc = idata.posterior_predictive.stack(sample=("draw", "chain")).copy()

plt.plot(predictor, ppc['obs'].data[:, :10], '.', c='orange', markersize=10)

# %%

post_mus = posterior['a'].data + posterior['b'].data * x[:, None]
mean_mus = post_mus.mean(axis=1)
hdis = az.hdi(post_mus.T)
sigmas = posterior['sigma'].data

y_ppd = rng.normal(post_mus, sigmas)
ppd_hdi = az.hdi(y_ppd.T)

_, ax = plt.subplots()
ax.scatter(predictor, outcome,s=10)
ax.plot(x, mean_mus, c='k')
ax.fill_between(x, hdis[:,0], hdis[:,1], color='tab:blue', alpha=0.4 )
ax.fill_between(x, ppd_hdi[:,0], ppd_hdi[:,1], color='tab:cyan', alpha=0.4 )



# %%

d = pd.read_csv("data/Howell.csv", sep=";", header=0)
d = d[d.age >= 18]
