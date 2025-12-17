#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats as st 
import matplotlib.pyplot as plt

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)

# Variables discrètes

# Loi de dirac

def dirac(a=0):
    x = np.array([a])
    p = np.array([1.0])
    return x, p

# Loi uniforme discrete

def uniforme_discrete(n=10):
    x = np.arange(1, n+1)
    p = np.ones(n) / n
    return x, p

# Loi binomiale

def binomiale(n=20, p=0.5):
    x = np.arange(0, n+1)
    pmf = st.binom.pmf(x, n, p)
    return x, pmf

# Loi poisson discrète

def poisson_discrete(lam=5, kmax=20):
    x = np.arange(0, kmax+1)
    pmf = st.poisson.pmf(x, lam)
    return x, pmf

# Loi de Zipf Mandelbrot

def zipf_mandelbrot(a=1, b=1, c=1.5, N=20):
    x = np.arange(1, N+1)
    p = 1 / ((a + b*x)**c)
    p /= np.sum(p)
    return x, p

# Variable continue

# Loi normale

def normale(mu=0, sigma=1):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    y = st.norm.pdf(x, mu, sigma)
    return x, y

# Loi log normale

def lognormale(mu=0, sigma=1):
    x = np.linspace(0.001, 5*np.exp(mu+sigma), 400)
    y = st.lognorm.pdf(x, sigma, scale=np.exp(mu))
    return x, y

# Loi uniforme continue

def uniforme_continue(a=0, b=5):
    x = np.linspace(a, b, 400)
    y = st.uniform.pdf(x, loc=a, scale=b-a)
    return x, y

# Loi chi deux

def chi_deux(df=4):
    x = np.linspace(0.01, 20, 400)
    y = st.chi2.pdf(x, df)
    return x, y

# Loi pareto

def pareto(alpha=3, xm=1):
    x = np.linspace(xm, 20, 400)
    y = st.pareto.pdf(x, alpha, scale=xm)
    return x, y




def plot_discrete(x, p, title):
    plt.figure()
    plt.stem(x, p, basefmt=" ")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_continuous(x, y, title):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.show()

# Afficahge 

# --- Discrètes ---
plot_discrete(*dirac(3), "Dirac δ₃")
plot_discrete(*uniforme_discrete(10), "Uniforme discrète")
plot_discrete(*binomiale(20, 0.3), "Binomiale (n=20, p=0.3)")
plot_discrete(*poisson_discrete(3), "Poisson discrète (λ=3)")
plot_discrete(*zipf_mandelbrot(), "Zipf–Mandelbrot")

# --- Continues ---
plot_continuous(*normale(0, 1), "Normale (μ=0, σ=1)")
plot_continuous(*lognormale(0, 1), "Log-normale (μ=0, σ=1)")
plot_continuous(*uniforme_continue(0, 5), "Uniforme continue (0,5)")
plot_continuous(*chi_deux(4), "Chi-deux (k=4)")
plot_continuous(*pareto(3, 1), "Pareto (xm=1, α=3)")