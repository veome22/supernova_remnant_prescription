import numpy as np
import matplotlib.pyplot as plt


def set_bins_from_posterior(vt_all, n_bins=10):
    v_min = 500
    v_max = 0
    
    for i in range(len(vt_all)):
        v_min = min(v_min, np.min(vt_all[i]))
        v_max = max(v_max, np.max(vt_all[i]))
        
    v_max = min(v_max, 1000)
    bins = np.linspace(v_min, v_max, n_bins+1)
    return bins

def draw_posterior_cdf(vt_all, n_bins=10):
    
    vt_draw = np.zeros(len(vt_all))    
    
    bins = set_bins_from_posterior(vt_all, n_bins=n_bins) 
    
    for i in range(len(vt_all)):
        vt_draw[i] = vt_all[i][np.random.randint(len(vt_all[i]))]
    
    count, bins_count = np.histogram(vt_draw, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    return vt_draw, bins_count[1:], cdf, pdf


def draw_model_cdf(model, vt_all, n_bins=10):
    n = len(vt_all)
    vt_draw = np.zeros(n)
    
    bins = set_bins_from_posterior(vt_all, n_bins=n_bins) 
    
    for i in range(n):
        vt_draw[i] = model[np.random.randint(len(model))]
    
    count, bins_count = np.histogram(vt_draw, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    return vt_draw, bins_count[1:], cdf, pdf


def plot_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                   n_cdf=100, PLOT_INDEX=0, color='b', n_bins=10):
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_model_cdf(NS_KICKS_2D[PLOT_INDEX], vt_all, n_bins=n_bins)
        plt.plot(bins, cdf, color=color, alpha = 10/n_cdf)
    plt.plot(bins, cdf, color=color, alpha = 10/n_cdf, \
             label=f'2D Projected Kicks (v_ns={NS_KICK_MULT[PLOT_INDEX]}, sigma={SIGMAS[PLOT_INDEX]})')
    return
    
    
def plot_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                       n_cdf=100, color='b', n_bins=10):
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_posterior_cdf(vt_all, n_bins=n_bins)
        plt.plot(bins, cdf, color=color, alpha = 10/n_cdf)
    plt.plot(bins, cdf, color=color, alpha = 10/n_cdf, label='Posterior CDF')
    return
    
    
def plot_model_pdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                   PLOT_INDEX=0, color='b', n_bins=10):
    vt_draw, bins, cdf, pdf = draw_model_cdf(NS_KICKS_2D[PLOT_INDEX], vt_all, n_bins=n_bins)
    plt.plot(bins, pdf, color=color, \
             label=f'2D Projected Kicks (v_ns={NS_KICK_MULT[PLOT_INDEX]}, sigma={SIGMAS[PLOT_INDEX]})')
    return
    
def plot_posterior_pdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                       color='b', n_bins=10):
    vt_draw, bins, cdf, pdf = draw_posterior_cdf(vt_all, n_bins=n_bins)
    plt.plot(bins, pdf, color=color, label='Posterior PDF')
    return
        

def get_avg_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                      n_cdf, PLOT_INDEX, n_bins=10):
   
    cdf_all = np.zeros((n_cdf, n_bins))
    
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_model_cdf(NS_KICKS_2D[PLOT_INDEX], vt_all, n_bins=n_bins)
        cdf_all[i] = cdf
        
    cdf_med = np.median(cdf_all, axis=0)
    cdf_min = np.min(cdf_all, axis=0)
    cdf_max = np.max(cdf_all, axis=0)
    
    return bins, cdf_med, cdf_min, cdf_max


def get_avg_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                           n_cdf, n_bins=10):
   
    cdf_all = np.zeros((n_cdf, n_bins))
    
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_posterior_cdf(vt_all, n_bins=n_bins)
        cdf_all[i] = cdf
        
    cdf_med = np.median(cdf_all, axis=0)
    cdf_min = np.min(cdf_all, axis=0)
    cdf_max = np.max(cdf_all, axis=0)
    
    return bins, cdf_med, cdf_min, cdf_max


def plot_avg_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                       n_cdf=100, PLOT_INDEX=0, n_bins=10, color='C0'):
    
    bins, cdf_med, cdf_min, cdf_max = get_avg_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, n_cdf, PLOT_INDEX, n_bins=n_bins)
    
    plt.plot(bins, cdf_med, color=color, \
             label=f'2D Projected Kicks (v_ns={NS_KICK_MULT[PLOT_INDEX]}, sigma={SIGMAS[PLOT_INDEX]})')
    plt.fill_between(bins, cdf_min, cdf_max, color=color, alpha=0.2)
    
    return
    
    
def plot_avg_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                           n_cdf=100, n_bins=10, color='C0'):
    
    bins, cdf_med, cdf_min, cdf_max = get_avg_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, n_cdf, n_bins=n_bins)    
    
    plt.plot(bins, cdf_med, color=color, label='Posterior CDF')
    plt.fill_between(bins, cdf_min, cdf_max, color=color, alpha=0.2)
    
    return
    