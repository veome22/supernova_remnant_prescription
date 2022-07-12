import numpy as np
import matplotlib.pyplot as plt


def set_bins_from_posterior(vt_all, n_bins=10):
    # some fiducial values
    v_min = 100
    v_max = 500
    
    for i in range(len(vt_all)):
        v_min = min(v_min, np.min(vt_all[i]))
        v_max = max(v_max, np.max(vt_all[i]))
        
    v_max = min(v_max, 1000)
    bins = np.linspace(v_min, v_max, n_bins+1)
    return bins

def draw_posterior_cdf(vt_all, n_bins=10, n_draws=89):
    # vt_all: list of all pulsar posteriors, [list of lists]
    # n_bins: number of bins in the CDF [int]
    # n_draws: size of dataset to draw from pulsar data [int]
    
    bins = set_bins_from_posterior(vt_all, n_bins=n_bins) 

    
    # Legacy Behavior: draw one posterior point from each of the 89 pulsars
    vt_draw = np.zeros(len(vt_all))        
    for i in range(len(vt_draw)):
        vt_draw[i] = vt_all[i][np.random.randint(len(vt_all[i]))]
    
#     # Alternate behavior: draw n_draws points across all posteriors
#     vt_draw = np.zeros(n_draws)    
#     for i in range(len(vt_draw)):
#         vt_draw[i] = vt_all[np.random.randint(len(vt_all))][np.random.randint(len(vt_all[i]))]

    count, bins_count = np.histogram(vt_draw, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    return vt_draw, bins_count[1:], cdf, pdf


def draw_model_cdf(model, vt_all, n_bins=10, n_draws=300):
    n_draws = min(n_draws, len(model))
    vt_draw = np.zeros(n_draws)
    
    bins = set_bins_from_posterior(vt_all, n_bins=n_bins) 
    
    for i in range(len(vt_draw)):
        vt_draw[i] = model[np.random.randint(len(model))]
    
    count, bins_count = np.histogram(vt_draw, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    
    return vt_draw, bins_count[1:], cdf, pdf



def plot_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                   n_cdf=100, PLOT_INDEX=0, color='b', alpha=0.1, n_bins=10, n_draws=300, lw=2, label="cdf"):    
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_model_cdf(NS_KICKS_2D[PLOT_INDEX], vt_all, n_bins=n_bins, n_draws = n_draws)
        plt.plot(bins, cdf, color=color, alpha=alpha)
        
    plt.plot(bins, cdf, color=color, alpha=alpha, lw=lw,\
             label=label)
    return
    
    
    
def plot_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                       n_cdf=100, color='b', alpha=0.1, n_bins=10, n_draws=89, lw = 2, label="Pulsar Data"):
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_posterior_cdf(vt_all, n_bins=n_bins, n_draws=n_draws)
        plt.plot(bins, cdf, color=color, alpha=alpha)
        
    plt.plot(bins, cdf, color=color, alpha=alpha, lw=lw, label=label)
    return
    
    
def plot_model_pdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                   PLOT_INDEX=0, color='b', n_bins=10, n_draws=300):
    vt_draw, bins, cdf, pdf = draw_model_cdf(NS_KICKS_2D[PLOT_INDEX], vt_all, n_bins=n_bins, n_draws=n_draws)
    plt.plot(bins, pdf, color=color, \
             label=f'v_ns={NS_KICK_MULT[PLOT_INDEX]}, sigma={SIGMAS[PLOT_INDEX]}')
    return
    
def plot_posterior_pdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                       color='b', n_bins=10, n_draws=89):
    vt_draw, bins, cdf, pdf = draw_posterior_cdf(vt_all, n_bins=n_bins, n_draws=n_draws)
    plt.plot(bins, pdf, color=color, label=f'Pulsar Posterior')
    return
        

def get_avg_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                      n_cdf, PLOT_INDEX, n_bins=10, n_draws=300):
   
    cdf_all = np.zeros((n_cdf, n_bins))
    
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_model_cdf(NS_KICKS_2D[PLOT_INDEX], vt_all, n_bins=n_bins, n_draws=n_draws)
        cdf_all[i] = cdf
        
    cdf_med = np.median(cdf_all, axis=0)
    
    cdf_min = np.percentile(cdf_all, 5, axis=0)
    cdf_max = np.percentile(cdf_all, 95, axis=0)
    
    return bins, cdf_med, cdf_min, cdf_max, len(vt_draw)


def get_avg_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                           n_cdf, n_bins=10, n_draws=89):
   
    cdf_all = np.zeros((n_cdf, n_bins))
    
    for i in range(n_cdf):
        vt_draw, bins, cdf, pdf = draw_posterior_cdf(vt_all, n_bins=n_bins, n_draws=n_draws)
        cdf_all[i] = cdf
        
    cdf_med = np.median(cdf_all, axis=0)
    cdf_min = np.percentile(cdf_all, 5, axis=0)
    cdf_max = np.percentile(cdf_all, 95, axis=0)
    
    return bins, cdf_med, cdf_min, cdf_max, len(vt_draw)


def plot_avg_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                       n_cdf=100, PLOT_INDEX=0, n_bins=10, n_draws=300, color='C0', alpha=0.2, lw=2, label="cdf"):
    
    bins, cdf_med, cdf_min, cdf_max, draws = get_avg_model_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                                                        n_cdf, PLOT_INDEX, n_bins=n_bins, n_draws=n_draws)
    
    plt.plot(bins, cdf_med, color=color, lw=lw,\
             label=label)
    plt.fill_between(bins, cdf_min, cdf_max, color=color, alpha=alpha)
    
    return
    
    
def plot_avg_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                           n_cdf=100, n_bins=10, n_draws=89, color='C0', alpha=0.2, lw=2, label="cdf"):
    
    bins, cdf_med, cdf_min, cdf_max, draws = get_avg_posterior_cdf(vt_all, NS_KICKS_2D, NS_KICK_MULT, SIGMAS, \
                                                            n_cdf, n_bins=n_bins, n_draws=n_draws)    
    
    plt.plot(bins, cdf_med, color=color, lw=lw, label=label)
    plt.fill_between(bins, cdf_min, cdf_max, color=color, alpha=alpha)
    
    return
    