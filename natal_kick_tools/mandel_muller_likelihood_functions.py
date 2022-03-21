import h5py as h5
import numpy as np
from scipy import interpolate
import glob
import time
import os

def load_sim_data(bh_kicks=[200], ns_kicks=[200, 400, 800], sigmas = [0.01, 0.3, 0.7]):  
               
    SN_KICK_BH_ALL = []
    SN_KICK_NS_ALL = []
    NS_KICK_MULT = []
    SIGMAS = []
       
    for bh_kick in bh_kicks:
        for ns_kick in ns_kicks:
            for sigma in sigmas:
                path = os.environ['WORK'] + f'/supernova_remnant/bh_{bh_kick}_ns_{ns_kick}_sigma_{sigma}_combined.h5'
                print("Loading Mandel Muller model data from", path)
            
                fdata = h5.File(path, 'r')

                SN_STELLAR_TYPE = fdata['SSE_Supernovae']["Stellar_Type"][...].squeeze()
                SN_TYPE = fdata['SSE_Supernovae']["SN_Type"][...].squeeze() 
                SN_KICK = fdata['SSE_Supernovae']["Applied_Kick_Magnitude"][...].squeeze()

                maskSN_NS = ((SN_STELLAR_TYPE ==13) * (SN_TYPE == 1)) # select NSs, ignore electron capture SN
                maskSN_BH = ((SN_STELLAR_TYPE ==14) * (SN_TYPE == 1)) # select BHs, ignore electron capture SN

                SN_KICK_NS = SN_KICK[maskSN_NS]
                SN_KICK_BH = SN_KICK[maskSN_BH] 

                fdata.close()

                SN_KICK_NS_ALL.append(SN_KICK_NS)
                SN_KICK_BH_ALL.append(SN_KICK_BH)
                NS_KICK_MULT.append(ns_kick)
                SIGMAS.append(sigma)
            
    return SN_KICK_NS_ALL, SN_KICK_BH_ALL, NS_KICK_MULT, SIGMAS


def load_local_sim_data(bh_kicks=[200], ns_kicks=[400], sigmas = [0.3]):  
    paths = []
    
    for bh_kick in bh_kicks:
        for ns_kick in ns_kicks:
            for sigma in sigmas:
                path = f'../COMPAS_runs/bh_{bh_kick}_ns_{ns_kick}_sigma_{sigma}_combined.h5'
                paths.append(path)
            
    SN_KICK_BH_ALL = []
    SN_KICK_NS_ALL = []
    NS_KICK_MULT = []
    SIGMAS = []
       
    
    for ns_kick in ns_kicks:
        for sigma in sigmas:
            path = f'../COMPAS_runs/bh_{bh_kick}_ns_{ns_kick}_sigma_{sigma}_combined.h5'
            print("Loading Mandel Muller model data from", path)
            
            fdata = h5.File(path, 'r')
            
            SN_STELLAR_TYPE = fdata['SSE_Supernovae']["Stellar_Type"][...].squeeze()
            SN_TYPE = fdata['SSE_Supernovae']["SN_Type"][...].squeeze() 
            SN_KICK = fdata['SSE_Supernovae']["Applied_Kick_Magnitude"][...].squeeze()

            maskSN_NS = ((SN_STELLAR_TYPE ==13) * (SN_TYPE == 1)) # select NSs, ignore electron capture SN
            maskSN_BH = ((SN_STELLAR_TYPE ==14) * (SN_TYPE == 1)) # select BHs, ignore electron capture SN
            
            SN_KICK_NS = SN_KICK[maskSN_NS]
            SN_KICK_BH = SN_KICK[maskSN_BH] 

            fdata.close()
            
            SN_KICK_NS_ALL.append(SN_KICK_NS)
            SN_KICK_BH_ALL.append(SN_KICK_BH)
            NS_KICK_MULT.append(ns_kick)
            SIGMAS.append(sigma)
            
    return SN_KICK_NS_ALL, SN_KICK_BH_ALL, NS_KICK_MULT, SIGMAS

def p_vi_from_model(vt, model):
    '''
    Arguments:
        vt: array of pulsar transverse velocities for a given pulsar [km/s]
        model: array of ns_kicks produced using the Mandel Muller prescription in COMPAS [km/s]

        
    Return:
      p_vi_M: probability of drawing each velocity v_i in vt from the model  
    '''
    
    p_vi_M = np.zeros(len(vt))
    
    # Create interpolated pdf from model kicks
    y,x = np.histogram(model, density=True, bins="scott")
    prob_M = interpolate.interp1d(x[:-1], y, bounds_error=False, fill_value=0)
    
    # calculate probability of drawing each velocity in vt
    for i in range(len(vt)):
        p_vi_M[i] = prob_M(vt[i])
     
    return p_vi_M


def get_pulsar_probability(pulsar_data_loc, bh_kicks=[200], ns_kicks=[400], sigmas=[0.3], local=False):
    '''
    This is the MAIN function that takes in a range of Muller Mandel prescriptions and calculates the probability of drawing a given set of pulsars from them.
    
    Arguments:
    pulsar_data_loc:    The folder which contains the pulsar posterior data
    bh_kicks:           List of v_bh multipliers used to generate kicks (from the reference model)
    ns_kicks:           List of v_ns multipliers used to generate kicks (from the reference model)
    sigmas:             List of sigmas used to generate kicks (from the reference model)
    local:              a flag to switch to local versions of data loading functions. Should be replaced with a model_data_loc 
                        parameter in the future
                        
    '''
    
    # Read in the model kicks 
    if local:
        SN_KICKS_NS, SN_KICKS_BH, NS_KICK_MULT, SIGMAS = load_local_sim_data(bh_kicks=bh_kicks, ns_kicks=ns_kicks, sigmas=sigmas)
    else:
        SN_KICKS_NS, SN_KICKS_BH, NS_KICK_MULT, SIGMAS = load_sim_data(bh_kicks=bh_kicks, ns_kicks=ns_kicks, sigmas=sigmas)
        
    # Read in the posteriors
    vt_all = []
    for file in glob.glob(f'{pulsar_data_loc}/*.bootstraps'):
        vt_all.append(np.loadtxt(file, unpack=True, usecols=5))
    print(f"\n Successfully read {len(vt_all)} pulsar data files\n")

    likelihoods = np.empty(len(SN_KICKS_NS)) # array to store combined likelihoods for all models
    
    for k in range(len(SN_KICKS_NS)):       
        model_data_3d = SN_KICKS_NS[k]
        
        model_data = get_projected_velocity(model_data_3d)
        
        # Save the original ns velocities and the velocity projections for future analysis
        save_velocities(k, model_data_3d, model_data, NS_KICK_MULT, SIGMAS)

        start = time.time() 
        
        # Create interpolated pdf from model kicks
        y,x = np.histogram(model_data, density=True, bins="scott")
        model_pdf = interpolate.interp1d(x[:-1], y, bounds_error=False, fill_value=0)

        # Create array of likelihoods for all the pulsar objects
        p_vi_M_all = np.empty(len(vt_all), dtype=object)

        # Calculate probability of drawing each pulsar from the model
        p_di_M = np.empty(len(p_vi_M_all))

        for i in range(len(vt_all)):
            p_vi_M = model_pdf(vt_all[i]) # calculate probability of drawing each velocity in the given pulsar posterior         
            p_vi_M_all[i] = p_vi_M
            p_di_M[i] = np.mean(p_vi_M_all[i])

        end = time.time()
        print(f"Likelihood calculation for vns_{NS_KICK_MULT[k]}_sigma_{SIGMAS[k]} completed in:", end - start, "s")


        # Save probabilities of drawing each pulsar given the model
        fname = f"calculatedModelLikelihoods/vns_{NS_KICK_MULT[k]}_sigma_{SIGMAS[k]}"
        print(f"Writing pulsar probabilities to file: {fname}")
        np.savetxt(fname, p_di_M)

        likelihoods[k] = np.prod(p_di_M)
        print("Calculation Complete!")
        print()
        
    return likelihoods


def save_velocities(k, model_data_3d, model_data, NS_KICK_MULT, SIGMAS):
    fname = f"vns_{NS_KICK_MULT[k]}_sigma_{SIGMAS[k]}_velocities"
    if not os.path.exists('model_velocities'):
        os.makedirs('model_velocities')
    np.savetxt("model_velocities/"+fname, np.c_[model_data_3d, model_data], header="3D Velocities \t Projected 2D Velocities")
    return
    
    
def get_projected_velocity(model_velocities):
    # Project onto a plane with isotropically distributed orientation.
    # Only need to project wrt theta, since phi orientation doesn't matter for a transverse velocity
    
    theta = np.random.rand(len(model_velocities))*2*np.pi
    transverse_velocities = np.abs(model_velocities*np.sin(theta))
    
    return transverse_velocities
    
