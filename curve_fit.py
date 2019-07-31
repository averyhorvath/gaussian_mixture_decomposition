import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from scipy import stats
from functools import partial
import pylab
import os
import random
import peakutils
from detect_peaks import detect_peaks
import time
import matplotlib.gridspec as gridspec
import pdb
import matplotlib.mlab as mlab
from scipy.signal import argrelmin
from scipy.stats import skew
import argparse
from argparse import RawTextHelpFormatter

np.set_printoptions(threshold=np.inf)
np.random.seed(2343)

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("iters", type=int,
                            help = "number of iterations")
parser.add_argument("num_gauss", type=int,
                            help = "number of gaussians to fit")
args   = parser.parse_args()

DEBUG     = 0
NUM_X_VALS= args.iters
BURN_IN   = 100
NOISE_BOOL= 1
NUM_GAUSS = args.num_gauss

x = np.linspace(-10,10,NUM_X_VALS)
gs1 = gridspec.GridSpec(2,1)

def gauss_function(amp, mu, sigma): 
    return amp * np.exp(-(x - mu) ** 2. / (2. * sigma ** 2.))

def create_mock_data():
    
    gauss_dists = 0
    if NOISE_BOOL:
        gauss_NOISE_BOOL = 0.2 * np.random.normal(size=x.size)
        gauss_dists += gauss_NOISE_BOOL
    
    plt.figure(figsize = (12,6))
    plt.subplot(gs1[0])
    for gauss in range(NUM_GAUSS):
        if gauss == 0:
            amp   = 1.0
            mu    = 3.4
            sigma = 0.5
            gauss_dists += gauss_function(amp,mu,sigma)
            plt.plot(x,gauss_function(amp,mu,sigma),'--',zorder=10)
        if gauss == 1:
            amp = 1.0
            mu  = 4.5
            sigma = 0.5

            gauss_dists += gauss_function(amp,mu,sigma)
            plt.plot(x,gauss_function(amp,mu,sigma),'--',zorder = 10)
        if gauss == 2:
            amp   = 1.0
            mu    = 2.5
            sigma = 1.0

            gauss_dists += gauss_function(amp,mu,sigma)
            plt.plot(x,gauss_function(amp,mu,sigma),'--', zorder=10)
        if gauss == 3:
            amp   = 0.5
            mu    = -3.0
            sigma = 0.5

            gauss_dists += gauss_function(amp,mu,sigma)
            plt.plot(x,gauss_function(amp,mu,sigma),'--', zorder=10)
        if gauss == 4:
            amp   = 3.0
            mu    = -4.5
            sigma = 0.5

            gauss_dists += gauss_function(amp,mu,sigma)
            plt.plot(x,gauss_function(amp,mu,sigma),'--', zorder=10)
        if gauss >= 5:
            amp   = np.random.uniform(1,2)
            mu    = np.random.uniform(-9.5,9.5)
            sigma = np.random.uniform(.2,1.0)

            gauss_dists += gauss_function(amp,mu,sigma) 
            plt.plot(x,gauss_function(amp,mu,sigma),'--',zorder=10)
            
    return gauss_dists


y        = create_mock_data() 

def fTAR(theta):
    target_dist = 0.

    #target_dist = np.sum(theta[:][0] * np.exp(-(x - theta[:][1])**2/(2*theta[:][2]**2), axis = 0))
    for gauss in range(NUM_GAUSS):
        amp=theta[gauss][0]
        mu=theta[gauss][1]
        sigma=theta[gauss][2]
        
        single_gauss = gauss_function(amp,mu,sigma)
        target_dist +=  single_gauss # theta = [[amp0,mu0,sigma0],[amp1,mu1,sigma1],etc.]
        
    return target_dist

def fTAR_final(theta):
    target_dist = 0
    for gauss in range(NUM_GAUSS):
        single_gauss = gauss_function(amp=theta[gauss][0],mu=theta[gauss][1],sigma=theta[gauss][2])
        target_dist +=  single_gauss # theta = [[amp0,mu0,sigma0],[amp1,mu1,sigma1],etc.]
        plt.plot(x,single_gauss,'--', zorder = 10)
    return target_dist

def box_muller(num_vals):
    
    x_r = np.zeros(num_vals)

    for i in range(num_vals):
        U1 = np.random.rand()
        U2 = np.random.rand()
        X1 = ((-2.*np.log(U1))**(1/2))*np.cos(2*np.pi*U2)
        X2 = ((-2.*np.log(U1))**(1/2))*np.sin(2*np.pi*U2)
        x_r[i] = X1
        #if (len(x_r) != num_vals):
        #   x_r[i] = X2

    return x_r


def theta_chi2(N,y,theta):

    chi2 = 0.

    for i in range(0,N):
        chi2 += np.log(1./np.sqrt(2*np.pi))#*sy[i]*sy[i])
        chi2 += -1.*((y[i]-fTAR(theta)[i])**2)#*sy[i]*sy[i]))

    return chi2 #- np.log(np.sqrt(8 * np.pi))


def prior_func(params,xdata, init_guess):
    THRESHOLD = 0.20
    amp = params[0]
    mu  = params[1]
    sig = params[2]

    amp_init = init_guess[0]
    mu_init  = init_guess[1]
    sig_init = init_guess[2]


    prior_arr = np.zeros(3)
    if amp < 0:
        prior_arr[0]  = -60
    elif amp > amp_init + amp_init * THRESHOLD or amp < amp_init - amp_init * THRESHOLD:
        prior_arr[0]  = -60

    if np.min(xdata) <= mu <= np.max(xdata):
        prior_arr[1] = -60
    elif mu > mu_init + mu_init * THRESHOLD or mu < mu_init - mu_init * THRESHOLD:
        prior_arr[1]  = -60

    if sig < 0:
        prior_arr[2] = -60
    elif sig > sig_init + sig_init * THRESHOLD * 2 or sig < sig_init - sig_init * THRESHOLD * 2:
        prior_arr[2]  = -60

    prior = prior_arr[0]+prior_arr[1]+prior_arr[2]

    
    return prior


def methastings(R,theta):

    N        = len(y)

    deltaAmp = 0.1
    deltaMu  = 0.1
    deltaSig = 0.1

    params_vals = np.zeros((NUM_GAUSS,R,3))

    randoms     = box_muller(3*NUM_GAUSS*R).reshape((NUM_GAUSS,R,3))
    params_old  = theta

    chi2_old    = theta_chi2(N,y,params_old)
    
    u = np.log(np.random.uniform(low = 0,high = 1,size=R))

    for i in range(0,R):
        params_new = np.zeros((NUM_GAUSS,3))
        #prior_list = np.zeros((NUM_GAUSS,3))
        #prior_list_old = np.zeros((NUM_GAUSS,3))
        for gauss in range(NUM_GAUSS):
            params_new[gauss][:]     = [randoms[gauss,i,0]*deltaAmp + params_old[gauss][0], randoms[gauss,i,1]*deltaMu + params_old[gauss][1], randoms[gauss,i,2]*deltaSig + params_old[gauss][2]]
            #prior_list[gauss][:]     = prior_func(params_new[gauss],x,theta[gauss])
            #prior_list_old[gauss][:] = prior_func(params_old[gauss],x,theta[gauss])
            
        chi2_old = theta_chi2(N,y,params_old) #+ np.sum(prior_list_old)
        chi2_new = theta_chi2(N,y,params_new) #+ np.sum(prior_list)

        if (chi2_new > chi2_old):
            params_vals[:,i] = params_new
            params_old       = params_new

        elif (chi2_new <= chi2_old):
            if (u[i] <= (chi2_new - chi2_old)):
                params_vals[:,i] = params_new
                params_old       = params_new

            else:
                params_vals[:,i] = params_old
                params_old       = params_old   
                
        #chi2_old=chi2_new
        if i % 50 == 0:
            print("%s%% COMPLETED" % (i/NUM_X_VALS*100))

    
    print(params_vals[:][BURN_IN::,:])
    
    return params_vals

def bin_data(ydata):
    NUM_BINS   = NUM_X_VALS/5

    bins        = np.linspace(min(x), max(x), 40)
    digitized   = np.digitize(x, bins)
    bin_means   = [ydata[digitized == i].mean() for i in range(0, len(bins))]

    return np.array(bins),np.array(bin_means)

def find_mu_amp_arrays(bins,y_binned):
        # want second deriv negative peaks (where most concave down) where original data is at least a certain height
        first_deriv = np.gradient(y_binned)
        second_deriv= np.gradient(first_deriv)

        
        min_idx              = argrelmin(second_deriv)[0]
        second_deriv_pks     = second_deriv[min_idx]
        bins_cut             = bins[min_idx]

        yvals_at_deriv_pks = y_binned[min_idx]
        print(bins_cut)
        # Only want where the second derivative is negative
        second_deriv_neg_pks = second_deriv_pks[second_deriv_pks < 0]
        bins_deriv_neg_pks   = bins_cut[second_deriv_pks < 0] # xvalues at second deriv negative peaks
        yvals_at_deriv_pks   = yvals_at_deriv_pks[second_deriv_pks < 0]

        # Want Second Derivative Peaks where the original data is the largest
        yvals_at_deriv_pks_sorted_idx = np.argsort(yvals_at_deriv_pks)[::-1]

        #second_deriv_neg_pks_sorted   = np.argsort(yvals)
        
        # Sort/Prioritize Second Derivative Peaks Where original data is largest
        second_deriv_neg_pks       = second_deriv_neg_pks[yvals_at_deriv_pks_sorted_idx]
        bins_deriv_neg_pks         = bins_deriv_neg_pks[yvals_at_deriv_pks_sorted_idx]
        yvals_at_deriv_pks_sorted  = yvals_at_deriv_pks[yvals_at_deriv_pks_sorted_idx]
        #print(bins_deriv_neg_pks)
        #print(second)
        second_deriv_neg_pks      = second_deriv_neg_pks[:NUM_GAUSS]
        bins_deriv_neg_pks        = bins_deriv_neg_pks[:NUM_GAUSS]
        yvals_at_deriv_pks_sorted = yvals_at_deriv_pks_sorted[:NUM_GAUSS]

        if DEBUG:
            plt.subplot(313)
            plt.plot(bins,second_deriv)
            plt.scatter(bins_deriv_neg_pks,second_deriv_neg_pks,marker ='x',color='red')
            plt.xlabel("x")
            plt.ylabel("Second Derivative of Smoothed Data")
            plt.xlim(-10,10)
            plt.savefig("./gauss_images/%s_gaussians/second_derivative.jpg" % (NUM_GAUSS),dpi = 300,  bbox_inches="tight")
            plt.show()
            raise ValueError
            

        return bins_deriv_neg_pks, yvals_at_deriv_pks_sorted

def find_peaks(bins, y_binned):
    indexes = detect_peaks(y_binned, mph=0.00004, mpd=2)
    
    peaks_ind        = np.argsort(y_binned[indexes])
    peaks_ind        = peaks_ind[::-1]

    peak_yval_sorted = y_binned[indexes][peaks_ind]
    peak_xval_sorted = bins[indexes][peaks_ind]
    indexes_sorted   = indexes[peaks_ind]

    return peak_yval_sorted, indexes_sorted

def main():
    SAVE_FIG   = True
    PLOT_SHOW  = True

    if NOISE_BOOL == 0:
        NOISE_BOOL_str = "NO"
    elif NOISE_BOOL == 1:
        NOISE_BOOL_str = "with"
    
    
    gauss_dists = y
    plt.plot(x,gauss_dists,zorder=0,color='k')
  
    R      = NUM_X_VALS
    
    method = "Methast"

    plt.tight_layout()
    plt.plot(x,gauss_dists,color = 'k')
    plt.xlabel("x", fontsize = 12)
    plt.ylabel("Original Data", fontsize = 12)
    #plt.title("%s Fitting %s Gaussians %s Noise \n Samples = %s" % (method,NUM_GAUSS,NOISE_BOOL_str,R), fontsize = 14)
    plt.xlim(-10,10)
    
    
    # smooth data via binning
    plt.subplot(gs1[1])
    bins, y_binned = bin_data(gauss_dists)
    bins = np.array(bins)
    plt.plot(bins,y_binned, 'k')
    plt.xlim(-10,10)
    
    

    peak_yval_sorted, indexes_sorted = find_peaks(bins,y_binned)
    mu_arr, amp_arr  = find_mu_amp_arrays(bins,y_binned)

    #plt.plot(bins[indexes_sorted][:NUM_GAUSS],peak_yval_sorted[:NUM_GAUSS],"o", label = "Peak finding algorithm")
    plt.plot(mu_arr,amp_arr,".", markersize = 15,color = 'orange',label = "Initial Guesses using 2nd derivative")
    
    plt.ylabel("Smoothed Data with Peaks")
    plt.xlabel("x")
    plt.legend()
    plt.savefig("./gauss_images/%s_gaussians/peaks.jpg" % (NUM_GAUSS),dpi = 300,  bbox_inches="tight")
    plt.show()
    raise ValueError
    #plt.bar(bins,y_binned,color='darkblue',edgecolor='.75',zorder=10)
    #plt.savefig("./gauss_images/%s_gaussians/peaks_vs_derivative.jpg" % (NUM_GAUSS),dpi = 300,  bbox_inches="tight")

    theta = []
    for gauss in range(NUM_GAUSS):
       
        mu_ind  = np.array(np.where(bins == mu_arr[gauss])).flatten()[0]
        print("mu ind", mu_ind)

        cond    = np.array(np.where(y_binned[mu_ind-8:mu_ind+8] > amp_arr[gauss]/2.0)).flatten()
        
        i=0
        while len(bins[cond]) < 3:
            i+=1
            cond = np.array(np.where(y_binned[mu_ind - i : mu_ind + i] > amp_arr[gauss]/2.0)).flatten()
        print(bins[cond])
        fwhm = np.abs(max(bins[cond]) - min(bins[cond]))
        sigma = np.exp((fwhm/8)**2)

        print("GAUSS ", gauss)
        print("\tamp initial guess = ", amp_arr[gauss])
        print("\tmu initial guess = ", mu_arr[gauss])
        print("\tsigma initial guess = ", sigma)
        theta.append([amp_arr[gauss],mu_arr[gauss],sigma])

    
    
    start   = time.time()
    samples = methastings(R,theta)
    end     = time.time()

    #pdb.set_trace()
    mean_params = np.zeros((NUM_GAUSS,3))
    for gauss in range(NUM_GAUSS):
        mean_params[gauss] = samples[gauss].mean(axis=0)
    import cProfile
    #cProfile.run('methastings(1000,[[1.0192301732732039, -5.9183673469387754, 1.2244897959183678], [0.95111433854981375, 2.6530612244897966, 1.0204081632653064]])')
    #cProfile.run('methastings(1000,[[1.0192301732732039, -5.9183673469387754, 1.2244897959183678]])')
    plt.subplot(gs1[-1])
    plt.plot(x,fTAR_final(mean_params),zorder=0,color = 'k')

    
    plt.xlabel("x", fontsize = 12)
    plt.ylabel("Model Fit", fontsize = 12)
    plt.xlim(-10,10)

    
    #plt.tight_layout()
    #plt.rcParams["figure.figsize"] = [16,9] 

    
    if not os.path.exists("./gauss_images/%s_gaussians/%s_samples" % (NUM_GAUSS,R)):
        os.makedirs("./gauss_images/%s_gaussians/%s_samples" % (NUM_GAUSS,R))
    if SAVE_FIG:
        plt.savefig("./gauss_images/%s_gaussians/%s_samples/%s_%s_gauss_%s_noise.jpg" % (NUM_GAUSS,R,method,NUM_GAUSS,NOISE_BOOL_str),dpi = 300,  bbox_inches="tight")
    
    plt.show()

    
    for gauss in range(NUM_GAUSS):
        print("GAUSS NUMBER %s" % gauss)
        plt.figure(figsize=(16,9))
        gs2 = gridspec.GridSpec(3,2,width_ratios=[3,1])
        plt.subplot(gs2[0])
        n,bins   = np.histogram(samples[gauss][:,0],bins=100)
        peak_amp = 0.5*(bins[np.where(n == n.max())[0]]+(bins[1+np.where(n == n.max())[0]]))
        
        print("\n\tPEAK value of amp: ",peak_amp[0])
        print("\tMEAN value of amp: %s" % np.mean(samples[gauss][:,0]))
        print("\tSTD. DEV value of amp: %s" % np.std(samples[gauss][:,0]))
        
        text = "PEAK value of amp: %.2f \nMEAN value of amp: %.2f \nSTD. DEV value of amp: %.2f" % (peak_amp[0],np.mean(samples[gauss][:,0]),np.std(samples[gauss][:,0]))
        plt.hist(samples[gauss][:,0],bins=100, label = text)
        plt.title("Distribution of Parameters", fontsize = 14)
        plt.xlabel("Amplitude", fontsize = 12)
        plt.ylabel("Frequency of Amplitude Value", fontsize = 12)
        plt.legend(fontsize=11)


        plt.subplot(gs2[2])
        n,bins= np.histogram(samples[gauss][:,1],bins=100)
        peak_mu = 0.5*(bins[np.where(n == n.max())[0]]+(bins[1+np.where(n == n.max())[0]]))
        
        print("\n\tPEAK value of mu:",peak_mu[0])
        print("\tMEAN value of mu: %s" % np.mean(samples[gauss][:,1]))
        print("\tSTD. DEV value of mu: %s" % np.std(samples[gauss][:,1]))
        
        text = "PEAK value of mu: %.2f \nMEAN value of mu: %.2f \nSTD. DEV value of mu: %.2f" % (peak_mu[0],np.mean(samples[gauss][:,1]),np.std(samples[gauss][:,1]))
        plt.hist(samples[gauss][:,1],bins=100, label = text)
        plt.xlabel("Mu", fontsize = 12)
        plt.ylabel("Frequency of Mu Value", fontsize = 12)
        plt.legend(fontsize=11)

        plt.subplot(gs2[4])
        n,bins= np.histogram(samples[gauss][:,2],bins=100)
        peak_sig = 0.5*(bins[np.where(n == n.max())[0]]+(bins[1+np.where(n == n.max())[0]]))
        print("\n\tPEAK value of sig:",peak_sig[0])
        print("\tMEAN value of sig: %s" % np.mean(samples[gauss][:,2]))
        print("\tSTD. DEV value of sig: %s" % np.std(samples[gauss][:,2]))

        text = "PEAK value of sig: %.2f \nMEAN value of sig: %.2f \nSTD. DEV value of sig: %.2f" % (peak_sig[0],np.mean(samples[gauss][:,2]),np.std(samples[gauss][:,2]))
        
        plt.hist(samples[gauss][:,2],bins=100,label = text)
        plt.ylabel("Frequency of Sigma Value", fontsize = 12)
        plt.xlabel("Sigma", fontsize = 12)
        plt.legend(fontsize=11)

        plt.subplot(gs2[1])
        plt.title("Value of Parameter at Each Iteration", fontsize = 14)
        plt.plot(samples[gauss][:,0],np.arange(0,R))
        plt.xlabel("Amplitude Value", fontsize = 12)
        plt.ylabel("Iteration", fontsize = 12)
        
        plt.subplot(gs2[3])
        plt.plot(samples[gauss][:,1],np.arange(0,R))
        plt.xlabel("Mu Value", fontsize = 12)
        plt.ylabel("Iteration", fontsize = 12)

        plt.subplot(gs2[5])
        plt.plot(samples[gauss][:,2],np.arange(0,R))
        plt.xlabel("Sigma Value", fontsize = 12)
        plt.ylabel("Iteration", fontsize = 12)

        
        plt.tight_layout()

        if SAVE_FIG:
            plt.savefig("./gauss_images/%s_gaussians/%s_samples/%s_dist_gaussNum_%s_%s_noise.jpg" % (NUM_GAUSS,R,method,gauss,NOISE_BOOL_str), dpi = 300)
        if PLOT_SHOW:
            plt.show()

        
    print("METHAST TIME TAKEN TO RUN = %.2f" % (end-start))
    print("METHAST TIME TAKEN TO RUN PER GAUSS = %.2f" % ((end-start)/NUM_GAUSS))
main()
