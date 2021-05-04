
# coding: utf-8

# In[ ]:

import argparse
import ast
import logging
import math
import os
from array import array

import numpy as np
import scipy
from scipy import stats
from scipy.special import comb

import ROOT
from dihiggs_dataset import DiHiggsSignalMCDataset
from ROOT import TH1F, TCanvas, TGraph, TLorentzVector
    

PT_INDEX = 0 # index of transverse momentum in event data
ETA_INDEX = 1 # index of pseudorapidity in event data
PHI_INDEX = 2 # index of phi in event data
MASS_INDEX = 3 # index of mass in event data
BTAG_INDEX = 4 # index of b-tag score in event data
FEATURES_PER_JET = 5
JETS_PER_EVENT = 5

# Below lists are used for plotting purposes
FEATURE_NAME_LIST = ['pts', 'etas', 'phis', 'masses', 'btags']
MOMENTUMS = ['pxs', 'pys', 'pzs']
DELTA_RS = ['dr12', 'dr13', 'dr14', 'dr15', 'dr23', 'dr24', 'dr25', 'dr34', 'dr35', 'dr45']
DELTA_R_HIGGS = ['dr12']
HIGGS_FEATURES = [['lead_pts', 'lead_masses'], ['subl_pts', 'subl_masses'], ['dihiggs_etas', 'dihiggs_masses']]

def process_output_and_metric(configDir, mcdir = '/depot/darkmatter/apps/awildrid/data'):


    #is_plotting = args.plot
    #plotting_epochs = args.epochs
    test_path = configDir
    mc_path = mcdir
    # print("Plotting: " + str(is_plotting))
    # print("The epochs to plot: " + str(plotting_epochs))
    # print("The test directory: " + str(test_path))
    # print("The MC directory: " + str(mc_path))

    mc_data = DiHiggsSignalMCDataset(mc_path, download=False, generator_level=False, normalize=True)

    try:
        os.makedirs(os.path.join(test_path, 'InputLevel'))
        os.makedirs(os.path.join(test_path, 'PXYZ'))
        os.makedirs(os.path.join(test_path, 'Jet_DeltaR'))
        os.makedirs(os.path.join(test_path, 'Higgs'))
    except OSError:
        logging.warning("Output folders already exist. May overwrite some output files.")

    all_files = os.listdir(test_path)   # TODO: Make sure this matches with where stuff is saved
    gen_datas = [filename for filename in all_files if filename.endswith(".npy")]
    names = np.array([gen_data.split('_')[0] for gen_data in gen_datas])
    names = np.array([name[:-3] + 'k' if len(name) >= 4 else '0k' for name in names]) # only grab thousands prefix: 90000 -> 90k
    
    # remove data that has been read to save space
    for filename in all_files:
        if filename.endswith(".npy"):
            os.remove(filename)
            
    epochs = np.array([int(name[:-1]) for name in names])
    epoch_order = epochs.argsort()
    epochs = epochs[epoch_order]

    # plot_data_idxs = []
    # if plotting_epochs is None:
    #     plot_data_idxs = [names.tolist().index(str(epochs[len(gen_datas) - 1]) + 'k'), names.tolist().index(str(epochs[int(len(gen_datas) / 2)]) + 'k'), names.tolist().index(str(epochs[4]) + 'k')]
    # else:
    #     plot_data_idxs = [gen_datas.index(filename) for filename in gen_datas if (filename.split('_')[0] in plotting_epochs)]

    gen_events = [np.load(test_path + '/' + gen_data) for gen_data in gen_datas]

    generated_pts, generated_etas, generated_phis, generated_masses, generated_btags = scale_data(gen_events, mc_data)
    mc_pts, mc_etas, mc_phis, mc_masses, mc_btags = scale_data([mc_data], mc_data)

    pxs, pys, pzs = calc_pxyz(generated_pts, generated_etas, generated_phis)
    mc_pxs, mc_pys, mc_pzs = calc_pxyz(mc_pts, mc_etas, mc_phis)    

    drs = calc_drs(generated_etas, generated_phis)
    mc_drs = calc_drs(mc_etas, mc_phis)

    gen_lead_higgs_bosons, gen_subl_higgs_bosons, gen_dihiggs, gen_deltaR_higgs = reconstruct_higgs(generated_pts, generated_etas, generated_phis, generated_masses, generated_btags)
    mc_lead_higgs_bosons, mc_subl_higgs_bosons, mc_dihiggs, mc_deltaR_higgs = reconstruct_higgs(mc_pts, mc_etas, mc_phis, mc_masses, mc_btags)


    # Calculate and plot all of the Kolmogorov-Smirnov distances
    epochs = epochs.astype("float64")

    folder = 'InputLevel'
    pts_ks_values = np.array([scipy.stats.ks_2samp(generated_pts[i], mc_pts[0])[0] for i in range(len(generated_pts))], dtype="float64")[epoch_order]
    etas_ks_values = np.array([scipy.stats.ks_2samp(generated_etas[i], mc_etas[0])[0] for i in range(len(generated_etas))], dtype="float64")[epoch_order]
    phis_ks_values = np.array([scipy.stats.ks_2samp(generated_phis[i], mc_phis[0])[0] for i in range(len(generated_phis))], dtype="float64")[epoch_order]
    masses_ks_values = np.array([scipy.stats.ks_2samp(generated_masses[i], mc_masses[0])[0] for i in range(len(generated_masses))], dtype="float64")[epoch_order]
    btags_ks_values = np.array([scipy.stats.ks_2samp(generated_btags[i], mc_btags[0])[0] for i in range(len(generated_btags))], dtype="float64")[epoch_order]

    np.save(os.path.join(test_path, folder, 'pts_ks_values.npy'), pts_ks_values)
    np.save(os.path.join(test_path, folder, 'etas_ks_values.npy'), etas_ks_values)
    np.save(os.path.join(test_path, folder, 'phis_ks_values.npy'), phis_ks_values)
    np.save(os.path.join(test_path, folder, 'masses_ks_values.npy'), masses_ks_values)
    np.save(os.path.join(test_path, folder, 'btags_ks_values.npy'), btags_ks_values)
    ####################################################################

    pxs_ks_values = np.array([scipy.stats.ks_2samp(pxs[i], mc_pxs[0])[0] for i in range(len(pxs))], dtype="float64")[epoch_order]
    pys_ks_values = np.array([scipy.stats.ks_2samp(pys[i], mc_pys[0])[0] for i in range(len(pys))], dtype="float64")[epoch_order]
    pzs_ks_values = np.array([scipy.stats.ks_2samp(pzs[i], mc_pzs[0])[0] for i in range(len(pzs))], dtype="float64")[epoch_order]
    folder = 'PXYZ'
    np.save(os.path.join(test_path, folder, 'pxs_ks_values.npy'), pxs_ks_values)
    np.save(os.path.join(test_path, folder, 'pxsy_ks_values.npy'), pys_ks_values)
    np.save(os.path.join(test_path, folder, 'pzs_ks_values.npy'), pzs_ks_values)


    dr12_ks_values = np.array([scipy.stats.ks_2samp(drs[0][i], mc_drs[0][0])[0] for i in range(len(drs[0]))], dtype="float64")[epoch_order]
    dr13_ks_values = np.array([scipy.stats.ks_2samp(drs[1][i], mc_drs[1][0])[0] for i in range(len(drs[1]))], dtype="float64")[epoch_order]
    dr14_ks_values = np.array([scipy.stats.ks_2samp(drs[2][i], mc_drs[2][0])[0] for i in range(len(drs[2]))], dtype="float64")[epoch_order]
    dr15_ks_values = np.array([scipy.stats.ks_2samp(drs[3][i], mc_drs[3][0])[0] for i in range(len(drs[3]))], dtype="float64")[epoch_order]
    dr23_ks_values = np.array([scipy.stats.ks_2samp(drs[4][i], mc_drs[4][0])[0] for i in range(len(drs[4]))], dtype="float64")[epoch_order]
    dr24_ks_values = np.array([scipy.stats.ks_2samp(drs[5][i], mc_drs[5][0])[0] for i in range(len(drs[5]))], dtype="float64")[epoch_order]
    dr25_ks_values = np.array([scipy.stats.ks_2samp(drs[6][i], mc_drs[6][0])[0] for i in range(len(drs[6]))], dtype="float64")[epoch_order]
    dr34_ks_values = np.array([scipy.stats.ks_2samp(drs[7][i], mc_drs[7][0])[0] for i in range(len(drs[7]))], dtype="float64")[epoch_order]
    dr35_ks_values = np.array([scipy.stats.ks_2samp(drs[8][i], mc_drs[8][0])[0] for i in range(len(drs[8]))], dtype="float64")[epoch_order]
    dr45_ks_values = np.array([scipy.stats.ks_2samp(drs[9][i], mc_drs[9][0])[0] for i in range(len(drs[9]))], dtype="float64")[epoch_order]
    folder = 'Jet_DeltaR'
    np.save(os.path.join(test_path, folder, 'dr12_ks_values.npy'), dr12_ks_values)
    np.save(os.path.join(test_path, folder, 'dr13_ks_values.npy'), dr13_ks_values)
    np.save(os.path.join(test_path, folder, 'dr14_ks_values.npy'), dr14_ks_values)
    np.save(os.path.join(test_path, folder, 'dr15_ks_values.npy'), dr15_ks_values)
    np.save(os.path.join(test_path, folder, 'dr23_ks_values.npy'), dr23_ks_values)
    np.save(os.path.join(test_path, folder, 'dr24_ks_values.npy'), dr24_ks_values)
    np.save(os.path.join(test_path, folder, 'dr25_ks_values.npy'), dr25_ks_values)
    np.save(os.path.join(test_path, folder, 'dr34_ks_values.npy'), dr34_ks_values)
    np.save(os.path.join(test_path, folder, 'dr35_ks_values.npy'), dr35_ks_values)
    np.save(os.path.join(test_path, folder, 'dr45_ks_values.npy'), dr45_ks_values)


    gen_lead_higgs_masses = [[gen_lead_higgs_bosons[j][k].M() for k in range(len(gen_lead_higgs_bosons[j]))] for j in range(len(gen_lead_higgs_bosons))]
    gen_subl_higgs_masses = [[gen_subl_higgs_bosons[j][k].M() for k in range(len(gen_subl_higgs_bosons[j]))] for j in range(len(gen_subl_higgs_bosons))]
    gen_dihiggs_masses = [[gen_dihiggs[j][k].M() for k in range(len(gen_dihiggs[j]))] for j in range(len(gen_dihiggs))]
    gen_lead_higgs_pts = [[gen_lead_higgs_bosons[j][k].Pt() for k in range(len(gen_lead_higgs_bosons[j]))] for j in range(len(gen_lead_higgs_bosons))]
    gen_subl_higgs_pts = [[gen_subl_higgs_bosons[j][k].Pt() for k in range(len(gen_subl_higgs_bosons[j]))] for j in range(len(gen_subl_higgs_bosons))]
    gen_dihiggs_etas = [[gen_dihiggs[j][k].Eta() for k in range(len(gen_dihiggs[j]))] for j in range(len(gen_dihiggs))]

    mc_lead_higgs_masses = [[mc_lead_higgs_bosons[j][k].M() for k in range(len(mc_lead_higgs_bosons[j]))] for j in range(len(mc_lead_higgs_bosons))]
    mc_subl_higgs_masses = [[mc_subl_higgs_bosons[j][k].M() for k in range(len(mc_subl_higgs_bosons[j]))] for j in range(len(mc_subl_higgs_bosons))]
    mc_dihiggs_masses = [[mc_dihiggs[j][k].M() for k in range(len(mc_dihiggs[j]))] for j in range(len(mc_dihiggs))]
    mc_lead_higgs_pts = [[mc_lead_higgs_bosons[j][k].Pt() for k in range(len(mc_lead_higgs_bosons[j]))] for j in range(len(mc_lead_higgs_bosons))]
    mc_subl_higgs_pts = [[mc_subl_higgs_bosons[j][k].Pt() for k in range(len(mc_subl_higgs_bosons[j]))] for j in range(len(mc_subl_higgs_bosons))]
    mc_dihiggs_etas = [[mc_dihiggs[j][k].Eta() for k in range(len(mc_dihiggs[j]))] for j in range(len(mc_dihiggs))]

    higgs_dr_ks_values = np.array([scipy.stats.ks_2samp(gen_deltaR_higgs[i], mc_deltaR_higgs[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_deltaR_higgs))], dtype="float64")[epoch_order]
    higgs_lead_m_ks_values = np.array([scipy.stats.ks_2samp(gen_lead_higgs_masses[i], mc_lead_higgs_masses[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_lead_higgs_masses))], dtype="float64")[epoch_order]
    higgs_subl_m_ks_values = np.array([scipy.stats.ks_2samp(gen_subl_higgs_masses[i], mc_subl_higgs_masses[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_subl_higgs_masses))], dtype="float64")[epoch_order]
    higgs_lead_pt_ks_values = np.array([scipy.stats.ks_2samp(gen_lead_higgs_pts[i], mc_lead_higgs_pts[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_lead_higgs_pts))], dtype="float64")[epoch_order]
    higgs_subl_pt_ks_values = np.array([scipy.stats.ks_2samp(gen_subl_higgs_pts[i], mc_subl_higgs_pts[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_subl_higgs_pts))], dtype="float64")[epoch_order]
    dihiggs_m_ks_values = np.array([scipy.stats.ks_2samp(gen_dihiggs_masses[i], mc_dihiggs_masses[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_dihiggs_masses))], dtype="float64")[epoch_order]
    dihiggs_eta_ks_values = np.array([scipy.stats.ks_2samp(gen_dihiggs_etas[i], mc_dihiggs_etas[0])[0] if len(gen_deltaR_higgs[i]) > 0 else 1.0 for i in range(len(gen_dihiggs_etas))], dtype="float64")[epoch_order]
    folder = 'Higgs'
    np.save(os.path.join(test_path, folder, 'higgs_dr_ks_values.npy'), higgs_dr_ks_values)
    np.save(os.path.join(test_path, folder, 'higgs_lead_m_ks_values.npy'), higgs_lead_m_ks_values)
    np.save(os.path.join(test_path, folder, 'higgs_subl_m_ks_values.npy'), higgs_subl_m_ks_values)
    np.save(os.path.join(test_path, folder, 'higgs_lead_pt_ks_values.npy'), higgs_lead_pt_ks_values)
    np.save(os.path.join(test_path, folder, 'higgs_subl_pt_ks_values.npy'), higgs_subl_pt_ks_values)
    np.save(os.path.join(test_path, folder, 'dihiggs_m_ks_values.npy'), dihiggs_m_ks_values)
    np.save(os.path.join(test_path, folder, 'dihiggs_eta_ks_values.npy'), dihiggs_eta_ks_values)
    
    # Calculate the metric for the GAN
    ten_percent = np.floor(0.1 * len(pts_ks_values))
    index = len(pts_ks_values) - ten_percent
    # slope_scaling = 0.5

    # pts_ks_length = np.mean(pts_ks_values[int(index):])
    # etas_ks_length = np.mean(etas_ks_values[int(index):])
    # phis_ks_length = np.mean(phis_ks_values[int(index):])
    # masses_ks_length = np.mean(masses_ks_values[int(index):])
    # btags_ks_length = np.mean(btags_ks_values[int(index):])

    higgs_dr_ks_length = np.mean(higgs_dr_ks_values[int(index):])
    higgs_lead_m_ks_length = np.mean(higgs_lead_m_ks_values[int(index):])
    higgs_subl_m_ks_length = np.mean(higgs_subl_m_ks_values[int(index):])
    higgs_lead_pt_ks_length = np.mean(higgs_lead_pt_ks_values[int(index):])
    higgs_subl_pt_ks_length = np.mean(higgs_subl_pt_ks_values[int(index):])
    dihiggs_m_ks_length = np.mean(dihiggs_m_ks_values[int(index):])
    dihiggs_eta_ks_length = np.mean(dihiggs_eta_ks_values[int(index):])

    # pts_ks_linear_data = stats.linregress(list(range(1,int(ten_percent+1))), pts_ks_values[int(index):])
    # etas_ks_linear_data = stats.linregress(list(range(1,int(ten_percent+1))), etas_ks_values[int(index):])
    # phis_ks_linear_data = stats.linregress(list(range(1,int(ten_percent+1))), phis_ks_values[int(index):])
    # masses_ks_linear_data = stats.linregress(list(range(1,int(ten_percent+1))), masses_ks_values[int(index):])
    # btags_ks_linear_data = stats.linregress(list(range(1,int(ten_percent+1))), btags_ks_values[int(index):])

    # pts_ks_weighted_slope = slope_scaling * pts_ks_linear_data[0]
    # etas_ks_weighted_slope = slope_scaling * etas_ks_linear_data[0]
    # phis_ks_weighted_slope = slope_scaling * phis_ks_linear_data[0]
    # masses_ks_weighted_slope = slope_scaling * masses_ks_linear_data[0]
    # btags_ks_weighted_slope = slope_scaling * btags_ks_linear_data[0]

    pts_metric = pts_ks_length # + pts_ks_weighted_slope
    etas_metric = etas_ks_length # + etas_ks_weighted_slope
    phis_metric = phis_ks_length # + phis_ks_weighted_slope
    masses_metric = masses_ks_length # + masses_ks_weighted_slope
    btags_metric = btags_ks_length # + btags_ks_weighted_slope

    higgs_dr_metric = higgs_dr_ks_length # + pts_ks_weighted_slope
    higgs_lead_m_metric = higgs_lead_m_ks_length # + etas_ks_weighted_slope
    higgs_subl_m_metric = higgs_subl_m_ks_length # + phis_ks_weighted_slope
    higgs_lead_pt_metric = higgs_lead_pt_ks_length # + masses_ks_weighted_slope
    higgs_subl_pt_metric = higgs_subl_pt_ks_length # + btags_ks_weighted_slope
    dihiggs_m_metric = dihiggs_m_ks_length # + masses_ks_weighted_slope
    dihiggs_eta_metric = dihiggs_eta_ks_length # + btags_ks_weighted_slope

   # \n",
    #WGAN_metric = pts_metric * etas_metric * phis_metric * masses_metric * btags_metric
    WGAN_metric = higgs_dr_metric * higgs_lead_m_metric * higgs_subl_m_metric * higgs_lead_pt_metric * higgs_subl_pt_metric * dihiggs_m_metric * dihiggs_eta_metric
    os.system("echo " + str(WGAN_metric) + " " + configDir + " >> metric_" + ".txt")
    
    return(WGAN_metric)





def scale_data(gen_datas, dataset):
    ''' Re-scales the data to be back to within its regular range. 

    Args:
        :param gen_datas: The generative-model-based dataset (GAN)
        :param dataset: The Monte Carlo-based dataset
    '''
    num_events = len(gen_datas[0][:,0::5])
    num_jets = int(dataset.n_features / FEATURES_PER_JET)

    all_pts = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_etas = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_phis = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_masses = np.array([], dtype=np.float).reshape(0, num_events * num_jets)
    all_btags = np.array([], dtype=np.float).reshape(0, num_events * num_jets)

    for events in gen_datas:
        pts = (events[:,PT_INDEX::num_jets] * dataset.pt_range) + dataset.min_pt
        etas = (events[:,ETA_INDEX::num_jets] * dataset.eta_range) + dataset.min_eta
        phis = (events[:,PHI_INDEX::num_jets] * dataset.phi_range) + dataset.min_phi
        masses = (events[:,MASS_INDEX::num_jets] * dataset.mass_range) + dataset.min_mass
        btags = (events[:,BTAG_INDEX::num_jets] * dataset.btag_range) + dataset.min_btags

        pts = np.reshape(pts, num_events * num_jets)
        etas = np.reshape(etas, num_events * num_jets)
        phis = np.reshape(phis, num_events * num_jets)
        masses = np.reshape(masses, num_events * num_jets)
        btags = np.reshape(btags, num_events * num_jets)

        all_pts = np.vstack((all_pts, pts))
        all_etas = np.vstack((all_etas, etas))
        all_phis = np.vstack((all_phis, phis))
        all_masses = np.vstack((all_masses, masses))
        all_btags = np.vstack((all_btags, btags))

    return all_pts, all_etas, all_phis, all_masses, all_btags



def calc_pxyz(all_pts, all_etas, all_phis):
    '''Calculates the x, y, and z coordinates of the 4-momentum.

    Args:
        :param all_pts: The collection of all transverse momentums being analyzed. Structure is K x M x 5
                        where K is the number of collections of events and M is the number of events. There
                        are 5 jets per event
        :param all_etas: The collection of all pseudorapidities. See transverse momentum for identical shape
        :param all_phis: The collection of all phis. See transverse momentum for identical shape

    Return:
        The x, y, and z coordinates of the 4-momentum
    '''
    all_pxs = all_pts * np.cos(all_phis)
    all_pys = all_pts * np.sin(all_phis)
    all_pzs = all_pts * np.sinh(all_etas)
    return all_pxs, all_pys, all_pzs


def calc_drs(all_etas, all_phis):
    ''' Calclates all of the different combinations of angular distances between a collection of all of the phis and etas.
    There are 10 combinations per event and does them per event for all collections of events. The distance metric being
    used is the L2 norm. 

    Args:
        :param all_etas: The collection of all pseudorapidites being analyzed. Structure is K x M x 5
                        where K is the number of collections of events and M is the number of events. There
                        are 5 jets per event.
        :param all_phis: The collection of all phis being analyzed. See pseudorapidities for  structure.

    Return:
        The L2 norm between the two (eta, phi) pairs'''
    num_jets = 5
    all_drs = np.array([], dtype=np.float).reshape(0, len(all_etas), int(len(all_etas[0])/5))
    first_indices, second_indices = np.triu_indices(num_jets, m=num_jets)
    for k in range(len(first_indices)):
        i = first_indices[k]
        j = second_indices[k]
        if i == j:
            continue
        drs = np.array([np.sqrt(np.power(all_etas[:,i::5] - all_etas[:,j::5], 2) + np.power(all_phis[:,i::5] - all_phis[:,j::5], 2))])
        all_drs = np.concatenate((all_drs[:], drs), axis=0)
    return all_drs


def reconstruct_higgs(all_pts, all_etas, all_phis, all_masses, all_btags):
    gan_pts = all_pts.reshape(len(all_pts), int(len(all_pts[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_etas = all_etas.reshape(len(all_etas), int(len(all_etas[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_phis = all_phis.reshape(len(all_phis), int(len(all_phis[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_masses = all_masses.reshape(len(all_masses), int(len(all_masses[0]) / JETS_PER_EVENT), JETS_PER_EVENT)
    gan_btags = all_btags.reshape(len(all_btags), int(len(all_btags[0]) / JETS_PER_EVENT), JETS_PER_EVENT)

    mask = (gan_pts > 30) & (np.absolute(gan_etas) < 2.4)
    gan_pts = np.where(mask, gan_pts, np.nan)
    gan_etas = np.where(mask, gan_etas, np.nan)
    gan_phis = np.where(mask, gan_phis, np.nan)
    gan_masses = np.where(mask, gan_masses, np.nan)
    gan_btags = np.where(mask, gan_btags, np.nan)

    is_b = gan_btags > 0.226
    has_4bs = np.count_nonzero(is_b, axis=2) >= 4
    nEventsWith4bs = np.count_nonzero(has_4bs, axis = 1)
    #print("Number of events with 4 b-tagged jets is " + str(nEventsWith4bs) + ' out of a total of ' + str(nEvents) + ' events.')
    #print('Percentage = ' + str(nEventsWith4bs/nEvents))

    gan_4b_pts = [gan_pts[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_etas = [gan_etas[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_phis = [gan_phis[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_masses = [gan_masses[i,has_4bs[i]] for i in range(len(has_4bs))]
    gan_4b_btags = [gan_btags[i,has_4bs[i]] for i in range(len(has_4bs))]

    lead_higgs_list = np.empty((len(has_4bs),),dtype=object)
    subl_higgs_list = np.empty((len(has_4bs),),dtype=object)
    dihiggs_list = np.empty((len(has_4bs),),dtype=object)
    delta_r_higgs_list = np.empty((len(has_4bs),),dtype=object)

    for i in range(len(has_4bs)):
        sorted_indices = np.array([gan_4b_btags[i][j].argsort()[::-1] for j in range(len(gan_4b_btags[i]))])
        gan_4b_pts[i] = np.array([gan_4b_pts[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
        gan_4b_etas[i] = np.array([gan_4b_etas[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
        gan_4b_phis[i] = np.array([gan_4b_phis[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])
        gan_4b_masses[i] = np.array([gan_4b_masses[i][j][sorted_indices[j]][1:] for j in range(len(sorted_indices))])

        lead_higgs_bosons = []
        subl_higgs_bosons = []
        delta_r_higgs = []
        dihiggses = []
        for j in range(len(gan_4b_pts[i])):
            jet1 = TLorentzVector()
            jet2 = TLorentzVector()
            jet3 = TLorentzVector()
            jet4 = TLorentzVector()
            jet1.SetPtEtaPhiM(gan_4b_pts[i][j][0], gan_4b_etas[i][j][0], gan_4b_phis[i][j][0], gan_4b_masses[i][j][0])
            jet2.SetPtEtaPhiM(gan_4b_pts[i][j][1], gan_4b_etas[i][j][1], gan_4b_phis[i][j][1], gan_4b_masses[i][j][1])
            jet3.SetPtEtaPhiM(gan_4b_pts[i][j][2], gan_4b_etas[i][j][2], gan_4b_phis[i][j][2], gan_4b_masses[i][j][2])
            jet4.SetPtEtaPhiM(gan_4b_pts[i][j][3], gan_4b_etas[i][j][3], gan_4b_phis[i][j][3], gan_4b_masses[i][j][3])

            higgs1 = TLorentzVector()
            higgs2 = TLorentzVector()
            mass_disc = 0
            jets_12_34_disc = abs((jet1 + jet2).M() - (jet3 + jet4).M())
            jets_13_24_disc = abs((jet1 + jet3).M() - (jet2 + jet4).M())
            jets_14_23_disc = abs((jet1 + jet4).M() - (jet2 + jet3).M())
            if mass_disc == 0 or jets_12_34_disc < mass_disc:
                mass_disc = jets_12_34_disc
                higgs1 = jet1 + jet2
                higgs2 = jet3 + jet4
            if jets_13_24_disc < mass_disc:
                mass_disc = jets_13_24_disc
                higgs1 = jet1 + jet3
                higgs2 = jet2 + jet4
            if jets_14_23_disc < mass_disc:
                mass_disc = jets_14_23_disc
                higgs1 = jet1 + jet4
                higgs2 = jet2 + jet3

            if higgs1.M() > higgs2.M():
                lead_higgs_bosons.append(higgs1)
                subl_higgs_bosons.append(higgs2)
            else:
                lead_higgs_bosons.append(higgs2)
                subl_higgs_bosons.append(higgs1)
            dihiggs = higgs1 + higgs2
            delta_r_higgs.append(higgs1.DeltaR(higgs2))
            dihiggses.append(dihiggs)

        lead_higgs_list[i] = lead_higgs_bosons
        subl_higgs_list[i] = subl_higgs_bosons
        dihiggs_list[i] = dihiggses
        delta_r_higgs_list[i] = delta_r_higgs
    return lead_higgs_list, subl_higgs_list, dihiggs_list, delta_r_higgs_list

