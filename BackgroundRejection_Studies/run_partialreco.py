#!/usr/bin/env python3
"""Wrapper Script to run selection checks on Fully recon. samples (EventCalc)."""

#-----------------------------------------------------------------------------------------------------------
from selectioncheck import main
import sys, argparse
import os,ROOT
import pandas as pd

U2_real = {
    'e': 0.447e-9,
    'mu': 7.15e-9,
    'tau': 1.88e-9,
}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("-p", "--path", default="/eos/experiment/ship/user/anupamar/Signal_EventCalc/2muv/12946374/HNL_1.000e+00_7.133e+04_4.808e-02_7.692e-01_1.827e-01",help="Path to simulation file") #mumunu 1 GeV

known, rest = p.parse_known_args(sys.argv[1:])

def define_weight_EventCalc(event,path=known.path):
	
	w_event=event.MCTrack[0].GetWeight()

	eventcalcdata_path='/eos/experiment/ship/user/anupamar/EventCalc_data/FairShip_benchmarkcoupling/'

	channel=path.split('/')[-3] 
	foldername=os.path.basename(path)
	
	
	masstag = float(foldername.split('_')[1])

	#print(f"channel:\t{channel},\tmass:\t{masstag}")

	datfile=f'{eventcalcdata_path}/{channel}_sample/HNL/total/HNL_4.808e-02_7.692e-01_1.827e-01_total.txt'
	inputfile=f'{eventcalcdata_path}/rootfiles/{channel}/{foldername}_data.root'
	
	#print(f"{path}\n\t should match \n {inputfile}\n\t does it?")

	file = ROOT.TFile.Open(inputfile)    
	tree=file.Get("LLP_tree")

	#print("nEntries",tree.GetEntries())
	n_events=tree.GetEntries()

	df = pd.read_csv(datfile, delim_whitespace=True)

	row = df.loc[df['mass'] == masstag]

	if row.empty:
	    raise ValueError(f"No entry for HNL mass = {masstag} in {datfile} EVENT WEIGHTS ARE AN ISSUE")

	row = row.iloc[0]

	# Extract factors
	N_LLP_tot      = row['N_LLP_tot']
	eps_pol        = row['epsilon_polar']
	eps_azi        = row['epsilon_azimuthal']
	BR_vis         = row['Br_visible']
	U2_gen_total   = row['coupling_squared']     # total coupling² used in your 1:1:1 sample
	U2_gen_per_flav = U2_gen_total / 3.0          # because sample was e:mu:tau = 1:1:1

	# 3) compute scale factors for e‐flavour and mu‐flavour
	scale_e  = U2_real['e']  / U2_gen_per_flav
	scale_mu = U2_real['mu'] / U2_gen_per_flav

	base = N_LLP_tot * eps_pol * eps_azi * BR_vis

	# 5) apply flavour‐specific scaling
	if 'e' in channel:
		expected   = base * scale_e * (w_event / n_events)
	if 'mu' in channel:
		expected   = base * scale_mu * (w_event / n_events)

	return expected


# Pass the parsed path plus any *remaining* CLI args to the core.
sys.argv = [sys.argv[0], *rest, "-p", known.path]
main(weight_function=define_weight_EventCalc)