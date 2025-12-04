"""
Calculate the invariant mass for HNL->rho lepton channel using a rho constraint using files generated using Fairship and plot it with neuDIS dangerouns background

Derived from python HNLinvmass_neuDIS_updated.py , but now also automatically choses the correct solution depending on the event.

selection checks= selection + UBT + 10<IP<250
"""

import HNLinvmass_EventCalc as functions
from argparse import ArgumentParser
import ROOT
from tabulate import tabulate
pdg = ROOT.TDatabasePDG.Instance()
import pythia8_conf
pythia8_conf.addHNLtoROOT()
import shipunit as u
from rootpyPickler import Unpickler
import os
import pandas as pd
import math
#from experimental import analysis_toolkit
import warnings
import glob
import sys
import numpy as np
sys.path.insert(1, '/afs/cern.ch/user/a/anupamar/alt_v2/FairShip_Analysis/BackgroundRejection_Studies/') 
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*elementwise comparison.*"
) #tabulate error

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal

mass_list={ '11':0.511*0.001,
            '211':139.57039*0.001,
            '13':105.66*0.001}

def pick_pi0(sol1,sol2,lepton_rot, pion_rot, calc_pion0_rot,h_prior):
    """ lookup each solution in the prior histogram and choose the one with higher prior. """
    if ROOT.TMath.IsNaN(sol1):
    	return sol2
    if ROOT.TMath.IsNaN(sol2):
    	return sol1
    
    b1 = h_prior.FindBin(sol1); p1 = h_prior.GetBinContent(b1)
    b2 = h_prior.FindBin(sol2); p2 = h_prior.GetBinContent(b2)
    
    return sol1 if p1 > p2 else sol2

def dump(event,mom_threshold=0):

    headers=['#','particle','pdgcode','mother_id','Momentum [Px,Py,Pz] (GeV/c)','StartVertex[x,y,z] (m)','Process', 'GetWeight()', ]
    
    event_table=[]
    for trackNr,track in enumerate(event.MCTrack): 
        
        if track.GetP()/u.GeV < mom_threshold :  continue
        
        try: particlename=pdg.GetParticle(track.GetPdgCode()).GetName()
        except: particlename='----'

        event_table.append([trackNr,
                        particlename,
                        track.GetPdgCode(),
                        track.GetMotherId(),
                        f"[{track.GetPx()/u.GeV:7.3f},{track.GetPy()/u.GeV:7.3f},{track.GetPz()/u.GeV:7.3f}]",
                        f"[{track.GetStartX()/u.m:7.3f},{track.GetStartY()/u.m:7.3f},{track.GetStartZ()/u.m:7.3f}]",
                        track.GetProcName().Data(),
                        track.GetWeight()
                        ])
    
    print(tabulate(event_table,headers=headers,floatfmt=".3f",tablefmt='simple_outline'))

def pid_decision(event,candidate):
    
    """
    Interim solution for PID check:Uses track truth info assuming 100% efficiency in CaloPID
    

    pid_code: 0     = hadronic,
              1     = dileptonic(any leptons),
              1.1   = dileptonic ee,
              1.2   = dileptonic μμ,         
              2     = semileptonic(any lepton),
              2.1   = semileptonic containing an e,
              2.2   = semileptonic containing a μ,
              3     = at least one track has unknown PID, #fallback, but never used since truth is used
              4     = fewer than two PID tracks available #two track candidates

    """

    if(len(event.Pid)<2):
        print("Pid is less  than 2 particles!") #sanity check maintained for historical reasons.
        return 4

    
    d1_mc=event.MCTrack[event.fitTrack2MC[candidate.GetDaughter(0)]]
    d1_pdg=d1_mc.GetPdgCode()

    d2_mc=event.MCTrack[event.fitTrack2MC[candidate.GetDaughter(1)]]
    d2_pdg=d2_mc.GetPdgCode()
    

    LEPTON_PDGS = {11, 13}      # 11 = electron, 13 = muon

    d1_is_lepton = abs(d1_pdg) in LEPTON_PDGS
    d2_is_lepton = abs(d2_pdg) in LEPTON_PDGS

    d1_is_mu= (abs(d1_pdg)==13)
    d2_is_mu= (abs(d2_pdg)==13)
    
    d1_is_e= (abs(d1_pdg)==11)
    d2_is_e= (abs(d2_pdg)==11)            

    # ---------- 4. decide on the return code ----------------------------------
    if d1_is_lepton and d2_is_lepton:                       # dileptonic final state
        if d1_is_e and d2_is_e:
            return 1.1
        if d1_is_mu and d2_is_mu:
            return 1.2
        return 1

    if d1_is_lepton or d2_is_lepton:                        # semileptonic
        if (d1_is_e or d2_is_e):
            return 2.1
        if (d1_is_mu or d2_is_mu):
            return 2.2
        return 2

    return 0 

def bookhistograms():

	h={}
	h["pion0_pp_truth_vs_chosensol"]       = ROOT.TH2D("pion0_pp_truth_vs_chosensol",    " ;truth  p_{\\parallel} (GeV/c)  ;chosen calc  p_{\\parallel} (sol1)(GeV/c)", 300, 0, 500, 500, 0, 500)
	h["pion0_pp_truth_vs_calcsol1"]        = ROOT.TH2D("pion0_pp_truth_vs_calcsol1",    " ;truth  p_{\\parallel} (GeV/c)  ;calc  p_{\\parallel} (sol1)(GeV/c)", 300, 0, 500, 500, 0, 500)
	h["pion0_pp_truth_vs_calcsol2"]        = ROOT.TH2D("pion0_pp_truth_vs_calcsol2",    " ;truth  p_{\\parallel} (GeV/c)  ;calc  p_{\\parallel} (sol2)(GeV/c)", 300, 0, 500, 500, 0, 500)

	h["truth_rho_M"]           = ROOT.TH1D("mass_rho",        	" ; m_{#rho} (GeV/c^{2});" 	, 100, 0.3, 1.3)
	h["HNL_decayvertex_z"]     = ROOT.TH1D("HNL_decayvertex_z", " ; z (cm);"                 	, 1000, -3000, 3000)
	h["HNL_prodvertex_z"]      = ROOT.TH1D("HNL_prodvertex_z",  " ; z (cm);"                 	, 146, -5888, -5742)

	h["truth_pion0_pt"]        = ROOT.TH1D("truth_pion0_pt",    " ; p_{T} (GeV/c);", 100, 0, 2)
	h["calc_pion0_pt"]         = ROOT.TH1D("calc_pion0_pt",     " ; p_{T} (GeV/c);", 100, 0, 2)

	h["truth_pion0_P_Mom"]     = ROOT.TH1D("truth_pion0_P_Mom", " ;p (GeV/c)                 ;", 300, -1500, 1500)
	h["truth_pion0_pp"]        = ROOT.TH1D("truth_pion0_pp",    " ;p_{\\parallel} (GeV/c)  ;", 300, -1500, 1500)
	h["sol1_calc_pion0_pp"]    = ROOT.TH1D("sol1_calc_pion0_pp"," ;p_{\\parallel} (GeV/c)  ;", 300, -1500, 1500)
	h["sol2_calc_pion0_pp"]    = ROOT.TH1D("sol2_calc_pion0_pp"," ;P_{\\parallel} (GeV/c)  ;", 300, -1500, 1500)

	h["calcmassdist"]         = ROOT.TH1D("calcmassdist",      " (#rho-constraint mass)    	;GeV/c^{2};", 100, 0, 2.5)
	
	h["calcmassdist1"]         = ROOT.TH1D("calcmassdist1",      " (#rho-constraint mass)    	;GeV/c^{2};", 100, 0, 2.5)
	h["calcmassdist2"]         = ROOT.TH1D("calcmassdist2",      " (#rho-constraint mass)    	;GeV/c^{2};", 100, 0, 2.5)
	h["invmassdist"]           = ROOT.TH1D("invmassdist",        " (M_{inv}(#it{l} #pi))	;GeV/c^{2};", 100, 0, 2.5)
	h["truemassdist"]          = ROOT.TH1D("truemassdist",       " true mass                 	;GeV/c^{2};", 100, 0, 2.5)
	h["IP"]          		   = ROOT.TH1D("impact_parameter",       " impact_parameter                 	;cm;", 500, 0, 500)
	h["HNL_decayvertex_xy"]    = ROOT.TH2D("HNL_decayvertex_xy", " ; x (cm); y (cm)", 1200, -600, 600, 1200, -600, 600)

	return h

def ip_category(ip_elem): 
	"""Return which sub-sample the event belongs to.""" 
	if ip_elem.startswith(("LiSc", "VetoInnerWall", "VetoOuterWall","VetoVerticalRib","VetoLongitRib")): 
		return "vesselCase" 
	if ip_elem.startswith("DecayVacuum"): 
		return "heliumCase" 
	return "all"    

def helium_rhoL(event, track_index=0,
                rho_he=1.78e-4, tol=6e-5, eps=1e-4, max_hops=256):
    """
    Return helium-only rho*L (g/cm^2) for a DIS vertex in any DV shape.
    Uses TGeoNavigator to follow the track in both directions from the vertex.
    """

    nav = ROOT.gGeoManager.GetCurrentNavigator()
    if not nav:
        raise RuntimeError("No TGeoNavigator; load geometry first.")

    # --- get vertex & direction ---
    v = ROOT.TVector3()
    event.MCTrack[track_index].GetStartVertex(v)
    xv, yv, zv = v.X(), v.Y(), v.Z()

    px = event.MCTrack[track_index].GetPx()
    py = event.MCTrack[track_index].GetPy()
    pz = event.MCTrack[track_index].GetPz()
    pm = math.sqrt(px*px + py*py + pz*pz) or 1.0
    dx, dy, dz = px/pm, py/pm, pz/pm

    # --- helper: is this point in helium? ---
    def in_helium(x, y, z):
        nav.SetCurrentPoint(x, y, z)
        nav.FindNode()
        node = nav.GetCurrentNode()
        if not node: return False
        try:
            rho = node.GetMedium().GetMaterial().GetDensity()
        except Exception:
            return False
        return abs(rho - rho_he) <= tol

    # quick check that vertex is in helium (nudge if needed)
    if not (in_helium(xv, yv, zv) or
            in_helium(xv + dx*eps, yv + dy*eps, zv + dz*eps) or
            in_helium(xv - dx*eps, yv - dy*eps, zv - dz*eps)):
        return 0.0

    # --- helper: integrate helium length from point along direction ---
    def helium_len_from(x0, y0, z0, dx, dy, dz):
        nav.SetCurrentPoint(x0 + dx*eps, y0 + dy*eps, z0 + dz*eps)
        nav.SetCurrentDirection(dx, dy, dz)
        nav.FindNode()
        total, seen_he = 0.0, False
        for _ in range(max_hops):
            node = nav.GetCurrentNode()
            if not node: break
            try:
                rho = node.GetMedium().GetMaterial().GetDensity()
            except Exception:
                rho = -1
            in_he = abs(rho - rho_he) <= tol
            nav.FindNextBoundaryAndStep()
            step = nav.GetStep()
            if in_he:
                total += step
                seen_he = True
            elif seen_he:
                break
            if nav.IsOutside(): break
            cp = nav.GetCurrentPoint()
            nav.SetCurrentPoint(cp[0] + dx*eps, cp[1] + dy*eps, cp[2] + dz*eps)
        return total

    # --- integrate forward + backward from vertex ---
    L_fwd = helium_len_from(xv, yv, zv,  dx,  dy,  dz)
    L_bwd = helium_len_from(xv, yv, zv, -dx, -dy, -dz)
    L_he  = L_fwd + L_bwd

    return rho_he * L_he  # g/cm^2

def calc_eventweight(event,path):
	
	def define_weight_neuDIS(event,SHiP_running=15,N_gen=6000*19969):
	    
	    interaction_point = ROOT.TVector3()
	    event.MCTrack[0].GetStartVertex(interaction_point)
	    
	    try:
	        node  = ROOT.gGeoManager.FindNode(interaction_point.X(),
	                                          interaction_point.Y(),
	                                          interaction_point.Z())
	        ip_elem = node.GetVolume().GetName()
	    except Exception:
	        ip_elem = ""                 # falls back to global-only

	    cat = ip_category(ip_elem)       # "all", "vesselCase", "heliumCase"
	    
	    #-----------------------------------------------------------------
	    
	    if cat=='heliumCase':

	    	w_DIS = helium_rhoL(event)

	    else:

	    	w_DIS    =  event.MCTrack[0].GetWeight()

	    nPOTinteraction     =(2.e+20)*(SHiP_running/5)
	    nPOTinteraction_perspill =5.e+13

	    n_Spill  = nPOTinteraction/nPOTinteraction_perspill #number of spill in SHiP_running(default=15) years
	    
	    nNu_perspill=4.51e+11       #number of neutrinos in a spill.
	    
	    N_nu=nNu_perspill*n_Spill   #Expected number of neutrinos in 15 years

	    w_nu=nNu_perspill/N_gen     #weight of each neutrino considered scaled to a spill such that sum(w_nu)=(nNu_perspill/N_gen)*N_gen= nNu_perspill = number of neutrinos in a spill.
	    
	    N_A=6.022*10**23
	    E_avg=2.57 #GeV
	    sigma_DIS=7*(10**-39)*E_avg*N_A  #cross section cm^2 per mole
	    
	    return w_DIS*sigma_DIS*w_nu*n_Spill  #(rho_L*N_nu*N_A*neu_crosssection*E_avg)/N_gen     #returns the number of the DIS interaction events of that type in SHiP running(default=5) years.   #DIS_multiplicity=1 here

	def define_weight_EventCalc(path):
		
		w_event=event.MCTrack[0].GetWeight()

		eventcalcdata_path='/eos/experiment/ship/user/anupamar/EventCalc_data/FairShip_benchmarkcoupling'

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

		base = N_LLP_tot * eps_pol * eps_azi * BR_vis

		# 5) apply flavour‐specific scaling
		if 'e' in channel:
			expected   = base * (w_event / n_events)
		if 'mu' in channel:
			expected   = base * (w_event / n_events)

		return expected

	def define_weight_muonDIS(event,SHiP_running=15):
	    """Calculate event weight in 15 years."""    
	    
	    w_mu=event.MCTrack[0].GetWeight()  #weight of the incoming muon*DIS multiplicity normalised to a full spill   sum(w_mu) = nMuons_perspill = number of muons in a spill. w_mu is not the same as N_muperspill/N_gen, where N_gen = nEvents*DISmultiplicity ( events enhanced in Pythia to increase statistics) .

	    cross=event.CrossSection
	    
	    interaction_point = ROOT.TVector3()
	    event.MCTrack[0].GetStartVertex(interaction_point)
	    try:
	        node  = ROOT.gGeoManager.FindNode(interaction_point.X(),
	                                          interaction_point.Y(),
	                                          interaction_point.Z())
	        ip_elem = node.GetVolume().GetName()
	    except Exception:
	        ip_elem = ""                 # falls back to global-only

	    cat = ip_category(ip_elem)       # "all", "vesselCase", "heliumCase"
	    # -----------------------------------------------------------------

	    if cat=='heliumCase':
	    	rho_l = helium_rhoL(event)
	    else:
	    	rho_l = event.MCTrack[2].GetWeight()
	    
	    N_a=6.022e+23 

	    sigma_DIS=cross*1e-27*N_a #cross section cm^2 per mole
	    
	    nPOTinteraction     =(2.e+20)*(SHiP_running/5) #in years
	    nPOTinteraction_perspill =5.e+13
	    
	    n_Spill  = nPOTinteraction/nPOTinteraction_perspill  #Number of Spills in SHiP running( default=5) years  
	        
	    weight_i = rho_l*sigma_DIS*w_mu*n_Spill 

	    return weight_i    

	if 'neuDIS' in path:
		
		return define_weight_neuDIS(event)
	if 'MuonDIS' in path:
		
		return define_weight_muonDIS(event)
	if 'Signal_EventCalc' in path:
		return define_weight_EventCalc(path)

def define_time_till_vtx(event):
	
	Mom = event.MCTrack[0].GetP()/u.GeV
	mass = event.MCTrack[0].GetMass()

	v= u.c_light*Mom/np.sqrt(Mom**2+(mass)**2)
	
	Target_Point = ROOT.TVector3(0., 0., -5814.25)  # production point
	Decay_Vertex = ROOT.TVector3(event.MCTrack[0].GetStartX(),  event.MCTrack[0].GetStartY(),  event.MCTrack[0].GetStartZ())  
	
	r_vec = Decay_Vertex - Target_Point          
	dist_from_target     = r_vec.Mag()              # cm   
	
	t_ns  = dist_from_target / v
	# return the time taken for the particle to reach the X,YZ from the start point
	return t_ns

def process_data(parent_path,h_prior,lepton_pdg_,check_PDG,check_SBT,check_UBT):
	
	def process_candidate(signal,event,event_weight):

		gamma_id=[]

		for i,track in enumerate(event.MCTrack):

			if track.GetMotherId()!=0: continue

			if abs(track.GetPdgCode())==22: 
			   	gamma_id.append(i)

		# Retrieve daughter tracks and masses
		
		d1_rec = event.FitTracks[signal.GetDaughter(0)]
		d1_mc  = event.MCTrack[event.fitTrack2MC[signal.GetDaughter(0)]]

		d2_rec = event.FitTracks[signal.GetDaughter(1)]
		d2_mc  = event.MCTrack[event.fitTrack2MC[signal.GetDaughter(1)]]

		lepton, pion = None, None
		
		if check_PDG:			

			for d_,d_mc in list(zip([d1_rec, d2_rec],[d1_mc, d2_mc])):
			    
			    d_pdg = d_mc.GetPdgCode() # ONLY TRUTH otherwise d_pdg = d_.getFittedState().getPDG()

			    if abs(d_pdg) in [13,11]: 
			    	lepton = d_
			    	#lepton_pdg=d_pdg
			    
			    if abs(d_pdg) == 211: 
			    	pion = d_	
			    	#pion_pdg=d_pdg

		else:	

			lepton=d1_rec
			pion=d2_rec
			#lepton_pdg,pion_pdg=None,None

		if not (lepton and pion):
		    return 

		if check_PDG:	

			pid=pid_decision(event,signal)

			if lepton_pdg_== '11' and not (pid==2.1 or int(pid)==3):
				return #False
			if lepton_pdg_== '13' and not (pid==2.2 or int(pid)==3): 
				return #False

		lepton_mass=mass_list[lepton_pdg_]
		pion_mass=mass_list['211']

		production_vertex = ROOT.TVector3(0, 0,  -5814.25)  # production point

		h["HNL_prodvertex_z"].Fill( -5814.25,event_weight)
		
		candidatePos = ROOT.TLorentzVector()
		signal.ProductionVertex(candidatePos)
		decay_vertex = ROOT.TVector3(candidatePos.X(), candidatePos.Y(), candidatePos.Z())

		h["HNL_decayvertex_z" ].Fill(candidatePos.Z(),event_weight)
		h["HNL_decayvertex_xy"].Fill(candidatePos.X(),candidatePos.Y(),event_weight)


		# Prepare momentum 4-vectors
		lepton_vec = ROOT.TLorentzVector()
		lepton_vec.SetPtEtaPhiM(lepton.getFittedState().getMom().Pt(),
		                        lepton.getFittedState().getMom().Eta(),
		                        lepton.getFittedState().getMom().Phi(),
		                        lepton_mass
		                        )

		lepton_rotated  =functions.rotate_momentum_to_hnl_frame(decay_vertex, production_vertex, lepton_vec)

		
		pion_vec = ROOT.TLorentzVector()
		pion_vec.SetPtEtaPhiM(  pion.getFittedState().getMom().Pt(),
		                        pion.getFittedState().getMom().Eta(),
		                        pion.getFittedState().getMom().Phi(),
		                        pion_mass
		                        )


		pion_rotated    =functions.rotate_momentum_to_hnl_frame(decay_vertex, production_vertex, pion_vec)

		
		if len(gamma_id)==2: # if there is pi0 decay in the daughters

			gamma1 = ROOT.TLorentzVector()
			gamma2 = ROOT.TLorentzVector()


			gamma1.SetPxPyPzE( 	event.MCTrack[gamma_id[0]].GetPx(),
								event.MCTrack[gamma_id[0]].GetPy(),
			            		event.MCTrack[gamma_id[0]].GetPz(),
			            		event.MCTrack[gamma_id[0]].GetEnergy()
			                  )

			gamma2.SetPxPyPzE( 	event.MCTrack[gamma_id[1]].GetPx(),
								event.MCTrack[gamma_id[1]].GetPy(),
			            		event.MCTrack[gamma_id[1]].GetPz(),
			            		event.MCTrack[gamma_id[1]].GetEnergy()
			                  )

			truth_pion0_vec = gamma1 + gamma2

			truth_rho=(truth_pion0_vec+pion_vec)

			h["truth_rho_M"].Fill(truth_rho.M(),event_weight)

			truth_pion0_new=functions.rotate_momentum_to_hnl_frame(decay_vertex, production_vertex, truth_pion0_vec) 

			h["truth_pion0_P_Mom"].Fill(truth_pion0_new.P(),event_weight)

			h["truth_pion0_pt"].Fill(truth_pion0_new.Pt(),event_weight)
			h["truth_pion0_pp"].Fill(truth_pion0_new.Pz(),event_weight)

			truemass = event.MCTrack[0].GetMass()
			h["truemassdist"].Fill(truemass,event_weight)


		calc_pion0_rotated = -(pion_rotated + lepton_rotated)

		pion0_pt_rotated = calc_pion0_rotated.Pt()
		h["calc_pion0_pt"].Fill(calc_pion0_rotated.Pt(),event_weight)          

		
		sol1_pion0_pp,sol2_pion0_pp 	= functions.calculate_pion0_parallel(lepton_rotated,pion_rotated,calc_pion0_rotated)


		if ROOT.TMath.IsNaN(sol1_pion0_pp) and ROOT.TMath.IsNaN(sol2_pion0_pp):
		    
		    rho_mass_min 				= functions.calculate_min_rho_mass(pion_rotated,lepton_rotated, calc_pion0_rotated)
		    rho_median   				= functions.rho_median_mass_from_min(rho_mass_min)
		    sol1_pion0_pp,sol2_pion0_pp = functions.calculate_pion0_parallel(lepton_rotated,pion_rotated,calc_pion0_rotated,rho_mass=rho_median)

		if not ROOT.TMath.IsNaN(sol1_pion0_pp):

			h["sol1_calc_pion0_pp"].Fill(sol1_pion0_pp,event_weight)
			HNLmass1 = functions.reconstruct_hnl_mass(sol1_pion0_pp,lepton_rotated, pion_rotated,calc_pion0_rotated)
			h["calcmassdist1"].Fill(HNLmass1,event_weight)

			if len(gamma_id)==2:
				h["pion0_pp_truth_vs_calcsol1"].Fill(truth_pion0_new.P(),sol1_pion0_pp)		

		
		if not ROOT.TMath.IsNaN(sol2_pion0_pp):
			h["sol2_calc_pion0_pp"].Fill(sol2_pion0_pp,event_weight)
			HNLmass2 = functions.reconstruct_hnl_mass(sol2_pion0_pp,lepton_rotated, pion_rotated,calc_pion0_rotated)
			h["calcmassdist2"].Fill(HNLmass2,event_weight)

			if len(gamma_id)==2:
				h["pion0_pp_truth_vs_calcsol2"].Fill(truth_pion0_new.P(),sol2_pion0_pp)


		if ROOT.TMath.IsNaN(sol1_pion0_pp) and ROOT.TMath.IsNaN(sol2_pion0_pp): 
			print("still negative radicands!")
			return

		
		chosen_p0z = pick_pi0(sol1_pion0_pp,sol2_pion0_pp,lepton_rotated, pion_rotated, calc_pion0_rotated,h_prior)
		if len(gamma_id)==2:
			h["pion0_pp_truth_vs_chosensol"].Fill(truth_pion0_new.P(),chosen_p0z)
		
		# Reconstruct exactly one HNL mass
		
		HNLmass = functions.reconstruct_hnl_mass(chosen_p0z,lepton_rotated,pion_rotated, calc_pion0_rotated)
		h["calcmassdist"].Fill(HNLmass,event_weight)

		inv_mass = signal.GetMass()
		h["invmassdist"].Fill(inv_mass,event_weight)
	
	h=bookhistograms()

	nReconstructed_candidates=0
	nSuccessful_candidates=0
	geo_file=None
	IP_cut = 250
	
	if 'MuonDIS' in parent_path:
		path_list=[f'{parent_path}/SBT',f'{parent_path}/Tr']
	else:
		path_list=[parent_path]

	for path in path_list:
		
		print("Processing data in: \n\t",path,"\n")

		for inputfileNr,inputFolder in enumerate(os.listdir(path)):

			if not os.path.isdir(f"{path}/{inputFolder}"):
				continue

			try:    
				
				if 'neuDIS' in path:
					file = ROOT.TFile.Open(f"{path}/{inputFolder}/events_withcandidates_filtered.root")
					geofile_path=f"{path}/geofile_full.conical.Genie-TGeant4.root"
				
				if 'MuonDIS' in path:
					file = ROOT.TFile.Open(f"{path}/{inputFolder}/ship.conical.muonDIS-TGeant4_rec.root")
					geofile_path="/eos/experiment/ship/simulation/bkg/MuonDIS_2024helium/8070735/SBT/job_99/geofile_full.conical.muonDIS-TGeant4.root"
				if 'Signal_EventCalc' in path:
					file = ROOT.TFile.Open(f"{path}/{inputFolder}/ship.conical.EvtCalc-TGeant4_rec.root")
					geofile_path=f"{path}/{inputFolder}/geofile_full.conical.EvtCalc-TGeant4.root"
				
				tree = file.cbmsim

			except:
				
				continue

			if args.testing_code and inputfileNr>1: break

			if geo_file==None:

				geo_file = ROOT.TFile.Open(geofile_path, "read")   

				import helperfunctions as analysis_toolkit #torch ROOT 6.32 crash workaround, import torch AFTER initialising ROOT

			ctx       = analysis_toolkit.AnalysisContext(tree, geo_file)

			selection = analysis_toolkit.selection_check(ctx)
			inspector = analysis_toolkit.event_inspector(ctx)
			veto_ship = analysis_toolkit.veto_tasks(ctx)

			for eventNr,event in enumerate(tree):

				if args.testing_code and nReconstructed_candidates>100: break
				
				#inspector.dump_event(track_p_threshold=0.5)
				
				event_weight= calc_eventweight(event,path)

				if not len(event.Particles): continue #only look at events with recon. candidate

				selection_pass={}

				for candidate_id_in_event,signal in enumerate(event.Particles):   
					
					nReconstructed_candidates += 1

					offset=define_time_till_vtx(event)
					
					preselection_flag = selection.preselection_cut(signal, IP_cut=IP_cut, show_table=False, offset=offset)
					
					if not preselection_flag: #only candidates surviving selection
						continue

					ip_val = selection.impact_parameter(signal)
					h["IP"].Fill(ip_val, event_weight)

					if ip_val < 10 * u.cm:
					    continue  # we want 10 <= IP < 250

					print(
					f"Event:{eventNr} Candidate_index: {candidate_id_in_event} <--passes the pre-selections\n\n"
					)
					
					if check_UBT:
						
						#if len(event.UpstreamTaggerPoint)>0: continue
					    UBT_veto, ubthits = veto_ship.UBT_decision()
					    selection_pass['UBT'] = not UBT_veto
					    if not selection_pass['UBT']:
					        continue  # vetoed by UBT
					
					else:
					    selection_pass['UBT'] = True  # effectively no UBT cut

					if check_SBT:

						xs, ys, zs,bestHits=[],[],[],[]

						AdvSBT90_veto,AdvSBT45_veto,AdvSBT0_veto=False,False,False

						track_index_first,track_index_last  =  signal.GetDaughter(0),signal.GetDaughter(1)

						for tr in [track_index_first,track_index_last]:

						    bestHit,xs_, ys_, zs_=veto_ship.extrapolateTrackToSBT(tr)
						    
						    xs.append(xs_)
						    ys.append(ys_)
						    zs.append(zs_)
						    if len(bestHit):
						        bestHits.extend(bestHit)

						for hit in bestHits:

						    AdvSBT0_veto=True
						    ELoss    = hit.GetEloss()
						    if ELoss>=90*0.001:
						        AdvSBT90_veto=True
						    if ELoss>=45*0.001:
						        AdvSBT45_veto=True

						selection_pass['AdvSBT@45MeV']   = not(AdvSBT45_veto) #Extrapolation SBT veto @45
						selection_pass['AdvSBT@90MeV']   = not(AdvSBT90_veto) #Extrapolation SBT veto @90

						reject, pBG = veto_ship.Veto_decision_GNNbinary_wdeltaT(threshold=0.6,offset=offset)
						selection_pass['GNNSBT@45MeV']   = not(reject) # specific GNN trained on neuDIS in He

						selection_pass['[ AdvSBT+GNNSBT ]@45MeV']   = selection_pass['AdvSBT@45MeV'] and selection_pass['GNNSBT@45MeV']

					else:

						selection_pass['[ AdvSBT+GNNSBT ]@45MeV']=True # effectively no SBT cut

					if selection_pass['[ AdvSBT+GNNSBT ]@45MeV'] and selection_pass['UBT']:

						process_candidate(signal,event,event_weight)

						nSuccessful_candidates += 1
			
			file.Close()
	
	geo_file.Close()

	print("nCandidates available:",nReconstructed_candidates,"\t nCandidates Successful=",nSuccessful_candidates)

	return h

def verifydata(path):
	
	print(f"Dumping Events of {path}")
	
	for inputFolder in os.listdir(path):
	
		try:    
			
			if 'neuDIS' in path:
				file = ROOT.TFile.Open(f"{path}/{inputFolder}/events_withcandidates_filtered.root")
			
			else:
				file = ROOT.TFile.Open(f"{path}/{inputFolder}/ship.conical.EvtCalc-TGeant4_rec.root")
			
			tree = file.cbmsim

		except:
			
			continue

		for eventNr,event in enumerate(tree):
			
			daughters=0
			for track in event.MCTrack:
				if track.GetMotherId()==0: 
					daughters+=1
			if daughters!=4:
				dump(event)
				dummy=input(f"Event{eventNr} has more than four daughters")

	

two_piepath 	= '/eos/experiment/ship/user/anupamar/Signal_EventCalc/2pie/12968088'
two_pimupath	= '/eos/experiment/ship/user/anupamar/Signal_EventCalc/2pimu/12968089'

epipath 		= '/eos/experiment/ship/user/anupamar/Signal_EventCalc/epi/12968094'
mupipath		= '/eos/experiment/ship/user/anupamar/Signal_EventCalc/mupi/12968090'

neuDISpath	= ['/eos/experiment/ship/user/anupamar/Filtered_neuDIS']
muonDISpath	= ['/eos/experiment/ship/simulation/bkg/MuonDIS_2024helium/8070735']

def main(lepton, mass, filename,PIDcheck,SBTcheck,UBTcheck):

	if lepton=='e':
		data_list=[epipath,two_piepath] #leptonrho and leptonpi samples
		lepton_pdg_='11'
		leptontag ='e'
	
	if lepton=='mu':
		data_list=[mupipath,two_pimupath]
		lepton_pdg_='13'
		leptontag= 'mu'

	print(f"\n\nRunning Analysis:{lepton}rho mass:{mass}, PIDcheck:{PIDcheck}\n\n")
	
	masstag=f"{float(mass):.3e}"
	
	prior_file = f"/afs/cern.ch/user/a/anupamar/alt_v2/FairShip_Analysis/HNL_leptonrho_massrecon/prior_hist/pi0priorhist_{leptontag}rho_{masstag}.root" #FIX THIS PATH LATER

	f_prior    = ROOT.TFile.Open(prior_file)
	h_prior    = f_prior.Get("h_pi0_pz_prior")

	if not h_prior:
	    raise RuntimeError(f"Could not find h_pi0_pz_prior in {prior_file}")

	
	out_f = ROOT.TFile.Open(filename, 'RECREATE')
	out_f.cd()

	for data_path in data_list:     

		samplename= glob.glob(f"{data_path}/HNL_{masstag}_*")[0]
		
		if '2pi' in data_path.split('/')[-2] :
		    d2tag='rho'
		else:
			d2tag='pi'
		
		lifetime  = float(samplename.split('_')[2])
	    			    			
		identifier = f"{leptontag}+{d2tag}_{masstag}"  # decaychannel_energy #_{parts[2]}      
		
		sample_dir = out_f.mkdir(identifier)
		sample_dir.cd()

		h=process_data(samplename,h_prior=h_prior,lepton_pdg_=lepton_pdg_,check_PDG=PIDcheck,check_SBT=SBTcheck,check_UBT=UBTcheck)

		#verifydata(f"{data_path}/{samplename}")

		sample_dir.cd()

		print(data_path, f"{leptontag}+{d2tag}", f"{masstag} GeV: Done")

		for name, hist in h.items():
			
			if name in ["calcmassdist1","calcmassdist2","calcmassdist"]:
				bin = hist.GetMaximumBin()
				mode = hist.GetBinCenter(bin)
				print(f"{name} \t Most probable HNL mass = {mode:.3f} GeV")
			hist.Write()

	for data_path in neuDISpath: 

		identifier = f"neuDIS"  # decaychannel_energy #_{parts[2]}      
		
		sample_dir = out_f.mkdir(identifier)
		sample_dir.cd()
			
		h=process_data(f"{data_path}",h_prior=h_prior,lepton_pdg_=lepton_pdg_,check_PDG=PIDcheck,check_SBT=SBTcheck,check_UBT=UBTcheck)

		sample_dir.cd()

		for hist in h.values():
		    hist.Write()		

		print(data_path," Done")

	for data_path in muonDISpath: 

		identifier = f"muonDIS"  # decaychannel_energy #_{parts[2]}      
		
		sample_dir = out_f.mkdir(identifier)
		sample_dir.cd()
		
		h=process_data(f"{data_path}",h_prior=h_prior,lepton_pdg_=lepton_pdg_,check_PDG=PIDcheck,check_SBT=SBTcheck,check_UBT=UBTcheck)

		sample_dir.cd()

		for hist in h.values():
		    hist.Write()		

		print(data_path," Done")

	out_f.Close()


def overlay_histos(file_name, hist_name="calcmassdist1",canvastitle="", norm=False, hatch=False):
    """
    Draw <hist_name> from every top-level sub-directory of <file_name>
    onto one canvas, then write the canvas back to the same file.

    * norm  – scale each histo to area = 1 (shape comparison)
    * hatch – give every histo a different fill-style
    """
    f = ROOT.TFile.Open(file_name, "UPDATE")
    if not f or f.IsZombie():
        raise RuntimeError("cannot open " + file_name)

    # ---------------- collect histograms ----------------
    hists, labels = [], []
    for key in f.GetListOfKeys():
        d = key.ReadObj()
        if not d.InheritsFrom("TDirectory"):  # skip canvases etc.
            continue
        h = d.Get(hist_name)
        if h:
            hists.append(h)
            labels.append(d.GetName())

    if not hists:
        print(f"[overlay] No '{hist_name}' found in any directory")
        f.Close()
        return

    # ---------------- styling ----------------
    #colors   = [ROOT.kBlue+1, ROOT.kRed+1, ROOT.kGreen+2,
    #            ROOT.kMagenta+2, ROOT.kOrange+7, ROOT.kCyan+1]
    colors = [
        ROOT.kBlue+1,      # deep blue
        ROOT.kRed+1,       # strong red
        ROOT.kGreen+2,     # vivid green
        ROOT.kMagenta+2,   # magenta
        ROOT.kOrange+7,    # bright orange
        ROOT.kCyan+1,      # cyan
        ROOT.kAzure+2,     # sky-blue
        ROOT.kPink+6,      # pink
        ROOT.kSpring+5,    # spring green
        ROOT.kViolet+1,    # violet
        ROOT.kYellow+2,    # yellow
        ROOT.kTeal+3       # teal
        ]

    patterns = [3004, 3005, 3006, 3007, 3013, 3018]

    can = ROOT.TCanvas(f"combined_{hist_name}",
                       f"combined_{hist_name}", 800, 600)

    #leg = ROOT.TLegend(0.60, 0.68, 0.88, 0.88)
    #leg.SetBorderSize(0); leg.SetFillStyle(0);leg.SetTextSize(0.035)

    leg = ROOT.TLegend(0.52, 0.62, 0.94, 0.88)  # (x1,y1,x2,y2) in NDC
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    #leg.SetTextFont(42)
    leg.SetTextSize(0.035)        # already added
    leg.SetNColumns(2)            # two-column layout
    leg.SetMargin(0.25)           # indent text a bit from the line samples
    
    max_value = max(h.GetMaximum() for h in hists)
    
    for i, (h, lab) in enumerate(zip(hists, labels)):

        h.SetStats(0)
        h.SetMaximum(max_value * 1.5) # Scale y-axis to avoid cutoff
        
        #if norm and h.Integral() > 0:
        #    h.Scale(1.0/h.Integral())
        
        col  = colors[i % len(colors)]
        
        h.SetLineColor(col); h.SetLineWidth(2);#h.SetFillColor(col);
        
        if hatch: h.SetFillStyle(patterns[i % len(patterns)])

        h.SetTitle(f"{canvastitle}"if i == 0 else "")
        h.Draw("EHIST" if i == 0 else "EHISTSAME")
        leg.AddEntry(h, lab, "f")
    
    leg.Draw(); can.SetGrid(); can.Update()

    # ---------------- write & close ----------------
    f.cd(); can.Write()     # stores canvas at file top level
    f.Close()
    print(f"[overlay] Wrote canvas 'combined_{hist_name}' "
          f"with {len(hists)} histograms")

def overlay_histos_by_mass(file_name, hist_name="calcmassdist", canvastitle="", norm=False, hatch=False):
    """
    Overlay signal (e+rho) histograms for each mass together with the
    corresponding e+pi histogram *and* the DIS backgrounds (neuDIS and
    muonDIS) on separate canvases.

    Each output canvas is named
        combined_<hist_name>_<mass>
    so you can quickly browse them in the ROOT file.
    """
    f = ROOT.TFile.Open(file_name, "UPDATE")
    if not f or f.IsZombie():
        raise RuntimeError("cannot open " + file_name)

    # ---------------- collect histograms ----------------
    mass_groups = {}   # {mass : [(hist, label), ...]}
    disc_hists  = []   # [(hist, label)]  for mass‑independent backgrounds

    for key in f.GetListOfKeys():
        d = key.ReadObj()
        if not d.InheritsFrom("TDirectory"):
            continue
        h = d.Get(hist_name)
        if not h:
            continue
        label = d.GetName()

        # Identify backgrounds that do **not** carry a mass tag
        if "DIS" in label:
            disc_hists.append((h, label))
            continue

        # Expect signal dir names like   e+rho_<mass>   or   e+pi_<mass>
        if "+" in label and "_" in label:
            try:
                mass_tag = label.split("_")[1]
            except IndexError:
                continue
            mass_groups.setdefault(mass_tag, []).append((h, label))

    if not mass_groups and not disc_hists:
        print(f"[overlay_by_mass] No histograms named '{hist_name}' found")
        f.Close(); return

    # ---------------- styling ----------------
    colors = [
        ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, ROOT.kOrange+7,
        ROOT.kMagenta+2, ROOT.kCyan+1, ROOT.kAzure+2
    ]
    patterns = [3004, 3005, 3006, 3007, 3013, 3018]

    # ---------------- one canvas per mass ----------------
    for mass_tag, sig_hists in mass_groups.items():
        # add mass‑independent backgrounds (DIS) to every mass canvas
        plot_items = sig_hists + disc_hists
        if not plot_items:
            continue

        can = ROOT.TCanvas(f"combined_{hist_name}_{mass_tag}",
                           f"combined_{hist_name} :: m = {mass_tag}", 800, 600)
        leg = ROOT.TLegend(0.52, 0.62, 0.94, 0.88)
        leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.035)
        leg.SetNColumns(2); leg.SetMargin(0.25)

        max_value = max(h.GetMaximum() for h, _ in plot_items)
        for i, (h, lab) in enumerate(plot_items):
            h.SetStats(0)
            h.SetMaximum(max_value * 1.5)
            col = colors[i % len(colors)]
            h.SetLineColor(col); h.SetLineWidth(2)
            if hatch:
                h.SetFillStyle(patterns[i % len(patterns)])
            h.SetTitle(canvastitle if i == 0 else "")
            h.Draw("EHIST" if i == 0 else "EHISTSAME")
            leg.AddEntry(h, lab, "f")

        leg.Draw(); can.SetGrid(); can.Update(); f.cd(); can.Write()
        print(f"[overlay_by_mass] Wrote canvas '{can.GetName()}' with {len(plot_items)} histograms")

    f.Close()

def overlay_pi0_groups(file_name, normalize=False, hatch=False):
    """
    For each HNL‐mass directory in file_name, draw two canvases:
      1) calc_pion0_pt vs truth_pion0_pt
      2) sol1_calc_pion0_pp vs sol2_calc_pion0_pp vs truth_pion0_pp
    """
    f = ROOT.TFile.Open(file_name, "UPDATE")
    if not f or f.IsZombie():
        raise RuntimeError("cannot open " + file_name)

    groups = [
        (["calc_pion0_pt", "truth_pion0_pt"], "Pion0 Transverse Momentum", "p_{T} (GeV/c)"),
        (["sol1_calc_pion0_pp", "sol2_calc_pion0_pp", "truth_pion0_pp"],
         "Pion0 Parallel Momentum", "p_{∥} (GeV/c)")
    ]

    # Loop over each top‐level directory (each HNL mass)
    for key in f.GetListOfKeys():
        d = key.ReadObj()
        if not d.InheritsFrom("TDirectory"):
            continue

        mass_dir = d.GetName()
        f.cd(mass_dir)

        for hist_keys, title, x_label in groups:
            # collect histos & labels
            hists, labels = [], []
            for hname in hist_keys:
                h = d.Get(hname)
                if h:
                    hists.append(h)
                    labels.append(hname)
            if not hists:
                continue

            # make canvas
            can = ROOT.TCanvas(f"{mass_dir}_{hist_keys[0]}", f"{mass_dir} :: {title}", 800, 600)
            leg = ROOT.TLegend(0.6,0.7,0.9,0.9)
            maxy = max(h.GetMaximum() for h in hists) * 1.5

            # styling
            colors  = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2]
            patterns= [3004, 3005, 3001]

            for i, h in enumerate(hists):
                h.SetStats(0)
                if normalize and h.Integral()>0:
                    h.Scale(1.0/h.Integral())
                h.SetLineColor(colors[i])
                h.SetLineWidth(2)
                if hatch:
                    h.SetFillStyle(patterns[i])
                    h.SetFillColor(colors[i])
                h.SetMaximum(maxy)
                h.SetTitle(title)
                h.GetXaxis().SetTitle(x_label)
                h.GetYaxis().SetTitle("Entries")
                draw_opt = "HIST" if i==0 else "HISTSAME"
                h.Draw(draw_opt)
                leg.AddEntry(h, labels[i], "l" if not hatch else "f")

            leg.Draw()
            can.Write()

        f.cd("/")  # back to root
    f.Close()
    print(f"[overlay_pi0_groups] Done writing π⁰‐comparisons to {file_name}")


if __name__ == "__main__":

	from argparse import ArgumentParser

	parser = ArgumentParser()
	
	parser.add_argument("--lepton", choices=['e', 'mu'], required=True, help="Choose lepton flavor: e or mu")
	parser.add_argument("--mass", choices=['1.0','1.1','1.2','1.3','1.4'], required=True, help="Choose mass range")
	parser.add_argument("--test", dest="testing_code" , help="Run Test" , required=False, action="store_true",default=False)
	parser.add_argument("--wPID", dest="PIDcheck",action="store_true",help="Enable PID check (default: disabled)")
	parser.add_argument("--wSBT", dest="SBTcheck",action="store_true",help="Enable AdvSBT+GNNSBT@45MeV veto (default: disabled)")
	parser.add_argument("--wUBT", dest="UBTcheck",action="store_true",help="Enable simple UBT veto (default: disabled)")

	
	args = parser.parse_args()

	
	SBTflag='wSBT' if args.SBTcheck else 'noSBT'
	UBTflag='wUBT' if args.UBTcheck else 'noUBT'
	PIDflag='wPID' if args.PIDcheck else 'noPID'

	args.output=f"HNLinvmass_wBG_{PIDflag}_{UBTflag}_{SBTflag}_{args.lepton}rho_{args.mass}.root"

	if args.testing_code:
		args.output="test_"+args.output

	main(lepton=args.lepton,mass=args.mass, filename=args.output,PIDcheck=args.PIDcheck,SBTcheck=args.SBTcheck,UBTcheck=args.UBTcheck)


	if args.lepton=='mu':
		lepton_latex='#mu'
	
	if args.lepton=='e':
		lepton_latex='#it{e}'

	#overlay_histos(
	#    args.output,
	#    hist_name="calcmassdist",
	#    canvastitle=f'#it{{N}} #rightarrow {lepton_latex}+#rho search'
	#)

	#overlay_histos(
	#    args.output,
	#    hist_name="invmassdist",
	#    canvastitle=f'#it{{N}} #rightarrow {lepton_latex}+#pi search'
	#)

	# 2. Per‑mass overlays: e+rho vs e+pi vs DIS backgrounds
	#overlay_histos_by_mass(
	#    args.output,
	#    hist_name="calcmassdist",
	#    canvastitle=f'#it{{N}} #rightarrow {lepton_latex}+#rho search'
	#)

	#overlay_pi0_groups(
	#    file_name=args.output,
	#    normalize=False,
	#    hatch=True
	#)