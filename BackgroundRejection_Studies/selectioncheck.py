#!/usr/bin/env python3
"""Script to evaluate signal selection on different samples (Studies separated into : all,vesselCase,heliumCase) """

#-----------------------------------------------------------------------------------------------------------

from argparse import ArgumentParser
import ROOT
import rootUtils as ut
import shipunit as u
from collections import defaultdict
#import helperfunctions as analysis_toolkit
from tabulate import tabulate
import numpy as np
import pandas as pd
import pathlib, datetime 
import glob
import math
from array import array


def dump_summary_csv(job_id,event_stats,pass_stats,out_dir= ".",**meta):
    """Dump Analysis Summary onto csv file."""  

    baseline = dict(job=job_id,time=datetime.datetime.utcnow().isoformat(timespec="seconds"),**meta)

    def pack(tag, ev_map):
        return {**baseline,"tag": tag,"nCandidates": len(ev_map),"nEvents15y": sum(ev_map.values())}

    rows = [pack(t, event_stats[t]) for t in ("simulated", "reconstructed")] \
         + [pack(t, m) for t, m in pass_stats.items()]

    path = pathlib.Path(out_dir) / f"selection_summary_{job_id.replace('/', '_')}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Output summary saved in", path)

def ip_category(ip_elem):
    """Return which sub-sample the event belongs to."""
    
    if ip_elem.startswith(("LiSc", "VetoInnerWall", "VetoOuterWall","VetoVerticalRib","VetoLongitRib")):
        return "vesselCase"
    if ip_elem.startswith("DecayVacuum"):
        return "heliumCase"
    
    return "all"       

def fixwidth_tabulate(rows, headers, *, width=50, **kw):
    """Fix tabulate width for first coloumn"""

    pad = lambda s: f"{s:<{width}}"
    rows2    = [[pad(str(r[0])), *r[1:]] for r in rows]
    headers2 = [pad(str(headers[0])), *headers[1:]]
    return tabulate(rows2,
                    headers=headers2,
                    **kw)

def helium_rhoL(event, track_index=0,
                rho_he=1.78e-4, tol=6e-5, eps=1e-4, max_hops=256):
    
    """
    Return helium-only rho*L (g/cm^2) for a DIS vertex.
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


def main(weight_function,IP_CUT = 250,fixTDC=None,fix_candidatetime=None,fix_nDIS=None,finalstate='None'):
    
    """Main function to analyse the selection efficiency of different cuts."""
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--path"  ,dest="path"         ,help="Path to simulation file",required=True)
    parser.add_argument("-i","--jobDir"  ,dest="jobDir"      ,help="job name of input file",  type=str,required=True)
    parser.add_argument(     "--test"    ,dest="testing_code",help="Run Test on 100 events of the input file"              ,  action="store_true")

    options = parser.parse_args()

    if isinstance(IP_CUT, (list, tuple)):        # [10, 250) case
        ip_low, ip_high = sorted(IP_CUT)
    else:                                        # [0, 250) case
        ip_low, ip_high = 0.0, float(IP_CUT)

    print(f"IP_CUT set as [{ip_low},{ip_high})\n\n")
    
    file_name = glob.glob(f"{options.path}/{options.jobDir}/ship.conical*_rec.root")[0]
    
    f = ROOT.TFile.Open(file_name,"read")
    tree = f.Get("cbmsim")

    geofile_name = glob.glob(f"{options.path}/{options.jobDir}/geofile_full.conical*.root")[0]
    geo_file = ROOT.TFile.Open(geofile_name,"read") 
    
    import helperfunctions as analysis_toolkit #torch ROOT 6.32 crash workaround, import torch AFTER initialising ROOT

    ctx       = analysis_toolkit.AnalysisContext(tree, geo_file)

    selection = analysis_toolkit.selection_check(ctx)
    inspector = analysis_toolkit.event_inspector(ctx)
    veto_ship = analysis_toolkit.veto_tasks(ctx)


    pre_tags = [
                "n_particles",
                "fiducial",
                "dist2innerwall",
                "dist2vesselentrance",
                "impact_par",
                "doca",
                "n_dof",
                "reduced_chi2",
                "d_mom",
                "preselection",
                ]
    veto_tags = [
                "BasicSBT@45MeV",
                "BasicSBT@90MeV",
                "BasicSBT@0MeV",
                "AdvSBT@45MeV",
                "AdvSBT@90MeV",
                'GNNSBT@45MeV',
                "[ AdvSBT+GNNSBT ]@45MeV",
                "UBT",
                ]

    other_tags= ["inv_mass",
                
                "PID",

                "PID_leptonic",
                "PID_ee",
                "PID_mumu",
                
                "PID_semileptonic",
                "PID_semileptonic_e",
                "PID_semileptonic_mu",
                ]


    combinedveto_tags = ["UBT+BasicSBT@45MeV",
                         "UBT+BasicSBT@90MeV",
                         "UBT+AdvSBT@45MeV",
                         "UBT+AdvSBT@90MeV",
                         "UBT+GNNSBT@45MeV",
                         "UBT+[ AdvSBT+GNNSBT ]@45MeV"
                            ]


    combined_Basic45       = ["preselection+UBT",
                        "preselection+UBT+BasicSBT@45MeV",
                        "preselection+UBT+BasicSBT@45MeV+PID",
                        "preselection+UBT+BasicSBT@45MeV+PID+inv_mass"]

    combined_Basic90       = ["preselection+UBT",
                        "preselection+UBT+BasicSBT@90MeV",
                        "preselection+UBT+BasicSBT@90MeV+PID",
                        "preselection+UBT+BasicSBT@90MeV+PID+inv_mass"]
    
    combined_Adv45       = ["preselection+UBT",
                        "preselection+UBT+AdvSBT@45MeV",
                        "preselection+UBT+AdvSBT@45MeV+PID",
                        "preselection+UBT+AdvSBT@45MeV+PID+inv_mass"]

    combined_Adv90       = ["preselection+UBT",
                        "preselection+UBT+AdvSBT@90MeV",
                        "preselection+UBT+AdvSBT@90MeV+PID",
                        "preselection+UBT+AdvSBT@90MeV+PID+inv_mass"]

    
    combined_GNN45       = ["preselection+UBT",
                        "preselection+UBT+GNNSBT@45MeV",
                        "preselection+UBT+GNNSBT@45MeV+PID",
                        "preselection+UBT+GNNSBT@45MeV+PID+inv_mass"]
    combined_45         = ["preselection+UBT",
                        "preselection+UBT+[ AdvSBT+GNNSBT ]@45MeV",
                        "preselection+UBT+[ AdvSBT+GNNSBT ]@45MeV+PID",
                        "preselection+UBT+[ AdvSBT+GNNSBT ]@45MeV+PID+inv_mass"]


    combined_other    = [
                        "preselection+AdvSBT@45MeV",
                        "preselection+AdvSBT@90MeV",

                        "preselection+BasicSBT@45MeV",
                        "preselection+BasicSBT@90MeV",
                        "preselection+inv_mass",
                        "preselection+UBT+inv_mass",
                        "preselection+UBT+PID",
                        "preselection+UBT+PID+inv_mass",
                        "preselection+UBT+BasicSBT@45MeV+inv_mass",
                        "preselection+UBT+BasicSBT@90MeV+inv_mass",

                        "preselection+UBT+AdvSBT@45MeV+inv_mass",
                        "preselection+UBT+AdvSBT@90MeV+inv_mass",
                        'GoodDaughters+n_particles+FiducialCuts+doca+impact_par+UBT',
                        'GoodDaughters+n_particles+FiducialCuts+doca+impact_par',
                        'GoodDaughters+n_particles+FiducialCuts+doca',
                        'GoodDaughters+n_particles+FiducialCuts',
                        'GoodDaughters+n_particles',
                        'FiducialCuts',
                        'GoodDaughters',
                        ]
    
    
    COMBO_POS_TAGS = [

        'preselection',
        'preselection+BasicSBT@45MeV',
        'preselection+AdvSBT@45MeV',
        'preselection+[ AdvSBT+GNNSBT ]@45MeV',  

        'preselection+UBT',
        'preselection+UBT+BasicSBT@45MeV',
        'preselection+UBT+AdvSBT@45MeV',
        'preselection+UBT+[ AdvSBT+GNNSBT ]@45MeV',  


        'preselection+PID',        
        'preselection+BasicSBT@45MeV+PID',
        'preselection+AdvSBT@45MeV+PID',
        'preselection+[ AdvSBT+GNNSBT ]@45MeV+PID',  


        'preselection+UBT+PID',        
        'preselection+UBT+BasicSBT@45MeV+PID',
        'preselection+UBT+AdvSBT@45MeV+PID',
        'preselection+UBT+[ AdvSBT+GNNSBT ]@45MeV+PID',  

        'preselection+UBT+BasicSBT@45MeV+PID+inv_mass',
        'preselection+UBT+GNNSBT@45MeV+PID',
        'preselection+UBT+GNNSBT@45MeV+PID+inv_mass',
        'preselection+UBT+[ AdvSBT+GNNSBT ]@45MeV+PID+inv_mass',  

    ]



    from itertools import chain

    ALL_TAGS = list(chain(
        pre_tags,
        veto_tags,
        other_tags,
        combinedveto_tags,
        combined_Basic45, combined_Basic90,
        combined_Adv45,   combined_Adv90,
        combined_GNN45,   combined_other
    ))

    cats = ("all", "vesselCase", "heliumCase")

    #--------------------------------------------------------------------------------------------------------------

    # Trees + branch buffers to store (x,y,z,w) for passing candidates
    
    pos_trees = {c: {} for c in cats}
    pos_bufs  = {c: {} for c in cats}

    def _san(s):  # safe tree name
        return s.replace('[','').replace(']','').replace(' ','').replace('+','_')

    for c in cats:
        for tag in COMBO_POS_TAGS:
            tname = f"{c}_{_san(tag)}_pos"
            t     = ROOT.TTree(tname, f"Candidate production positions for {tag} ({c})")
            x = array('f', [0.0]); y = array('f', [0.0]); z = array('f', [0.0]); w = array('f', [0.0])
            t.Branch("x", x, "x/F"); t.Branch("y", y, "y/F"); t.Branch("z", z, "z/F"); t.Branch("w", w, "w/F")
            pos_trees[c][tag] = t

            ipx = array('f', [0.0]); ipy = array('f', [0.0]); ipz = array('f', [0.0])
            t.Branch("IP_x", ipx, "IP_x/F")
            t.Branch("IP_y", ipy, "IP_y/F")
            t.Branch("IP_z", ipz, "IP_z/F")

            pos_bufs[c][tag]  = (x, y, z, w, ipx, ipy, ipz)

    #--------------------------------------------------------------------------------------------------------------

    pass_stats = {
        c: defaultdict(lambda: {}, {t: {} for t in ALL_TAGS})
        for c in cats
    }

    event_stats = {c: {"simulated": {}, "reconstructed": {}} for c in cats}

    hist_dict = {}
    
    for c in cats:
        
        hist_dict[c]={}
        
        if c=='heliumCase':
            ut.bookHist(hist_dict[c], f"{c}_rho_l", "rho_l" , 1000, 0, 1)
        else:
            ut.bookHist(hist_dict[c], f"{c}_rho_l", "rho_l" , 1000, 0, 1000)
        
        pre = f"{c}_"
        
        ut.bookHist(hist_dict[c], pre +"candidate_time","candidate time @ decay vertex; ns",300,0, 300)
        ut.bookHist(hist_dict[c], pre +"impact_parameter", "Impact parameter; cm", 500, 0, 500)
        ut.bookHist(hist_dict[c], pre +"dist_to_innerwall", "Distance to inner wall; cm", 200, 0, 100)
        ut.bookHist(hist_dict[c], pre +"dist_to_vesselentrance","Distance to Decay Vessel Entrance; cm", 500, 0,5000)
        ut.bookHist(hist_dict[c], pre +"inv_mass", "Invariant mass; GeV", 500, 0, 5)
        ut.bookHist(hist_dict[c], pre +"DOCA", "Distance of closest approach; cm", 1000, 0, 10)
        ut.bookHist(hist_dict[c], pre +"len_Particles", "len(tree.Particles); Number of candidates per event", 5,0,5)
        ut.bookHist(hist_dict[c], pre +"d_mom", "momentum of daughters; d1 (GeV); d2 (GeV)",400,0,400,400,0,400)
        ut.bookHist(hist_dict[c], pre +"nDOF", "nDOF; d1; d2", 30, 0, 30, 30, 0, 30)
        ut.bookHist(hist_dict[c], pre +"chi2nDOF", "chi2nDOF; d1; d2", 10, 0, 10, 10, 0, 10)
        ut.bookHist(hist_dict[c], pre +"x_pos", "Candidate x pos; cm", 300,  -300, 300)
        ut.bookHist(hist_dict[c], pre +"y_pos", "Candidate y pos; cm", 300,  -300, 300)
        ut.bookHist(hist_dict[c], pre +"z_pos", "Candidate z pos; cm", 6000, -3000, 3000)
        ut.bookHist(hist_dict[c], pre +"nSBThits", "nSBThits>45MeV for events passing selection; nSBThits; ",100,0,100)

        

    for event_nr, event in enumerate(tree):
                
        if options.testing_code and event_nr>99: continue
        
        if event_nr%100==0:
            print(f"Event {event_nr}")
        
        #inspector.dump_event(track_p_threshold=0.5)  # in GeV
        
        # -----------------------------------------------------------------
        
        # interaction-point category
        
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
            corrected_rhoL= helium_rhoL(event)
            event_weight  = weight_function(event,w_DIS=corrected_rhoL)
            rhoL=corrected_rhoL
        else:
            event_weight = weight_function(event)
            rhoL=event.MCTrack[2].GetWeight()
    
        scalefactor={}
        
        for c in {"all", cat}:
            
            if c not in scalefactor:         

                if fix_nDIS:
                    scalefactor[c] = fix_nDIS(options.jobDir,event_nr,c) 
                else:
                    scalefactor[c] = 1 #no scaling of weights

            pre = f"{c}_"

            hist_dict[c][pre+"rho_l"].Fill(rhoL)
    
            event_stats[c]["simulated"][event_nr] = event_weight * scalefactor[c]
    
            if len(event.Particles):
                event_stats[c]["reconstructed"][event_nr] = event_weight * scalefactor[c]

        for candidate_id_in_event, signal in enumerate(event.Particles):
            
            selection_list =  defaultdict(dict) 
            
            #--------------------------Preselection---------------------------------

            if len(event.Particles) ==1:
                selection_list["n_particles"] = True

            if selection.is_in_fiducial(signal):
                selection_list["fiducial"] = True
                
            if selection.dist_to_innerwall(signal) > 5 * u.cm:
                selection_list["dist2innerwall"] = True
                        
            if selection.dist_to_vesselentrance(signal) > 20 * u.cm:
                selection_list["dist2vesselentrance"] = True
            
            if ip_low * u.cm <= selection.impact_parameter(signal) < ip_high * u.cm:
                selection_list["impact_par"] = True                

            if selection.DOCA(signal) < 1 * u.cm:
                selection_list["doca"] = True

            if np.all(selection.nDOF(signal) > 25):
                selection_list["n_dof"] = True

            if np.all(selection.chi2nDOF(signal) < 5):
                selection_list["reduced_chi2"] = True

            if np.all(selection.daughtermomentum(signal) > 1 * u.GeV):
                selection_list["d_mom"] = True

            if fix_candidatetime:           #fix timing for EventCalc special case.
                offset=fix_candidatetime(event)
            else:
                offset=0

            #-------------------------------------------------------------------------
            #-------------------------Signal Selections----------------

            selection_list['FiducialCuts'   ]      = selection_list["fiducial"] and selection_list["dist2innerwall"] and selection_list["dist2vesselentrance"]
            selection_list['GoodDaughters'  ]      = selection_list["n_dof"] and selection_list["reduced_chi2"] and selection_list["d_mom"]
            
            selection_list['GoodDaughters+'+'n_particles']                                       = selection_list['GoodDaughters'] and selection_list["n_particles"]
            selection_list['GoodDaughters+'+'n_particles+'+'FiducialCuts']                       = selection_list['GoodDaughters'] and selection_list["n_particles"] and selection_list["fiducial"] and selection_list["dist2innerwall"] and selection_list["dist2vesselentrance"]
            selection_list['GoodDaughters+'+'n_particles+'+'FiducialCuts+'+'doca']               = selection_list['GoodDaughters'] and selection_list["n_particles"] and selection_list["fiducial"] and selection_list["dist2innerwall"] and selection_list["dist2vesselentrance"] and selection_list["doca"]
            selection_list['GoodDaughters+'+'n_particles+'+'FiducialCuts+'+'doca+'+'impact_par'] = selection_list['GoodDaughters'] and selection_list["n_particles"] and selection_list["fiducial"] and selection_list["dist2innerwall"] and selection_list["dist2vesselentrance"] and selection_list["doca"] and selection_list["impact_par"]        

            
            pre_ok = selection.preselection_cut(signal, IP_cut=ip_high, show_table=False,offset=offset) and selection_list["impact_par"] #adding refined IP check since preselection_cut only sets upperlimit on IP
            
            #if pre_ok:
            #    print(f"Event:{event_nr} Candidate_index: {candidate_id_in_event} <--passes the pre-selection\n\n")
            
            selection_list['preselection'] = pre_ok

            #-------------------------------------------------------------------------
            #-------------------------Veto Decisions----------------------------------
            
            BasicSBT45_veto ,wBasicSBT45,HitsSBT45  =   veto_ship.SBT_decision(threshold=45)
            selection_list['BasicSBT@45MeV'        ]   = not(BasicSBT45_veto)

            
            BasicSBT90_veto ,wBasicSBT90,HitsSBT90     = veto_ship.SBT_decision(threshold=90)
            selection_list['BasicSBT@90MeV'        ]   = not(BasicSBT90_veto)
            
            
            BasicSBT0_veto                             = bool(len(event.Digi_SBTHits)) #any sbt activity
            selection_list['BasicSBT@0MeV'         ]   = not(BasicSBT0_veto)


            UBT_veto,ubthits                           = veto_ship.UBT_decision()
            selection_list['UBT'                   ]   = not(UBT_veto)

            selection_list['GoodDaughters+'+'n_particles+'+'FiducialCuts+'+'doca+'+'impact_par+'+'UBT'] = selection_list['GoodDaughters'] and selection_list["n_particles"] and selection_list["fiducial"] and selection_list["dist2innerwall"] and selection_list["dist2vesselentrance"] and selection_list["doca"] and selection_list["impact_par"] and selection_list['UBT']
            
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
            
            selection_list['AdvSBT@45MeV'          ]   = not(AdvSBT45_veto) #Extrapolation SBT veto @45
            selection_list['AdvSBT@90MeV'          ]   = not(AdvSBT90_veto) #Extrapolation SBT veto @90

            reject, pBG = veto_ship.Veto_decision_GNNbinary_wdeltaT(threshold=0.6,offset=offset)

            selection_list['GNNSBT@45MeV'          ]   = not(reject) # specific GNN trained on neuDIS in He

            selection_list['[ AdvSBT+GNNSBT ]@45MeV'           ]   = selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV']
            
            selection_list['GoodDaughters+'+'BasicSBT@45MeV'            ]   = selection_list['GoodDaughters'] and selection_list['BasicSBT@45MeV']
            selection_list['GoodDaughters+'+'AdvSBT@45MeV'              ]   = selection_list['GoodDaughters'] and selection_list['AdvSBT@45MeV'  ]
            selection_list['GoodDaughters+'+'[ AdvSBT+GNNSBT ]@45MeV'   ]   = selection_list['GoodDaughters'] and selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV']

            #-------------------------------------------------------------------------
            #-----------------------------Other Cuts----------------------------------
            
            inv_mass_pass=selection.invariant_mass(signal)  > 0.15*u.GeV

            selection_list['inv_mass']   = inv_mass_pass


            #--------------PID------------------------------------------------------

            pid= selection.pid_decision(candidate=signal)

            if finalstate == 'semileptonic':   pid_pass   = (int(pid)==2 or int(pid)==3)   #semi leptonic. final state or unknown(fallback)
            if finalstate == 'dileptonic'  :   pid_pass   = (int(pid)==1 or int(pid)==3)   #dileptonic final state or unknown(fallback)

            selection_list['PID']                   = pid_pass

            selection_list['PID_leptonic']          = (int(pid)==1 or int(pid)==3)
            selection_list['PID_semileptonic']      = (int(pid)==2 or int(pid)==3)

            selection_list['PID_ee']                = pid==1.1 or int(pid)==3
            selection_list['PID_mumu']              = pid==1.2 or int(pid)==3

            selection_list['PID_semileptonic_e']    = pid==2.1 or int(pid)==3
            selection_list['PID_semileptonic_mu']   = pid==2.2 or int(pid)==3


            #--------------------------------Combined Cuts-----------------------------------------            
            

            #UBT+ SBT

            selection_list['UBT+'+'BasicSBT@45MeV'          ]   =   selection_list['UBT'] and selection_list['BasicSBT@45MeV']
            selection_list['UBT+'+'BasicSBT@90MeV'          ]   =   selection_list['UBT'] and selection_list['BasicSBT@90MeV']
            selection_list['UBT+'+'AdvSBT@45MeV'            ]   =   selection_list['UBT'] and selection_list['AdvSBT@45MeV']
            selection_list['UBT+'+'AdvSBT@90MeV'            ]   =   selection_list['UBT'] and selection_list['AdvSBT@90MeV']            
            selection_list['UBT+'+'GNNSBT@45MeV'            ]   =   selection_list['UBT'] and selection_list['GNNSBT@45MeV']      
            selection_list['UBT+'+'[ AdvSBT+GNNSBT ]@45MeV' ]   =   selection_list['UBT'] and (selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV'])

            #Sel. + SBT
            
            selection_list['preselection+'+'BasicSBT@45MeV'             ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']
            selection_list['preselection+'+'BasicSBT@90MeV'             ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']
            selection_list['preselection+'+'AdvSBT@45MeV'               ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']
            selection_list['preselection+'+'AdvSBT@90MeV'               ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV'    ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV']

            #Sel. + UBT+ SBT

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV'          ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV'          ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV' ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV'])

            #Sel. + UBT
            
            selection_list['preselection+'+'UBT'   ]   = selection_list['preselection'] and selection_list['UBT']

            #Sel. + UBT + PID           

            selection_list['preselection+'+'UBT+'+'PID'                 ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID']  
            selection_list['preselection+'+'UBT+'+'PID_ee'              ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_ee']
            selection_list['preselection+'+'UBT+'+'PID_mumu'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_mumu']
            selection_list['preselection+'+'UBT+'+'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'UBT+'+'PID_semileptonic_mu' ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_semileptonic_mu']

            #Sel. + UBT + PID + SBT
            
            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'+'PID'             ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'+'PID'             ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'   +'PID'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'   +'PID'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'   +'PID'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+'   +'PID' ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV'])   and selection_list['PID']
            
            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_ee'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_ee']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_ee'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_ee']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_ee'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_ee']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_ee'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_ee']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_ee'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_ee']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_ee'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_ee']

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_mumu']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_mumu']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_mumu']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_mumu']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_mumu']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_mumu']

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_e']

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_mu']

            #Sel. + UBT + PID + SBT + inv.mass

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'+'PID+'+'inv_mass'             ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID'] and selection_list['inv_mass']            
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'+'PID+'+'inv_mass'             ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID'] and selection_list['inv_mass']            
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'   +'PID+'+'inv_mass'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'   +'PID+'+'inv_mass'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'   +'PID+'+'inv_mass'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+'   +'PID+'+'inv_mass' ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV'])   and selection_list['PID'] and selection_list['inv_mass']
            
            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_ee'] and selection_list['inv_mass']

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_mumu'] and selection_list['inv_mass']

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']

            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'          +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'          +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'            +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'            +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'            +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']

            #Sel. + PID

            selection_list['preselection+'+'PID'                ]   = selection_list['preselection'] and selection_list['PID']
            selection_list['preselection+'+'PID_ee'             ]   = selection_list['preselection'] and selection_list['PID_ee']
            selection_list['preselection+'+'PID_mumu'           ]   = selection_list['preselection'] and selection_list['PID_mumu']
            selection_list['preselection+'+'PID_semileptonic_e' ]   = selection_list['preselection'] and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'PID_semileptonic_mu']   = selection_list['preselection'] and selection_list['PID_semileptonic_mu']

            #Sel. + SBT + PID

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV'] and selection_list['PID']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV'] and selection_list['PID']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']   and selection_list['PID']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']   and selection_list['PID']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']   and selection_list['PID']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']  and selection_list['GNNSBT@45MeV']) and selection_list['PID']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_ee'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_ee']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_ee'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_ee']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_ee'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_ee']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_ee'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_ee']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_ee'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_ee']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_ee'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_ee']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_mumu']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_mumu']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_mumu']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_mumu']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_mumu'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_mumu']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_mumu'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_mumu']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_semileptonic_e'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_e']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_e'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_e']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_mu']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_mu'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_mu']

            #Sel. + SBT + PID + inv.mass

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV'] and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV'] and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']   and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']   and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']   and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID+'+'inv_mass'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']  and selection_list['GNNSBT@45MeV']) and selection_list['PID'] and selection_list['inv_mass']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_ee+'+'inv_mass'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_ee'] and selection_list['inv_mass']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_mumu+'+'inv_mass'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_mumu'] and selection_list['inv_mass']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']

            selection_list['preselection+'+'BasicSBT@45MeV+'          +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@45MeV']  and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'BasicSBT@90MeV+'          +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['BasicSBT@90MeV']  and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@45MeV+'            +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@45MeV']    and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'AdvSBT@90MeV+'            +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['AdvSBT@90MeV']    and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'GNNSBT@45MeV+'            +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['GNNSBT@45MeV']    and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']
            selection_list['preselection+'+'[ AdvSBT+GNNSBT ]@45MeV+' +'PID_semileptonic_mu+'+'inv_mass'  ]   = selection_list['preselection'] and (selection_list['AdvSBT@45MeV']   and selection_list['GNNSBT@45MeV']) and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']


            #Sel. + inv. mass

            selection_list['preselection+'+'inv_mass' ]   = selection_list['preselection'] and selection_list['inv_mass']

            #Sel. + PID + inv. mass

            selection_list['preselection+'+'PID+' +'inv_mass'               ]   = selection_list['preselection'] and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'PID_ee+'  +'inv_mass'           ]   = selection_list['preselection'] and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'PID_mumu+' +'inv_mass'          ]   = selection_list['preselection'] and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'PID_semileptonic_e+'+'inv_mass' ]   = selection_list['preselection'] and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'PID_semileptonic_mu+'+'inv_mass']   = selection_list['preselection'] and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']


            #Sel. + UBT + inv. mass

            selection_list['preselection+'+'UBT+'+'inv_mass' ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['inv_mass']

            #Sel. + UBT + PID + inv. mass

            selection_list['preselection+'+'UBT+'+'PID+' +'inv_mass'                ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'PID_ee+'+'inv_mass'              ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_ee'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'PID_mumu+'+'inv_mass'            ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_mumu'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'PID_semileptonic_e+'+'inv_mass'  ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_semileptonic_e'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'PID_semileptonic_mu+'+'inv_mass' ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['PID_semileptonic_mu'] and selection_list['inv_mass']

            #Sel. + UBT + SBT + inv. mass
            
            selection_list['preselection+'+'UBT+'+'BasicSBT@45MeV+'       +'inv_mass'   ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@45MeV'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'BasicSBT@90MeV+'       +'inv_mass'   ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['BasicSBT@90MeV'] and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@45MeV+'         +'inv_mass'   ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@45MeV']   and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'AdvSBT@90MeV+'         +'inv_mass'   ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['AdvSBT@90MeV']   and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'GNNSBT@45MeV+'         +'inv_mass'   ]   = selection_list['preselection'] and selection_list['UBT'] and selection_list['GNNSBT@45MeV']   and selection_list['inv_mass']
            selection_list['preselection+'+'UBT+'+'[ AdvSBT+GNNSBT ]@45MeV+'+'inv_mass' ]   = selection_list['preselection'] and selection_list['UBT'] and (selection_list['AdvSBT@45MeV'] and selection_list['GNNSBT@45MeV']) and selection_list['inv_mass']
            

            #-------------------------------------------------------------------------

            for c in {"all", cat}:
                
                pre = f"{c}_"

                event_weight_rescaled=event_weight*scalefactor[c]

                hist_dict[c][pre+"impact_parameter"        ].Fill(selection.impact_parameter(signal),event_weight_rescaled)   
                hist_dict[c][pre+"dist_to_innerwall"       ].Fill(selection.dist_to_innerwall(signal),event_weight_rescaled)
                hist_dict[c][pre+"dist_to_vesselentrance"  ].Fill(selection.dist_to_vesselentrance(signal),event_weight_rescaled)
                hist_dict[c][pre+"DOCA"                    ].Fill(selection.DOCA(signal),event_weight_rescaled)
                hist_dict[c][pre+"len_Particles"           ].Fill(len(tree.Particles),event_weight_rescaled)
                hist_dict[c][pre+"d_mom"                   ].Fill(*selection.daughtermomentum(signal),event_weight_rescaled)
                hist_dict[c][pre+"nDOF"                    ].Fill(*selection.nDOF(signal),event_weight_rescaled)
                hist_dict[c][pre+"chi2nDOF"                ].Fill(*selection.chi2nDOF(signal),event_weight_rescaled)
                hist_dict[c][pre+"inv_mass"                ].Fill(selection.invariant_mass(signal),event_weight_rescaled)
                
                signal_pos = ROOT.TLorentzVector()
                signal.ProductionVertex(signal_pos)
                hist_dict[c][pre+"z_pos"                   ].Fill(signal_pos.Z(),event_weight_rescaled)
                hist_dict[c][pre+"x_pos"                   ].Fill(signal_pos.X(),event_weight_rescaled)
                hist_dict[c][pre+"y_pos"                   ].Fill(signal_pos.Y(),event_weight_rescaled)
                hist_dict[c][pre+"nSBThits"                ].Fill(len(HitsSBT45),event_weight_rescaled)      

                
            #-------------------------------------------------------------------------

            # record positions for any passing combination for visualisation later
            
            for tag in COMBO_POS_TAGS:
                if selection_list.get(tag, False):
                    
                    for c2 in {"all", cat}:
                        x, y, z, w, ipx, ipy, ipz = pos_bufs[c2][tag]
                        x[0]   = signal_pos.X()
                        y[0]   = signal_pos.Y()
                        z[0]   = signal_pos.Z()
                        w[0]   = event_weight * scalefactor[c2]
                        ipx[0] = interaction_point.X()
                        ipy[0] = interaction_point.Y()
                        ipz[0] = interaction_point.Z()
                        pos_trees[c2][tag].Fill()

            #-------------------------------------------------------------------------

            for selection_name, passed in selection_list.items():

                if passed:
                    for c in {"all", cat}:
                        pass_stats[c][selection_name][event_nr] = event_weight * scalefactor[c]

    for cat in cats:

        # write hists onto root file
        ut.writeHists(hist_dict[cat], f"selectionparameters_{cat}.root")

        # append also the trees 
        with ROOT.TFile(f"selectionparameters_{cat}.root", "UPDATE") as rf:
            for tag, t in pos_trees[cat].items():
                rf.cd()
                t.Write()
    
        dump_summary_csv(
            job_id=f"{cat}_{options.jobDir}",     
            event_stats=event_stats[cat],
            pass_stats =pass_stats[cat],
            out_dir    =".",
            sample     =cat,                     
            ship_version="2024_helium"
        )

        if not options.testing_code: continue  #only dump the tables if test case
        
        print(f"----------------------------------------------------------{cat}---------------------------------------------------------")
        
        printed_tags =set()

        #--Table 1------------------------
        rows_event_stats = [
                                [tag,
                                len(ev_dict),                 # nCandidates
                                sum(ev_dict.values())]        # nEvents in 15 y
                            for tag, ev_dict in event_stats[cat].items()
                            ]

        
        recon_table = fixwidth_tabulate(
                        rows_event_stats,     
                        headers=[" ", "  nEvents  ", "nEvents in 15 y"],
                        floatfmt=".2e",
                        tablefmt="rounded_grid",
                    )

        print(recon_table)
        printed_tags.update(event_stats[cat])

        #--Table 2------------------------
        rows_pass_stats_presel  = [
                                        [tag,
                                        len(pass_stats[cat].get(tag, {})),                 # nCandidates
                                        sum((pass_stats[cat].get(tag, {})).values())]      # nEvents in 15 y
                                    for tag in pre_tags
                                    ]

        presel_table = fixwidth_tabulate(
                        rows_pass_stats_presel,
                        headers=["Pre-Selection Cut", "nCandidates", "nEvents in 15 y"],
                        floatfmt=".2e", tablefmt="rounded_grid"
                    )

        print(presel_table)
        printed_tags.update(pre_tags)

        #--Table 3--------------------------
        rows_pass_stats_vetosystems  = [
                                            [tag,
                                            len(pass_stats[cat].get(tag, {})),                 # nCandidates
                                            sum((pass_stats[cat].get(tag, {})).values())]      # nEvents in 15 y
                                        for tag in veto_tags
                                        ]

        veto_table = fixwidth_tabulate(
            rows_pass_stats_vetosystems,
            headers=["Veto System",   "nCandidates", "nEvents in 15 y"],
            floatfmt=".2e", tablefmt="rounded_grid"
        )
        
        print(veto_table)
        printed_tags.update(veto_tags)


        #--Tables 4-7--------------------------
        
        table_specs = [
                        ("Combined vetosystem cuts (UBT x SBT)"             , combinedveto_tags),
                        ("Ordered cuts (BasicSBT@45 MeV threshold)"         , combined_Basic45),
                        ("Ordered cuts (BasicSBT@90 MeV threshold)"         , combined_Basic90),
                        ("Ordered cuts (AdvSBT@45 MeV threshold)"           , combined_Adv45),
                        ("Ordered cuts (AdvSBT@90 MeV threshold)"           , combined_Adv90),
                        ("Ordered cuts (GNNSBT@45 MeV threshold)"           , combined_GNN45),
                        ("Ordered cuts ([ AdvSBT+GNNSBT ]@45MeV threshold)" ,  combined_45),
                        ]
        

        all_tables = []

        for title, tag_order in table_specs:
            rows = []
            printed_tags.update(tag_order)
            for tag in tag_order:
                ev_map = pass_stats[cat].get(tag, {})  # empty dict if tag never filled
                n_ev   = len(ev_map)                          # how many distinct events
                w_sum  = sum(ev_map.values())                 # total weighted events
                rows.append([tag, n_ev, f"{w_sum:.2e}"])

            tbl = fixwidth_tabulate(
                rows,
                headers=[title, "nCandidates", "nEvents in 15 y"],
                floatfmt=".2e",
                tablefmt="rounded_grid",
            )
            all_tables.append(tbl)


        for tbl in all_tables:
            print(tbl, end="\n\n")


        #--Table 8------------------------
        
        fallback_tags = [
            tag for tag, ev_map in pass_stats[cat].items()
            if tag not in printed_tags                      #and ev_map if  at least one event passed
        ]


        if fallback_tags:
            rows = [
                [tag, len(pass_stats[cat][tag]), f"{sum(pass_stats[cat][tag].values()):.2e}"]
                for tag in sorted(fallback_tags)
            ]
            fallback_table=fixwidth_tabulate(rows,
                                  headers=["Other cuts ", "nCandidates", "nEvents in 15 y"],
                                  floatfmt=".2e",
                                  tablefmt="rounded_grid")

            print(fallback_table)
        #---------------------------------------------------------------------
        
        
#if __name__ == "__main__":
#    main()