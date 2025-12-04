#!/usr/bin/env python3
"""
Build and save the π⁰ longitudinal momentum prior for a given MC signal ROOT file.

Produces a ROOT file containing a TH1D named "h_pi0_pz_prior".
"""
import ROOT,os
import argparse
import pandas as pd
from HNLinvmass_EventCalc import rotate_momentum_to_hnl_frame, PION0_MASS

def calc_eventweight(GetWeight,path):
    
    def define_weight_EventCalc(path):
        
        w_event=GetWeight

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

    if 'Signal_EventCalc' in path:

        return define_weight_EventCalc(path)

def main():

    p = argparse.ArgumentParser(description="Build π0 Pz prior from MC signal file")

    
    p.add_argument("--lepton", type=str, default='e', help="lepton (e/mu)")
    p.add_argument("--mass",  type=float,help='mass of the HNL',required=True)
    p.add_argument("--output", type=str, help="Output ROOT file to save prior histogram")
    p.add_argument("--bins",   type=int, default=400, help="Number of bins for P_parallel histogram")
    p.add_argument("--test", dest="testing_code" , help="Run Test" , required=False, action="store_true",default=False)
    
    args = p.parse_args()

    if args.lepton=='e':
        path     = '/eos/experiment/ship/user/anupamar/Signal_EventCalc/2pie/12968088'
    if args.lepton=='mu':
        path    = '/eos/experiment/ship/user/anupamar/Signal_EventCalc/2pimu/12968089'

    mass_str = f"{args.mass:.3e}" 
    
    if not args.output:
        args.output=f"pi0_pz_prior_{args.lepton}rho_{mass_str}.root"

    prefix = f"HNL_{mass_str}_"
    
    try:
        folder = next(d for d in os.listdir(path) if d.startswith(prefix))
    except StopIteration:
        raise RuntimeError(f"No folder starting with '{prefix}' found in {path}")

    h_prior = ROOT.TH1D("h_pi0_pz_prior",";#pi^{0} P_{#parallel} (GeV);PDF", args.bins, 0.0, 1000)
    
    print(f"{path}/{folder}")

    for inputFileNr,inputFolder in enumerate(os.listdir(f"{path}/{folder}")):

        try:
            file = ROOT.TFile.Open(f"{path}/{folder}/{inputFolder}/ship.conical.EvtCalc-TGeant4_rec.root")
            tree = file.cbmsim
            tree.SetBranchStatus("*", 0)            # Disable all branches
            tree.SetBranchStatus("MCTrack", 1)      # Enable only MCTrack
        except:
            continue
        
        if inputFileNr>10: break

        if args.testing_code and inputFileNr>1: break

        print(f"file {inputFileNr}")
        
        for eventNr,event in enumerate(tree):
            
            print(f"\t\t{eventNr}")
            
            event_weight= calc_eventweight(event.MCTrack[0].GetWeight(),f"{path}/{folder}")

            gamma_id=[]

            for i,track in enumerate(event.MCTrack):

                if track.GetMotherId()!=0: continue

                if abs(track.GetPdgCode())==22: 
                    gamma_id.append(i)

            if len(gamma_id)!=2: continue # only look for events with 2 gammas.

            gamma1 = ROOT.TLorentzVector()
            gamma2 = ROOT.TLorentzVector()


            gamma1.SetPxPyPzE(  event.MCTrack[gamma_id[0]].GetPx(),
                                event.MCTrack[gamma_id[0]].GetPy(),
                                event.MCTrack[gamma_id[0]].GetPz(),
                                event.MCTrack[gamma_id[0]].GetEnergy()
                              )

            gamma2.SetPxPyPzE(  event.MCTrack[gamma_id[1]].GetPx(),
                                event.MCTrack[gamma_id[1]].GetPy(),
                                event.MCTrack[gamma_id[1]].GetPz(),
                                event.MCTrack[gamma_id[1]].GetEnergy()
                              )

            truth_pion0 = gamma1 + gamma2

            decay_vtx = ROOT.TVector3(event.MCTrack[0].GetStartX(),event.MCTrack[0].GetStartY(), event.MCTrack[0].GetStartZ())
            prod_vtx = ROOT.TVector3(0, 0,  -5814.25)  # production point

            truth_rot = rotate_momentum_to_hnl_frame(decay_vtx, prod_vtx, truth_pion0)
            pz = truth_rot.Pz()
            h_prior.Fill(pz)

    # normalize to unit area
    integral = h_prior.Integral("width")
    if integral > 0:
        h_prior.Scale(1.0 / integral)

    f_out = ROOT.TFile.Open(args.output, "RECREATE")
    h_prior.Write()
    f_out.Close()
    file.Close()

    print(f"Wrote prior histogram to {args.output} with {args.bins} bins up to 1000 GeV")
        

if __name__ == '__main__':
    main()
