"""
Calculate the invariant mass for HNL->rho lepton channel using a rho constraint

USAGE: python HNLinvmass_EventCalc.py --signalpath <locationtosignalfile>

"""
import ROOT
from argparse import ArgumentParser
import csv
pdg = ROOT.TDatabasePDG.Instance()


PION0_MASS = 134.9768*0.001             # Mass of the neutral pion.
RHO_MASS_NOMINAL = 775.45*0.001         # Mass of the rho meson.

def plot_distributions(h1,h2,filename="plots_HNLMass_EventCalc.root"):

    output_file = ROOT.TFile(filename, "RECREATE")
    
    for h in [h1,h2]:
        for akey in h:
            if not hasattr(h[akey],'Class'): continue
            cln = h[akey].Class().GetName()
            if not cln.find('TH')<0 or not cln.find('TP')<0:   h[akey].Write()
    
    canvas = ROOT.TCanvas("HNLMass", "", 800, 600)
    legend = ROOT.TLegend(0.55, 0.7, 0.9, 0.9);legend.SetTextSize(0.035);#ROOT.gPad.SetLogy()
    
    histograms = [
                h1["e+rho_calcmassdist1"],
                #h1["e+rho_calcmassdist2"],
                h1["e+rho_invmassdist"],
                h2["e+pi_calcmassdist1"],
                #h2["e+pi_calcmassdist2"],
                h2["e+pi_invmassdist"]
                ]

    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kYellow+2]
    #colors = [ROOT.kRed, ROOT.kBlue, ROOT.kCyan, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kYellow+2]
    
    max_value = max(h.GetMaximum() for h in histograms)
    
    for i, h in enumerate(histograms):
        h.SetStats(0)
        h.SetMaximum(max_value * 1.5) # Scale y-axis to avoid cutoff
        h.SetLineColor(colors[i])
        h.SetLineWidth(3)
        h.SetFillColor(colors[i])
        
        h.SetFillStyle(3005 if i % 2 == 0 else 3003)

        legend.AddEntry(h, h.GetTitle(), "f")
        h.SetTitle("")  
        if i == 0:
            h.GetXaxis().SetTitle("Reconstructed HNL Mass (GeV/c^{2})")
            h.Draw("HIST")  # Draw first histogram normally
        else:
            h.Draw("HIST SAME")  

    legend.Draw()
    canvas.Update()
    canvas.Write()


    nametags = ['e+rho', 'e+pi']

    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kOrange+7]

    for nametag in nametags:

        if nametag=='e+rho':
            title='#it{N}(1 GeV) #rightarrow e+#pi+#pi^{0}'
        if nametag=='e+pi':
            title='#it{N}(1 GeV) #rightarrow e+#pi'

        h = h1 if nametag == 'e+rho' else h2  

        histogram_groups = [
            (["calc_pion0_pt","truth_pion0_pt"], f"{title}: Pion0 Transverse Momentum", "Momentum (GeV/c)", "Entries"),
            (["sol1_calc_pion0_pp", "sol2_calc_pion0_pp","truth_pion0_pp"], f"{title}: Pion0 Parallel Momentum", "Momentum (GeV/c)", "Entries"),
            (["calcmassdist1", "calcmassdist2"], f"{title}: HNL Mass Calculation",  "Mass (GeV/c^{2})", "Entries")
        ]   

        for hist_keys, plot_title, x_label, y_label in histogram_groups:
            
            canvas = ROOT.TCanvas(f"Comparison_{nametag}_{plot_title.split(': ')[1]}", plot_title, 800, 600)
            
            legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.85)  # Legend in top-right corner
            
            first_hist = True  # Flag to handle first histogram drawing

            for i, hist_key in enumerate(hist_keys):
                
                hist_name = f"{nametag}_{hist_key}"
                
                if hist_name in h: #sanity check
                
                    hist = h[hist_name]
                    
                    hist.SetStats(0)
                    hist.SetLineColor(colors[i])  
                    hist.SetFillColor(colors[i])
                    hist.SetLineWidth(2)

                    if i%2==0:
                        hist.SetFillStyle(3005)
                    else:
                        hist.SetFillStyle(3004)
                    
                    if hist_key.startswith("truth"):
                        hist.SetFillStyle(3001)

                    
                    if first_hist:
                        hist.Draw("HIST")  # Draw first histogram
                        hist.GetXaxis().SetTitle(x_label)
                        hist.GetYaxis().SetTitle(y_label)
                        hist.SetTitle(f"Comparison of {plot_title}")
                        hist.GetYaxis().SetRangeUser(0.1, 50000)
                        first_hist = False
                    else:
                        hist.Draw("HIST SAME")  # Overlay subsequent histograms
                    legend.AddEntry(hist, hist_key.replace("_", " "), "f")            
            
            legend.Draw()
            canvas.Write() 

    output_file.Close()
    print(f"All plots have been saved in {filename}.")

def calculate_min_rho_mass(pion,lepton,pion0):
    """
    Calculate the minimum invariant mass of the rho meson allowed kinematically for the event.
    
    Returns:
    - rho_mass_min (in GeV)
    """
    pion_energy      = pion.E() 
    pion_mass   = pion.M()
    pion_phi    = pion.Phi()
    pion_pp     = pion.Pz()
    pion_pt     = pion.Pt()
    
    lepton_pt   = lepton.Pt()
    lepton_phi  = lepton.Phi()
    
    pion0_pt     = pion0.Pt()
    pion0_pp_min = pion_pp * ROOT.TMath.Sqrt((PION0_MASS**2 + pion0_pt**2) / (pion_energy**2 - pion_pp**2 )) #parallel component at rho_mass_min

    pion0_energy      = ROOT.TMath.Sqrt(pion0_pp_min**2 + pion0_pt**2 + PION0_MASS**2) 
    
    # Compute transverse dot product
    cos_phi_diff =  ROOT.TMath.Cos(lepton_phi - pion_phi) 
    dot_perp = - pion_pt**2 - pion_pt*lepton_pt * cos_phi_diff

    mass_squared = (pion_mass**2 + PION0_MASS**2
                    + 2 * pion_energy * pion0_energy
                    - 2 * pion_pp * pion0_pp_min
                    - 2 * dot_perp)
    
    return ROOT.TMath.Sqrt(mass_squared)

def rho_median_mass_from_min(rho_mass_min,width=0.149):
    """
    Calculate the median ρ mass given a lower bound (rho_mass_min),using the truncated Breit-Wigner distribution.
    
    Parameters:
    - rho_mass_min: allowed minimum mass of rho from event kinematics (in GeV)
    - width: width of rho (in GeV)
    
    Returns:
    - m_rho_median: median ρ mass (in GeV)
    """

    # Compute the CDF at rho_mass_min
    num     = rho_mass_min**2 - RHO_MASS_NOMINAL**2
    denom   = width * RHO_MASS_NOMINAL
    cdf     = (1 / ROOT.TMath.Pi()) * ROOT.TMath.ATan(num / denom) + 0.5

    # Find median rho from CDF value
    angle = ROOT.TMath.Pi() * ((1 + cdf) / 2 - 0.5)
    m_rho_med_squared   = RHO_MASS_NOMINAL**2 + width * RHO_MASS_NOMINAL * ROOT.TMath.Tan(angle)

    return ROOT.TMath.Sqrt(m_rho_med_squared)

def calculate_pion0_parallel(lepton_rotated,pion_rotated,calc_pion0_rotated,rho_mass=RHO_MASS_NOMINAL,use_mathematica = False):

    pion_pt     = pion_rotated.Pt()
    pion_phi    = pion_rotated.Phi()
    pion_mass   = pion_rotated.M()
    pion_pp     = pion_rotated.Pz() 
    pion_energy = pion_rotated.E()

    lepton_pt   = lepton_rotated.Pt()
    lepton_phi  = lepton_rotated.Phi()

    pion0_pt    =  calc_pion0_rotated.Pt()       

    energy_squared_diff = pion_energy**2 - pion_pp**2  #E^2-pp^2 = pT^2 + m^2
    
    cos_phi_diff =  ROOT.TMath.Cos(lepton_phi - pion_phi) 
    
    if use_mathematica:

        sqrt_inner_term = (
                            pion_energy**2 * (
                                              pion_mass**4 
                                              - 4 * pion_energy**2 * PION0_MASS**2
                                              + 2 * pion_mass**2   * PION0_MASS**2
                                              +     PION0_MASS**4 
                                              - 2 * pion_mass**2   * rho_mass**2
                                              - 2 * PION0_MASS**2  * rho_mass**2
                                              +     rho_mass**4
                                              + 4 * PION0_MASS**2  * pion_pp**2
                                              + 4 * pion_mass**2   * pion_pt**2
                                              + 4 * PION0_MASS**2  * pion_pt**2
                                              - 4 * rho_mass**2    * pion_pt**2
                                              + 2 * lepton_pt**2   * pion_pt**2
                                              + 4 * pion_pt**4
                                              - 4 * pion_energy**2 * pion0_pt**2
                                              + 4 * pion_pp**2     * pion0_pt**2
                                              + 4 * pion_pt * lepton_pt * cos_phi_diff *   (
                                                                                            pion_mass**2 
                                                                                            + PION0_MASS**2 
                                                                                            - rho_mass**2 
                                                                                            + 2 * pion_pt**2
                                                                                            ) 

                                              + 2 * pion_pt**2 * lepton_pt**2 *  ROOT.TMath.Cos(2 * (lepton_phi - pion_phi))
                                              )
                            )
    else:

        #A_term = rho_mass**2 - pion_mass**2 - PION0_MASS**2 - 2 * pion_pt * (pion_pt + lepton_pt)
        A_term = (rho_mass**2
          - pion_mass**2
          - PION0_MASS**2
          - 2 * (pion_pt**2 + pion_pt * lepton_pt * cos_phi_diff))
        
        alpha = energy_squared_diff
        beta = - A_term * pion_pp
        
        #gamma =- A_term **2 + pion_energy**2 * (PION0_MASS**2 + (pion_pt + lepton_pt) * (pion_pt + lepton_pt))
        # --- γ ---------------------------------------------------------------
        vec_sum_pt2 = (pion_pt**2
                       + lepton_pt**2
                       + 2 * pion_pt * lepton_pt * cos_phi_diff)

        gamma = -A_term**2 + pion_energy**2 * (PION0_MASS**2 + vec_sum_pt2)

        sqrt_inner_term = (beta**2 - 4 * alpha * gamma)
        
    
    term2_part2 =  ROOT.TMath.Sqrt(sqrt_inner_term)

    
    term1 = (pion_mass**2 * pion_pp  
            + PION0_MASS**2 * pion_pp 
            - rho_mass**2 * pion_pp  
            + 2 * pion_pp * pion_pt**2 
            + 2 * pion_pp * pion_pt * lepton_pt * cos_phi_diff
            )
    
    #term2 = term2_part2 / energy_squared_diff
    
    pion0_pp_min = -((1 / (2 * energy_squared_diff)) * (term1 + term2_part2))  
    pion0_pp_max = -((1 / (2 * energy_squared_diff)) * (term1 - term2_part2)) 
        
    return pion0_pp_min,pion0_pp_max
def rotate_momentum_to_hnl_frame(decay_vertex, production_vertex,vec):

    def calculate_momentum_in_new_coordinates(vector, x, y, z):
        return ROOT.TVector3(vector.Dot(x), vector.Dot(y), vector.Dot(z))   

    def get_unit_vector(vector):
        return vector * (1.0 / vector.Mag()) 
    
    # Define the z-axis as the direction from the production to the decay vertex of HNL
    z_axis = get_unit_vector(decay_vertex - production_vertex)

    # Choose a reference axis that is not collinear with z-axis
    orthogonal_to_z = ROOT.TVector3(1, 0, 0)
    if abs(orthogonal_to_z.Dot(z_axis)) > 0.9999:  
        orthogonal_to_z = ROOT.TVector3(0, 1, 0) 

    # Define the x and y axes for the perpendicular plane
    x_axis = get_unit_vector(z_axis.Cross(orthogonal_to_z))
    y_axis = get_unit_vector(z_axis.Cross(x_axis))

    # Calculate momentum vector in the new coordinate system
    transformed_momentum = calculate_momentum_in_new_coordinates(vec.Vect(), x_axis, y_axis, z_axis)    
    
    recalc_E = ROOT.TMath.Sqrt(transformed_momentum.Mag2() + vec.M()**2)  # E' = sqrt(p^2 + m^2) sanity check

    transformed_vec = ROOT.TLorentzVector()
    transformed_vec.SetPxPyPzE(transformed_momentum.X(), transformed_momentum.Y(), transformed_momentum.Z(), recalc_E)

    return transformed_vec

def reconstruct_hnl_mass(pion0_pp,lepton_rotated, pion_rotated, calc_pion0_rotated,rho_mass=RHO_MASS_NOMINAL):

    lepton_mass     = lepton_rotated.M()
    lepton_energy   = lepton_rotated.E()
    lepton_pp       = lepton_rotated.Pz()
    lepton_pt       = lepton_rotated.Pt()

    pion_energy     = pion_rotated.E()
    pion_pp         = pion_rotated.Pz()
    
    pion_lepton_dot = pion_rotated.Dot(lepton_rotated)
    
    pion0_pt        = calc_pion0_rotated.Pt()   
    pion0_p         = ROOT.TMath.Sqrt(pion0_pt**2 + pion0_pp**2)
    pion0_energy    = ROOT.TMath.Sqrt(pion0_p**2 + PION0_MASS**2)
    
    term1 = lepton_energy * pion0_energy
    term2 = abs(pion_pp)*abs(lepton_pp) + abs(pion0_pp) * abs(lepton_pp)
    
    term3 = 2*lepton_energy*pion_energy
    
    term4 = 2*(lepton_pt**2)

    HNL_reconmass = ROOT.TMath.Sqrt( 
                            lepton_mass**2 
                            + rho_mass**2 
                            + 2 * term1 
                            - 2 * term2 
                            + term3 
                            + term4
                            )

    return HNL_reconmass

def book_histograms(nametag):
    
    h = {}
    
    if nametag=='e+rho':
        title='#it{N}(1 GeV) #rightarrow e+#pi+#pi^{0}'
    if nametag=='e+pi':
        title='#it{N}(1 GeV) #rightarrow e+#pi'


    h[f"{nametag}_truth_rho_M"]           = ROOT.TH1D(f"{nametag}_mass_rho",        f"{title} ; m_{{#rho}} (GeV/c^{{2}});" , 100, 0.3, 1.3)
    h[f"{nametag}_HNL_decayvertex_z"]     = ROOT.TH1D(f"{nametag}_HNL_decayvertex_z", f"{title} ; z (cm);"                 , 1000, -3000, 3000)
    h[f"{nametag}_HNL_prodvertex_z"]      = ROOT.TH1D(f"{nametag}_HNL_prodvertex_z",  f"{title} ; z (cm);"                 , 146, -5888, -5742)

    h[f"{nametag}_truth_pion0_pt"]        = ROOT.TH1D(f"{nametag}_truth_pion0_pt",    f"{title} ; p_{{T}} (GeV/c);", 100, 0, 2)
    h[f"{nametag}_calc_pion0_pt"]         = ROOT.TH1D(f"{nametag}_calc_pion0_pt",     f"{title} ; p_{{T}} (GeV/c);", 100, 0, 2)

    h[f"{nametag}_truth_pion0_P_Mom"]     = ROOT.TH1D(f"{nametag}_truth_pion0_P_Mom", f"{title} ;p (GeV/c)                 ;", 300, -1500, 1500)
    h[f"{nametag}_truth_pion0_pp"]        = ROOT.TH1D(f"{nametag}_truth_pion0_pp",    f"{title} ;p_{{\\parallel}} (GeV/c)  ;", 300, -1500, 1500)
    h[f"{nametag}_sol1_calc_pion0_pp"]    = ROOT.TH1D(f"{nametag}_sol1_calc_pion0_pp",f"{title} ;p_{{\\parallel}} (GeV/c)  ;", 300, -1500, 1500)
    h[f"{nametag}_sol2_calc_pion0_pp"]    = ROOT.TH1D(f"{nametag}_sol2_calc_pion0_pp",f"{title} ;P_{{\\parallel}} (GeV/c)  ;", 300, -1500, 1500)

    h[f"{nametag}_calcmassdist1"]         = ROOT.TH1D(f"{nametag}_calcmassdist1",      f"{title} (#rho-constraint mass)    ;GeV/c^{{2}};", 75, 0, 2.5)
    h[f"{nametag}_calcmassdist2"]         = ROOT.TH1D(f"{nametag}_calcmassdist2",      f"{title} (#rho-constraint mass)    ;GeV/c^{{2}};", 75, 0, 2.5)
    h[f"{nametag}_invmassdist"]           = ROOT.TH1D(f"{nametag}_invmassdist",        f"{title} (M_{{inv}}(e#pi))         ;GeV/c^{{2}};", 75, 0, 2.5)
    h[f"{nametag}_truemassdist"]          = ROOT.TH1D(f"{nametag}_truemassdist",       f"{title} true mass                 ;GeV/c^{{2}};", 75, 0, 2.5)

    h[f"{nametag}_HNL_decayvertex_xy"]    = ROOT.TH2D(f"{nametag}_HNL_decayvertex_xy", f"{title} ; x (cm); y (cm)", 1200, -600, 600, 1200, -600, 600)

    for key in h:
        h[key].SetDirectory(ROOT.gROOT)

    return h

def extract_decay_event_ids_to_csv(signal_file):

    file = ROOT.TFile.Open(signal_file)    
    tree=file.Get("LLP_tree")

    rho_events=set()
    epi_events=set()
    for i,event in enumerate(tree):
        
        #HNL_pdg=event.pdg_llp #this is dummy value ignore.
        
        HNL_px  = event.px_llp
        HNL_py  = event.py_llp
        HNL_pz  = event.pz_llp
        HNL_En  = event.e_llp
        HNL_mass=event.mass_llp
        HNL_decayprob=event.decay_prob
        HNL_production_vertex=[event.vx,event.vy,event.vz]
        nDaughters=event.ndau

        #print(f"HNL info: \
        #\n\tHNL_pdg={event.pdg_llp} \
        #\n\tHNL_px={event.px_llp} \
        #\n\tHNL_py={event.py_llp} \
        #\n\tHNL_pz={event.pz_llp} \
        #\n\tHNL_En={event.e_llp} \
        #\n\tHNL_mass={event.mass_llp} \
        #\n\tHNL_decayprob={event.decay_prob} \
        #\n\tHNL_production_vertex={[event.vx,event.vy,event.vz]} \
        #\n\tnDaughters={event.ndau}")
        
        daughters=[]

        for daughter_id in range(1,int(event.ndau)+1):
            branch_name = f"pdg_prod{daughter_id}"
            d_pdg = getattr(event, branch_name)
            if d_pdg==-999: continue
            daughters.append(abs(int(d_pdg)))
            particlename=pdg.GetParticle(int(d_pdg)).GetName()
            #print(f"Daughter {daughter_id} PDG: {d_pdg}; {particlename}")
        
        if (11 in daughters) and (211 in daughters) and len(daughters)==2:
            epi_events.add(i)
        
        if (22 in daughters) and (11 in daughters) and (211 in daughters):
            rho_events.add(i)
    
    file.Close()

    print(f"e+rho decay nEvents:{len(rho_events)}")
    rho_events_sorted = sorted(rho_events)
    with open("e+rho_events.csv", mode="w", newline="") as file_:
        writer = csv.writer(file_)
        for event_id in rho_events_sorted:
            writer.writerow([event_id])
    
    print(f"e+pi decay nEvents:{len(epi_events)}")
    epi_events_sorted = sorted(epi_events)
    with open("e+pi_events.csv", mode="w", newline="") as file_:
        writer = csv.writer(file_)
        for event_id in epi_events_sorted:
            writer.writerow([event_id])

def process_hnl_events_with_rho_constraint(signal_file,nametag):    
        
    filename=f'{nametag}_events.csv'
    
    h = book_histograms(nametag)
    print(f"Starting Analysis for: {nametag}")

    event_list=[]
    
    with open(filename, mode="r", newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            event_id = int(row[0])
            event_list.append(event_id)

    file = ROOT.TFile.Open(signal_file)    
    
    tree=file.Get("LLP_tree")
    
    nNegative_radicands=0
    
    nSuccessful_candidates=0

    for i,eventindex in enumerate(event_list):
        
        rc=tree.GetEntry(eventindex)
        
        event=tree
        
        production_vertex = ROOT.TVector3(0,0,0) #center of target

        h[f"{nametag}_HNL_prodvertex_z"].Fill(production_vertex[2])

        decay_vertex = ROOT.TVector3(tree.vx, tree.vy,tree.vz)

        h[f"{nametag}_HNL_decayvertex_z"].Fill(decay_vertex[2])
        h[f"{nametag}_HNL_decayvertex_xy"].Fill(decay_vertex[0],decay_vertex[1])
        
        gamma_id=[]

        for daughter_id in range(1,int(event.ndau)+1):
            
            branch_name = f"pdg_prod{daughter_id}"
            d_pdg = getattr(event, branch_name)
            
            if d_pdg==-999: continue
            
            if abs(d_pdg) in [13,11]: 
                lepton_id = daughter_id
            if abs(d_pdg) ==211: 
                pion_id = daughter_id
            if abs(d_pdg) ==22: 
                gamma_id.append(daughter_id)
            
            #particlename=pdg.GetParticle(int(d_pdg)).GetName()
            #print(f"Daughter {daughter_id} PDG: {d_pdg}; {particlename}")


        lepton_vec = ROOT.TLorentzVector()
        
        lepton_vec.SetPxPyPzE(  getattr(event, f"px_prod{lepton_id}"),  
                                getattr(event, f"py_prod{lepton_id}"), 
                                getattr(event, f"pz_prod{lepton_id}"),
                                getattr(event, f"e_prod{lepton_id}") 
                                )

        lepton_rotated  =rotate_momentum_to_hnl_frame(decay_vertex, production_vertex, lepton_vec) 
        
        pion_vec = ROOT.TLorentzVector()
        
        pion_vec.SetPxPyPzE(    getattr(event, f"px_prod{pion_id}"),
                                getattr(event, f"py_prod{pion_id}"),
                                getattr(event, f"pz_prod{pion_id}"),
                                getattr(event, f"e_prod{pion_id}")
                                )

        pion_rotated    =rotate_momentum_to_hnl_frame(decay_vertex, production_vertex, pion_vec)

        if len(gamma_id)==2:

            gamma1 = ROOT.TLorentzVector()
            gamma2 = ROOT.TLorentzVector()

            gamma1.SetPxPyPzE(  getattr(event, f"px_prod{gamma_id[0]}"),
                                getattr(event, f"py_prod{gamma_id[0]}"),
                                getattr(event, f"pz_prod{gamma_id[0]}"),
                                getattr(event, f"e_prod{gamma_id[0]}")
                              )
            gamma2.SetPxPyPzE(  getattr(event, f"px_prod{gamma_id[1]}"),
                                getattr(event, f"py_prod{gamma_id[1]}"),
                                getattr(event, f"pz_prod{gamma_id[1]}"),
                                getattr(event, f"e_prod{gamma_id[1]}")
                            )
            
            truth_pion0_vec = gamma1 + gamma2
            
            truth_rho=(truth_pion0_vec+pion_vec)

            h[f"{nametag}_truth_rho_M"].Fill(truth_rho.M())

            truth_pion0_new=rotate_momentum_to_hnl_frame(decay_vertex, production_vertex, truth_pion0_vec) 
            
            h[f"{nametag}_truth_pion0_P_Mom"].Fill(truth_pion0_new.P())

            h[f"{nametag}_truth_pion0_pt"].Fill(truth_pion0_new.Pt())
            h[f"{nametag}_truth_pion0_pp"].Fill(truth_pion0_new.Pz())


        calc_pion0_rotated = -(pion_rotated + lepton_rotated)
        
        h[f"{nametag}_calc_pion0_pt"].Fill(calc_pion0_rotated.Pt())            
        
        sol1_pion0_pp,sol2_pion0_pp=calculate_pion0_parallel(lepton_rotated,pion_rotated,calc_pion0_rotated)
        
        if ROOT.TMath.IsNaN(sol1_pion0_pp) or ROOT.TMath.IsNaN(sol2_pion0_pp):
            rho_mass_min=calculate_min_rho_mass(pion_rotated,lepton_rotated, calc_pion0_rotated)
            rho_median=rho_median_mass_from_min(rho_mass_min)
            sol1_pion0_pp,sol2_pion0_pp=calculate_pion0_parallel(lepton_rotated,pion_rotated,calc_pion0_rotated,rho_mass=rho_median)

        h[f"{nametag}_sol1_calc_pion0_pp"].Fill(sol1_pion0_pp)
        h[f"{nametag}_sol2_calc_pion0_pp"].Fill(sol2_pion0_pp)

        HNLmass1=reconstruct_hnl_mass(sol1_pion0_pp,lepton_rotated, pion_rotated,calc_pion0_rotated)
        HNLmass2=reconstruct_hnl_mass(sol2_pion0_pp,lepton_rotated, pion_rotated,calc_pion0_rotated)

        if ROOT.TMath.IsNaN(HNLmass1) or ROOT.TMath.IsNaN(HNLmass2):            
            nNegative_radicands+=1
            continue

        truemass = event.mass_llp

        h[f"{nametag}_truemassdist" ].Fill(truemass)
        h[f"{nametag}_calcmassdist1"].Fill(HNLmass1)
        h[f"{nametag}_calcmassdist2"].Fill(HNLmass2)

        lv_HNL = pion_vec + lepton_vec
        
        inv_mass=lv_HNL.M()
        
        h[f"{nametag}_invmassdist" ].Fill(inv_mass)

        nSuccessful_candidates += 1

    file.Close()
    
    print("nEvents available:",len(event_list),"\tEvents Successful=",nSuccessful_candidates)
    
    if nNegative_radicands:
        print("nNegative_radicands!!", nNegative_mass)
    
    return h


def main():

    parser = ArgumentParser()

    parser.add_argument("-sg", "--signalpath", dest="signal_path", help="Path to Signal Files", required=False,type=str, default= \
        '/eos/user/m/mfership/public/SHiP/MC/HNL_e/HNL_1.000e+00_6.000e-03_1.000e+00_0.000e+00_0.000e+00_data.root')

    options = parser.parse_args()

    extract_decay_event_ids_to_csv(options.signal_path)

    h_lrho = process_hnl_events_with_rho_constraint(options.signal_path,'e+rho')
    h_epi  = process_hnl_events_with_rho_constraint(options.signal_path,'e+pi')

    plot_distributions(h_lrho,h_epi)

if __name__ == "__main__":
    main()