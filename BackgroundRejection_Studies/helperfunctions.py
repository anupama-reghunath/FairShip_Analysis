#!/usr/bin/env python
"""Toolkit for Analysis."""

import numpy as np
import pythia8_conf
import ROOT
import shipunit as u
from rootpyPickler import Unpickler
import yaml
from ShipGeoConfig import AttrDict
from tabulate import tabulate
import geomGeant4
from array import array
import joblib
import torch
from sbtveto.model.nn_model import NN
from sbtveto.util.inference import nn_output
from sbtveto.model.gnn_model import EncodeProcessDecode
from sbtveto.util.inference import gnn_output_binary,gnn_output_binary_wdeltaT
#from torch_geometric.data import Data
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_DATA_DIR  = _REPO_ROOT / "sbtveto" / "data"

class AnalysisContext:
    def __init__(self, tree, geo_file):

        self.tree = tree
        
        #Initialize geometry configuration.
        self.geometry_manager = geo_file.Get("FAIRGeom")

        
        unpickler = Unpickler(geo_file)
        self.ship_geo = unpickler.load("ShipGeo")

        fairship = ROOT.gSystem.Getenv("FAIRSHIP")

        if self.ship_geo.DecayVolumeMedium == "helium":
            with open(fairship + "/geometry/veto_config_helium.yaml") as file:
                config = yaml.safe_load(file)
                self.veto_geo = AttrDict(config)
        
        if self.ship_geo.DecayVolumeMedium == "vacuums":
            with open(fairship + "/geometry/veto_config_vacuums.yaml") as file:
                config = yaml.safe_load(file)
                self.veto_geo = AttrDict(config)
        
        #medium   = self.ship_geo.DecayVolumeMedium
        #fn       = f"{fairship}/geometry/veto_config_{medium}.yaml"
        #with open(fn) as f:
        #    self.veto_geo = AttrDict(yaml.safe_load(f))

        self.fieldMaker = geomGeant4.addVMCFields(self.ship_geo, '', True, withVirtualMC = False)
        geoMat =  ROOT.genfit.TGeoMaterialInterface()
        ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
        bfield = ROOT.genfit.FairShipFields()
        bfield.setField(self.fieldMaker.getGlobalField())
        self.fM = ROOT.genfit.FieldManager.getInstance()
        self.fM.init(bfield)

        self.fitter = ROOT.genfit.DAF()   # good
        self.fitter.setMaxIterations(50)  # good, default is 10
        

class selection_check:
    """Class to perform various selection checks on the candidate."""

    def __init__(self, ctx: AnalysisContext):
        self.tree             = ctx.tree
        self.geometry_manager = ctx.geometry_manager
        self.ship_geo         = ctx.ship_geo
        self.veto_geo         = ctx.veto_geo


    def define_candidate_time(self, candidate,offset=0):
        """Calculate time associated with the candidate decay vertex using strawtubes MCPoint info."""

        t0 = self.tree.ShipEventHeader.GetEventTime()

        candidate_pos = ROOT.TVector3()
        candidate.GetVertex(candidate_pos)

        d1, d2 = candidate.GetDaughter(0), candidate.GetDaughter(1)
        d1_mc, d2_mc = self.tree.fitTrack2MC[d1], self.tree.fitTrack2MC[d2]

        time_vtx_from_strawhits = []

        for hit in self.tree.strawtubesPoint:
            if not (
                int(str(hit.GetDetectorID())[:1]) == 1
                or int(str(hit.GetDetectorID())[:1]) == 2
            ):
                continue

            if not (hit.GetTrackID() == d1_mc or hit.GetTrackID() == d2_mc):
                continue

            t_straw = hit.GetTime()

            dist = np.sqrt(
                (candidate_pos.X() - hit.GetX()) ** 2
                + (candidate_pos.Y() - hit.GetY()) ** 2
                + (candidate_pos.Z() - hit.GetZ()) ** 2
            )  # distance to the vertex in cm

            d_mom = self.tree.MCTrack[hit.GetTrackID()].GetP() / u.GeV
            mass = self.tree.MCTrack[hit.GetTrackID()].GetMass()
            v = u.c_light * d_mom / np.sqrt(d_mom**2 + (mass) ** 2)

            t_vertex = t_straw - (dist / v)

            time_vtx_from_strawhits.append(t_vertex)

        t_vtx = np.average(time_vtx_from_strawhits) + t0 + offset

        return t_vtx  # units in ns

    def impact_parameter(self, candidate):
        """Calculate the impact parameter of the candidate relative to (0,0,target.z0)."""
        candidate_pos = ROOT.TVector3()
        candidate.GetVertex(candidate_pos)

        candidate_mom = ROOT.TLorentzVector()
        candidate.Momentum(candidate_mom)
        target_point = ROOT.TVector3(0, 0, self.ship_geo.target.z0)

        projection_factor = 0
        if hasattr(candidate_mom, "P"):
            P = candidate_mom.P()
        else:
            P = candidate_mom.Mag()
        for i in range(3):
            projection_factor += (
                candidate_mom(i) / P * (target_point(i) - candidate_pos(i))
            )

        dist = 0
        for i in range(3):
            dist += (
                target_point(i)
                - candidate_pos(i)
                - projection_factor * candidate_mom(i) / P
            ) ** 2
        dist = ROOT.TMath.Sqrt(dist)

        return dist  # in cm

    def dist_to_innerwall(self, candidate):
        """Calculate the minimum distance(in XY plane) of the candidate decay vertex to the inner wall of the decay vessel. If outside the decay volume, or if distance > 100m, return 0."""
        candidate_pos = ROOT.TVector3()
        candidate.GetVertex(candidate_pos)
        position = (candidate_pos.X(), candidate_pos.Y(), candidate_pos.Z())

        nsteps = 8
        dalpha = 2 * ROOT.TMath.Pi() / nsteps
        min_distance = float("inf")

        node = self.geometry_manager.FindNode(*position)
        if not node:
            return 0  # is outside the decay volume

        # Loop over directions in the XY plane
        for n in range(nsteps):
            alpha = n * dalpha
            direction = (
                ROOT.TMath.Sin(alpha),
                ROOT.TMath.Cos(alpha),
                0.0,
            )  # Direction vector in XY plane
            self.geometry_manager.InitTrack(*position, *direction)
            if not self.geometry_manager.FindNextBoundary():
                continue
            # Get the distance to the boundary and update the minimum distance
            distance = self.geometry_manager.GetStep()
            min_distance = min(min_distance, distance)

        return min_distance if min_distance < 100 * u.m else 0

    def dist_to_vesselentrance(self, candidate):
        """Calculate the distance of the candidate decay vertex to the entrance of the decay vessel."""
        candidate_pos = ROOT.TVector3()
        candidate.GetVertex(candidate_pos)
        return candidate_pos.Z() - self.veto_geo.z0

    def nDOF(self, candidate):
        """Return the number of degrees of freedom (nDOF) for the particle's daughter tracks."""
        nmeas = []
        t1, t2 = candidate.GetDaughter(0), candidate.GetDaughter(1)

        for tr in [t1, t2]:
            fit_status = self.tree.FitTracks[tr].getFitStatus()
            nmeas.append(
                int(round(fit_status.getNdf()))
            )  # nmeas.append(fit_status.getNdf())

        return np.array(nmeas)

    def daughtermomentum(self, candidate):
        """Return the momentum(Mag) of the particle's daughter tracks."""
        daughter_mom = []
        t1, t2 = candidate.GetDaughter(0), candidate.GetDaughter(1)
        for trD in [t1, t2]:
            x = self.tree.FitTracks[trD]
            xx = x.getFittedState()
            daughter_mom.append(xx.getMom().Mag())

        return np.array(daughter_mom)

    def invariant_mass(self, candidate):
        """Invariant mass of the candidate."""
        return candidate.GetMass()

    def DOCA(self, candidate):
        """Distance of Closest Approach."""
        return candidate.GetDoca()

    def is_in_fiducial(self, candidate):
        """Check if the candidate is within the Fiducial Volume and has hits in all four tracking stations"""

        def tracks_in_fiducial(t1, t2):
            """
            Return True if BOTH daughter tracks (t1, t2) have hits in all
            four straw‐tube stations (1, 2, 3, 4).  Return False otherwise.
            """
            required_stations = {1, 2, 3, 4}

            for track_index in (t1, t2):

                mc_id = self.tree.fitTrack2MC[track_index]

                seen_stations = set()
                for hit in self.tree.strawtubesPoint:
                    if hit.GetTrackID() == mc_id:
                        det_id_str = str(hit.GetDetectorID())
                        station = int(det_id_str[0])
                        seen_stations.add(station)
                
                if not required_stations.issubset(seen_stations):
                    return False
            return True #both tracks are fine

        candidate_pos = ROOT.TVector3()
        candidate.GetVertex(candidate_pos)

        if candidate_pos.Z() > self.ship_geo.TrackStation1.z:
            return False
        if candidate_pos.Z() < self.veto_geo.z0:
            return False

        # if self.dist2InnerWall(candidate)<=5*u.cm: return False

        vertex_node = ROOT.gGeoManager.FindNode(
            candidate_pos.X(), candidate_pos.Y(), candidate_pos.Z()
        )
        vertex_elem = vertex_node.GetVolume().GetName()
        if not vertex_elem.startswith("DecayVacuum_"):
            return False

        t1, t2 = candidate.GetDaughter(0), candidate.GetDaughter(1)
        if not tracks_in_fiducial(t1, t2):
            return False
        return True

    def chi2nDOF(self, candidate):
        """Return the reduced chi^2 of the particle's daughter tracks."""
        t1, t2 = candidate.GetDaughter(0), candidate.GetDaughter(1)

        chi2ndf = []
        for tr in [t1, t2]:
            fit_status = self.tree.FitTracks[tr].getFitStatus()
            chi2ndf.append(fit_status.getChi2() / fit_status.getNdf())

        return np.array(chi2ndf)

    def pid_decision(self,candidate):
        
        """
        Interim solution for PID check:Uses track truth info assuming 100% efficiency in CaloPID
        

        pid_code: 0     = hadronic,
                  1     = dileptonic(any leptons),
                  1.1   = dileptonic ee,
                  1.2   = dileptonic μμ,         
                  2     = semileptonic(any lepton),
                  2.1   = semileptonic containing an e,
                  2.2   = semileptonic containing a μ,
                  3     = at least one track has unknown PID, #never used since truth but added for historical reasons
                  4     = fewer than two PID tracks available #two track candidates

        """

        if(len(self.tree.Pid)<2):
            print("Pid is less  than 2 particles!") #sanity check?
            return 4

        
        d1_mc=self.tree.MCTrack[self.tree.fitTrack2MC[candidate.GetDaughter(0)]]
        d1_pdg=d1_mc.GetPdgCode()

        d2_mc=self.tree.MCTrack[self.tree.fitTrack2MC[candidate.GetDaughter(1)]]
        d2_pdg=d2_mc.GetPdgCode()
        

        LEPTON_PDGS = {11, 13}      # 11 = electron, 13 = muon

        d1_is_lepton = abs(d1_pdg) in LEPTON_PDGS
        d2_is_lepton = abs(d2_pdg) in LEPTON_PDGS

        d1_is_mu= (abs(d1_pdg)==13)
        d2_is_mu= (abs(d2_pdg)==13)
        
        d1_is_e= (abs(d1_pdg)==11)
        d2_is_e= (abs(d2_pdg)==11)            

        
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

    def preselection_cut(self, candidate, IP_cut=250, show_table=False,offset=None):
        """
        Umbrella method to apply the pre-selection cuts on the candidate.

        show_table=True tabulates the pre-selection parameters.
        """
        flag = True

        if len(self.tree.Particles) != 1:
            flag = False
        if not (self.is_in_fiducial(candidate)):
            flag = False
        if self.dist_to_innerwall(candidate) <= 5 * u.cm:
            flag = False
        if self.dist_to_vesselentrance(candidate) <= 20 * u.cm:
            flag = False
        if self.impact_parameter(candidate) >= IP_cut * u.cm:
            flag = False
        if self.DOCA(candidate) >= 1 * u.cm:
            flag = False
        if np.any(self.nDOF(candidate) <= 25):
            flag = False
        if np.any(self.chi2nDOF(candidate) >= 5):
            flag = False
        if np.any(self.daughtermomentum(candidate) <= 1 * u.GeV):
            flag = False

        if show_table:
            table = [
                [
                    "Number of candidates in event",
                    len(self.tree.Particles),
                    "==1",
                    len(self.tree.Particles) == 1,
                ],
                [
                    "Time @ decay vertex (ns)",
                    self.define_candidate_time(candidate,offset),
                    "",
                    "",
                ],
                [
                    "Impact Parameter (cm)",
                    self.impact_parameter(candidate),
                    f"IP < {IP_cut * u.cm} cm",
                    self.impact_parameter(candidate) < IP_cut * u.cm,
                ],
                [
                    "DOCA (cm)",
                    self.DOCA(candidate),
                    "DOCA < 1 cm",
                    self.DOCA(candidate) < 1 * u.cm,
                ],
                [
                    "Is within Fiducial Volume?",
                    self.is_in_fiducial(candidate),
                    "==True",
                    self.is_in_fiducial(candidate),
                ],
                [
                    "Dist2InnerWall (cm)",
                    self.dist_to_innerwall(candidate),
                    "> 5 cm",
                    self.dist_to_innerwall(candidate) > 5 * u.cm,
                ],
                [
                    "Dist2VesselEntrance (cm)",
                    self.dist_to_vesselentrance(candidate),
                    "> 20 cm",
                    self.dist_to_vesselentrance(candidate) > 20 * u.cm,
                ],
                ["Invariant Mass (GeV)", self.invariant_mass(candidate), "", ""],
                [
                    "Daughter Momentum [d1, d2] (GeV)",
                    self.daughtermomentum(candidate),
                    "> 1 GeV",
                    np.all(self.daughtermomentum(candidate) > 1 * u.GeV),
                ],
                [
                    "Degrees of Freedom [d1, d2]",
                    self.nDOF(candidate),
                    "> 25",
                    np.all(self.nDOF(candidate) > 25),
                ],
                [
                    "Reduced Chi^2 [d1, d2]",
                    self.chi2nDOF(candidate),
                    "< 5",
                    np.all(self.chi2nDOF(candidate) < 5),
                ],
                ["\033[1mPre-selection passed:\033[0m", "", "", flag],
            ]

            for row in table:
                row[3] = (
                    f"\033[1;32m{row[3]}\033[0m"
                    if row[3]
                    else f"\033[1;31m{row[3]}\033[0m"
                )  # Green for True, Red for False

            print(
                tabulate(
                    table,
                    headers=[
                        "Parameter",
                        "Value",
                        "Pre-selection cut",
                        "Pre-selection Check",
                    ],
                    tablefmt="grid",
                )
            )
        return flag


class event_inspector:
    """Class to inspect MCtruth of an Event."""

    def __init__(self, ctx: AnalysisContext):
        """Initialize ROOT PDG database."""
        
        self.geometry_manager = ctx.geometry_manager
        self.ship_geo         = ctx.ship_geo
        self.veto_geo         = ctx.veto_geo
        
        self.pdg = ROOT.TDatabasePDG.Instance()
        pythia8_conf.addHNLtoROOT()
        self.event= ctx.tree
        self.fitter=ctx.fitter

    def dump_event(self, track_p_threshold=0):
        """Dump the MCtruth of the event."""
        headers = [
            "#",
            "particle",
            "pdgcode",
            "mother_id",
            "Momentum [Px,Py,Pz] (GeV/c)",
            "StartVertex[x,y,z] (m)",
            "Process",
            "GetWeight()",
        ]

        event_table = []
        for trackNr, track in enumerate(self.event.MCTrack):
            if track.GetP() / u.GeV < track_p_threshold:
                continue

            particle = self.pdg.GetParticle(track.GetPdgCode())
            particlename = particle.GetName() if particle else "----"

            event_table.append(
                [
                    trackNr,
                    particlename,
                    track.GetPdgCode(),
                    track.GetMotherId(),
                    f"[{track.GetPx() / u.GeV:7.3f},{track.GetPy() / u.GeV:7.3f},{track.GetPz() / u.GeV:7.3f}]",
                    f"[{track.GetStartX() / u.m:7.3f},{track.GetStartY() / u.m:7.3f},{track.GetStartZ() / u.m:7.3f}]",
                    track.GetProcName().Data(),
                    track.GetWeight(),
                ]
            )

        print(
            tabulate(
                event_table, headers=headers, floatfmt=".3f", tablefmt="simple_outline"
            )
        )


class veto_tasks():
    "Class derived from ShipVeto WIP"
    def __init__(self, ctx: AnalysisContext):

        self.SBTefficiency = 0.99  # Surrounding Background tagger: 99% efficiency picked up from TP
        self.UBTefficiency = 0.9   # Upstream background tagger
        self.random = ROOT.TRandom(13)
        
        self.tree = ctx.tree
        self.sel = selection_check(ctx)
    
    def _cell_fired(self, detID):
        
        """
        Decide whether an SBT detector cell is “live” in the current event at a given self.SBTefficiency.
        The function performs a Bernoulli trial with efficiency.  
        Fixed random seed ensures that every veto algorithm sees the same live/dead state for a given cell for a given event.
        """

        evt_time_ns = int(self.tree.ShipEventHeader.GetEventTime())  

        seed = (evt_time_ns * 131071 + detID) & 0x7FFFFFFF           #some unique id 
        ROOT.gRandom.SetSeed(seed)

        return ROOT.gRandom.Rndm() < self.SBTefficiency               
    

    def SBTcell_map(self): #provides a cell map with index in [0,nCells] for each cell.
       fGeo = ROOT.gGeoManager
       detList = {}
       LiSC = fGeo.GetTopVolume().GetNode('DecayVolume_1').GetVolume().GetNode('T2_1').GetVolume().GetNode('VetoLiSc_0')
       index = -1
       for LiSc_cell in LiSC.GetVolume().GetNodes():
          index += 1
          name = LiSc_cell.GetName()
          detList[index] = name[-6:]
       return detList

    def SBT_decision(self,mcParticle=None,detector='liquid',threshold=45,offset=0,Digi_SBTHits=None):
        """Implementation of Basic SBT veto. """

        hitSegments = []
        index = -1
        fdetector = detector=='liquid'

        if Digi_SBTHits==None:
            Digi_SBTHits=self.tree.Digi_SBTHits
        for aDigi in Digi_SBTHits:        
         
         index+=1 
         
         if not self._cell_fired(aDigi.GetDetectorID()):
            continue
         
         detID    = aDigi.GetDetectorID()

         if fdetector and detID > 999999:continue
         if not fdetector and not detID > 999999:continue 
         if mcParticle:
            found = False
            for mcP in self.tree.digiSBT2MC[index]: 
             if mcParticle>0 and mcParticle != mcP : found=True
             if mcParticle<0 and abs(mcParticle) == mcP : found=True
            if found: continue
         position = aDigi.GetXYZ()
         ELoss    = aDigi.GetEloss()

         if ELoss>=threshold*0.001: hitSegments.append(index)

        veto = len(hitSegments) > 0 

        return veto, None, hitSegments #hitSegments contain the index of the Digihit that causes the veto

    def UBT_decision(self):
      """Implementation of UBT veto. Simple MC check;  no efficiency, no mom. check """
      nHits = 0
      mom = ROOT.TVector3()
      for ahit in self.tree.UpstreamTaggerPoint:
         ahit.Momentum(mom)
         #if mom.Mag() > 0.1:
         nHits+=1
      #w = (1 - self.UBTefficiency) ** nHits
      #veto = self.random.Rndm() > w
      if nHits:
        veto=True  
      else:
        veto=False 
      return veto,nHits

    def extrapolateTrackToSBT(self, fitIndex, tol_cm=320.0, back_dist_m=60, n_steps=300,Digi_SBTHits=None):
        """
        
        Extrapolate a fitted GenFit track backwards onto SBT.
        Uniformly sample the trajectory in n_steps, stop when we first enter 
        any LiSc volume, and then match to the nearest digi hits within the tolerance (tol_cm).

        Returns:
          best_hits, xs, ys, zs
        
        """
        """
        if not self.fM:
            geoMat =  ROOT.genfit.TGeoMaterialInterface()
            ROOT.genfit.MaterialEffects.getInstance().init(geoMat)
            bfield = ROOT.genfit.FairShipFields()
            bfield.setField(self.fieldMaker.getGlobalField())
            self.fM = ROOT.genfit.FieldManager.getInstance()
            self.fM.init(bfield)

        """
        track = self.tree.FitTracks[fitIndex]
        fst   = track.getFitStatus()
        if not (fst.isFitConverged() and fst.getNdf()>0):
            return [], [], [], []

        # get fitted state & build the RK rep
        fstate = track.getFittedState(0)
        pos0   = fstate.getPos()
        mom0   = fstate.getMom()
        rep    = ROOT.genfit.RKTrackRep(fstate.getPDG())
        state  = ROOT.genfit.StateOnPlane(rep)
        rep.setPosMom(state, pos0, mom0)

        
        nav = ROOT.gGeoManager.GetCurrentNavigator()
        
        # point & direction for backwards stepping
        dx, dy, dz = (-mom0.Unit()).X(), (-mom0.Unit()).Y(), (-mom0.Unit()).Z()
        nav.InitTrack(pos0.X(), pos0.Y(), pos0.Z(), dx, dy, dz)

        # sample the trajectory
        back_cm = back_dist_m * u.m
        ds      = -back_cm / float(n_steps)
        xs = []; ys = []; zs = []
        predPos = None

        for i in range(n_steps+1):
            p = state.getPos()
            xs.append(p.X()); ys.append(p.Y()); zs.append(p.Z())

            node = nav.FindNode(p.X(), p.Y(), p.Z())
            if node and node.GetName().startswith(("LiSc", "VetoInnerWall", "VetoOuterWall","VetoVerticalRib","VetoLongitRib")):
                predPos = p
                predMom = state.getMom()
                break

            target = p + state.getMom().Unit()*ds # step one chunk backwards
            try:
                rep.extrapolateToPoint(state, target, False)
            except Exception as e:
                print(f"Exception at step {i}: {e}")
                break
            

        # if we never hit LiSc, still push to the first boundary
        
        if predPos is None:
            # reset & do one boundary‐stop propagate
            rep.setPosMom(state, pos0, mom0)
            full_target = pos0 + mom0.Unit()*ds*n_steps
            
            rep.extrapolateToPoint(state, full_target, True)
            predPos = state.getPos()
            predMom = state.getMom()
            return [], xs, ys, zs

        # match the nearest SBT hits
        
        hits_in_tol = []      # will hold tuples of (hit) within the 320 cm of the track 
        
        if Digi_SBTHits==None:
            Digi_SBTHits=self.tree.Digi_SBTHits
        

        for hit in Digi_SBTHits:
            
            if not self._cell_fired(hit.GetDetectorID()):
                continue
            
            d = (hit.GetXYZ() - predPos).Mag()
            if d < tol_cm:

                hits_in_tol.append(hit)

        return hits_in_tol, xs, ys, zs

    def Veto_decision_GNNbinary_wdeltaT(self,candidate=None,threshold=0.6,offset=0,Digi_SBTHits=None):
        """
        Binary SBT veto: returns (veto, P(background)) using energy, deltaT, vertex info.
        """

        # --- Geometry ---
        XYZ = np.load(_DATA_DIR / "SBT_new_geo_XYZ.npy")   # (3, 854)
        
        # --- Load Model ---
        model = EncodeProcessDecode(
            mlp_output_size=8,
            global_op=1,
            num_blocks=4
        )

        if not hasattr(self, "_gnn_bin_loaded"):
            model.load_state_dict(torch.load(_DATA_DIR / "GNN_SBTveto_BINARY_45MeV_25epochs_wdeltaT.pth", map_location="cpu",weights_only=False))
            model.eval()
            self._gnn_bin_loaded = model
        else:
            model = self._gnn_bin_loaded

        # --- Prepare Input Features ---
        detList = self.SBTcell_map()
        energy_array = np.zeros(854, dtype=np.float32)
        delta_t      = np.full(854, -9999.0, dtype=np.float32)

        if Digi_SBTHits==None:
            Digi_SBTHits=self.tree.Digi_SBTHits

        for aDigi in Digi_SBTHits:

            if not self._cell_fired(aDigi.GetDetectorID()):
                    continue

            detID = str(aDigi.GetDetectorID())
            idx   = [i for i, v in detList.items() if v == detID][0]
            if idx < 854:
                energy_array[idx] = aDigi.GetEloss()
                delta_t[idx] = aDigi.GetTDC()

        # --- Candidate vertex info ---
        if candidate is None:
            candidate = self.tree.Particles[0]
        
        cand_pos = ROOT.TLorentzVector()
        candidate.ProductionVertex(cand_pos)
        vertex_time = self.sel.define_candidate_time(candidate,offset)  # already in ns
        vertex_xyz  = np.array([cand_pos.X(), cand_pos.Y(), cand_pos.Z()], dtype=np.float32)

        # --- Δt = hit_time - vertex_time ---
        delta_t = delta_t - vertex_time

        # --- Apply energy and Δt masking (threshold in MeV) ---
        threshold_MeV = 45
        low_energy_mask = energy_array < (threshold_MeV * 1e-3)
        bad_deltat_mask = (delta_t < -150) | (delta_t > 150)

        invalid_mask = low_energy_mask | bad_deltat_mask
        energy_array[invalid_mask] = 0.0
        delta_t[invalid_mask]      = -9999

        # --- Check if any valid cell remains ---
        if not np.any(energy_array > 0.0):
            #print("\t\tNo fired cell → classify as signal and return")
            return False, 0.0

        # --- Construct input matrix: energy, deltaT, vertexX,vertexY,vertexZ ---
        features = np.stack([
            energy_array,
            delta_t,
            np.full(854, vertex_xyz[0], dtype=np.float32),
            np.full(854, vertex_xyz[1], dtype=np.float32),
            np.full(854, vertex_xyz[2], dtype=np.float32)
        ], axis=1)  # shape: (854, 5)

        inputmatrix = features[np.newaxis, :, :]  # shape: (1, 854, 5)

        # --- Run binary GNN decision ---
        veto, prob_bg = gnn_output_binary_wdeltaT(model, inputmatrix, XYZ, threshold)
        
        return veto, prob_bg
