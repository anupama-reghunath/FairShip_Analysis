#!/usr/bin/env python3
"""
Merge *_pos TTrees across many jobs and generate one figure per selection (tree name).
- Saves plots into a dedicated folder: plots_<sample>/.
"""

import sys, glob, re
from pathlib import Path
from argparse import ArgumentParser

import ROOT
ROOT.gROOT.SetBatch(True)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon, Rectangle

# ---------------- style & geometry ----------------
def load_style():
    # Keep your house style for cohesive look
    style_path = Path('/afs/cern.ch/user/a/anupamar/Analysis/Tools/thesis.mplstyle')
    try:
        plt.style.use(str(style_path))
    except Exception:
        plt.style.use('default')

def add_geometry(ax_zx, ax_zy, ax_xy, show_detectors=True):
    """Draw the SHiP DV views exactly like your eventDisplay_mini."""
    z_start, z_end = -2500, 2500

    # x vs z (top view)
    ax_zx.add_patch(Polygon([(z_start,-74),(z_start,74),(z_end,224),(z_end,-224)],
                            fill=True, facecolor='#e6f2fa', edgecolor='#6baed6',
                            linewidth=2, linestyle='-'))
    # y vs z (side view)
    ax_zy.add_patch(Polygon([(z_start,-159),(z_start,159),(z_end,324),(z_end,-324)],
                            fill=True, facecolor='#e6f2fa', edgecolor='#6baed6',
                            linewidth=2, linestyle='-'))
    # x vs y (back view): mirrored trapezoids + connectors
    back = [(-74,-159),(-74,159),(-224,324),(-224,-324)]
    backR = [(-x,y) for x,y in back]
    ax_xy.add_patch(Polygon(back,  fill=True, facecolor='#e6f2fa', edgecolor='#6baed6', linewidth=2))
    ax_xy.add_patch(Polygon(backR, fill=True, facecolor='#e6f2fa', edgecolor='#6baed6', linewidth=2))
    ax_xy.add_patch(Polygon([(-74,159),(74,159),(224,324),(-224,324)],
                            fill=True, facecolor='#e6f2fa', edgecolor='#6baed6', linewidth=2))
    ax_xy.add_patch(Polygon([(-74,-159),(74,-159),(224,-324),(-224,-324)],
                            fill=True, facecolor='#e6f2fa', edgecolor='#6baed6', linewidth=2))

    if not show_detectors:
        return

    detectors = {
        'Tr1_1':[2588,2608,-241.57,241.57,-333.28,333.28],
        'Tr2_2':[2788,2808,-241.57,241.57,-333.28,333.28],
        'ShipMagnet_1':[2902,3234,-274.66,274.66,-335.93,335.93],
        'Tr3_3':[3328,3348,-274.66,274.66,-335.93,335.93],
        'Tr4_4':[3528,3548,-274.66,274.66,-335.93,335.93],
        'Timing Detector_1':[3605,3610,-259,259,-331,331],
    }
    colors = {'Tr1_1':'#e6550d','Tr2_2':'#e6550d','ShipMagnet_1':'gray',
              'Tr3_3':'#e6550d','Tr4_4':'#e6550d','Timing Detector_1':'#31a354'}
    for n,(z0,z1,x0,x1,y0,y1) in detectors.items():
        ax_zx.add_patch(Rectangle((z0,x0), z1-z0, x1-x0, fill=False,
                                  edgecolor=colors.get(n,'black'), linewidth=2, alpha=0.4))
        ax_zy.add_patch(Rectangle((z0,y0), z1-z0, y1-y0, fill=False,
                                  edgecolor=colors.get(n,'black'), linewidth=2, alpha=0.4))

def safe_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s)

parser = ArgumentParser(description=__doc__)

parser.add_argument("--path", dest="main_path",help="parent path", required=False, default='/eos/experiment/ship/user/anupamar/BackgroundStudies/alt_v2/')

group1 = parser.add_mutually_exclusive_group(required=True)

group1.add_argument("--muonDIS",dest="foldername", const="muonDIS",help="muonDIS Background studies", action="store_const")
group1.add_argument("--neuDIS", dest="foldername", const="neuDIS" ,help="neuDIS Background studies",  action="store_const")

group1.add_argument("--mupi",dest="foldername",   const="signalEventCalc/mupi"   ,help="signal (e π) studies",   action="store_const")
group1.add_argument("--erho",dest="foldername",   const="signalEventCalc/erho"   ,help="signal (e ρ) studies",   action="store_const")
group1.add_argument("--murho",dest="foldername",  const="signalEventCalc/murho"  ,help="signal (μ ρ) studies",   action="store_const")
group1.add_argument("--mumuv",dest="foldername",  const="signalEventCalc/mumuv"  ,help="signal (μ μ ν) studies", action="store_const")


group3 = parser.add_mutually_exclusive_group(required=False)

group3.add_argument("--fullreco"    , dest="analysis_channel", action= "store_const",const="fullreco"    ,help="Background studies for fully reco. (l π) channel")
group3.add_argument("--partialreco" , dest="analysis_channel", action= "store_const",const="partialreco" ,help="Background studies for partial reco. (l l ν) channel")
group3.add_argument("--leptonrho"   , dest="analysis_channel", action= "store_const",const="leptonrho"   ,help="Background studies for partial reco. (l ρ) channel")


group2 = parser.add_mutually_exclusive_group(required=True)

group2.add_argument("--all"        , dest="keyword", action= "store_const",const="all"          ,help="Merge job summaries for interactions anywhere")
group2.add_argument("--vesselCase" , dest="keyword", action= "store_const",const="vesselCase"   ,help="Merge job summaries for interactions only in the SBT vessel")
group2.add_argument("--heliumCase" , dest="keyword", action= "store_const",const="heliumCase"   ,help="Merge job summaries for interactions only in the decay volume (He medium)")

parser.add_argument("--test", dest="testing_code", action="store_true", default=False, help="Process a small subset of files (quick check).")

options = parser.parse_args()

if (options.foldername=="muonDIS" or options.foldername=="neuDIS") and options.analysis_channel is None:
    parser.error(" Missing Analysis channel when using --muonDIS or --neuDIS.")
    exit(0)


main_path = options.main_path

keyword = options.keyword

main_path=options.main_path


if not options.analysis_channel:
    options.analysis_channel=''

foldername=f'{options.foldername}/{options.analysis_channel}'

if options.foldername=="muonDIS":
    pathlist = [
        f'{main_path}/{foldername}/SBT',
        f'{main_path}/{foldername}/Tr',
    ]
else:
    pathlist = [
                f'{main_path}/{foldername}/'
                ]

# Output directory
outdir = Path(f"plots_{foldername}")
outdir.mkdir(parents=True, exist_ok=True)

# ---------------- merge reader ----------------
def collect_groups(pathlist, keyword, limit_files=None):
    """
    Return dict:
      groups[treename] = {
          'pts': {'reco': {'zx':[Z,X], 'zy':[Z,Y], 'xy':[X,Y]},
                  'ip':   {'zx':[Z,X], 'zy':[Z,Y], 'xy':[X,Y]}},
          'has_ip': bool,
          'files': set([...])
      }
    """
    groups = {}
    files = []
    for base in pathlist:
        files.extend(glob.glob(f"{base}/job_*/selectionparameters_{keyword}.root"))
    files = sorted(files)
    if limit_files:
        files = files[:limit_files]

    if not files:
        print(f"(no selectionparameters_{keyword}.root found under {pathlist})")
        return groups

    for path in files:
        f = ROOT.TFile.Open(path, "READ")
        if not f or f.IsZombie():
            print(f"! skip unreadable: {path}")
            continue
        for key in f.GetListOfKeys():
            obj = key.ReadObj()
            if not obj.InheritsFrom("TTree"):
                continue
            name = obj.GetName()
            if not name.endswith("_pos"):
                continue   # only the position TTrees

            t = obj
            brs = [b.GetName() for b in t.GetListOfBranches()]
            has_ip = all(b in brs for b in ("IP_x","IP_y","IP_z"))

            if name not in groups:
                groups[name] = {
                    'pts': {
                        'reco': {'zx':[[],[]], 'zy':[[],[]], 'xy':[[],[]]},
                        'ip':   {'zx':[[],[]], 'zy':[[],[]], 'xy':[[],[]]},
                    },
                    'has_ip': has_ip,
                    'files': set(),
                }
            g = groups[name]
            g['has_ip'] = g['has_ip'] and has_ip
            g['files'].add(path)

            # Fill points
            for ev in t:
                x, y, z = float(ev.x), float(ev.y), float(ev.z)
                g['pts']['reco']['zx'][0].append(z); g['pts']['reco']['zx'][1].append(x)
                g['pts']['reco']['zy'][0].append(z); g['pts']['reco']['zy'][1].append(y)
                g['pts']['reco']['xy'][0].append(x); g['pts']['reco']['xy'][1].append(y)
                if has_ip:
                    IPx, IPy, IPz = float(ev.IP_x), float(ev.IP_y), float(ev.IP_z)
                    g['pts']['ip']['zx'][0].append(IPz); g['pts']['ip']['zx'][1].append(IPx)
                    g['pts']['ip']['zy'][0].append(IPz); g['pts']['ip']['zy'][1].append(IPy)
                    g['pts']['ip']['xy'][0].append(IPx); g['pts']['ip']['xy'][1].append(IPy)
        f.Close()
    return groups

# ---------------- plotting ----------------

def draw_one_merged(name, group, outdir: Path):
    
    try:
    
        load_style()
    
    except:
        pass

    pts = group['pts']

    fig = plt.figure(figsize=(15,7), constrained_layout=True)
    gs  = gridspec.GridSpec(2,8, figure=fig)

    ax_zx = fig.add_subplot(gs[0,0:5]); ax_zx.set_title('top view')
    ax_zx.set_ylabel('x (cm)'); ax_zx.set_xlim(-3000,3800); ax_zx.set_ylim(-600,600)

    ax_zy = fig.add_subplot(gs[1,0:5]); ax_zy.set_title('side view')
    ax_zy.set_xlabel('z (cm)'); ax_zy.set_ylabel('y (cm)')
    ax_zy.set_xlim(-3000,3800); ax_zy.set_ylim(-600,600)

    ax_xy = fig.add_subplot(gs[0:2,5:7]); ax_xy.set_title('back view')
    ax_xy.set_xlabel('x (cm)'); ax_xy.set_ylabel('y (cm)')
    ax_xy.set_xlim(-250,250);  ax_xy.set_ylim(-400,400)

    add_geometry(ax_zx, ax_zy, ax_xy, show_detectors=True)

    def scat(ax, X, Y, **kw):
        if X: ax.scatter(X, Y, **kw)

    # True IP (black star)
    scat(ax_zx, pts['ip']['zx'][0], pts['ip']['zx'][1], marker='*', s=100, color='black')
    scat(ax_zy, pts['ip']['zy'][0], pts['ip']['zy'][1], marker='*', s=100, color='black')
    scat(ax_xy, pts['ip']['xy'][0], pts['ip']['xy'][1], marker='*', s=100, color='black', label='DIS Interaction Point')

    # Reco vertex (red x)
    scat(ax_zx, pts['reco']['zx'][0], pts['reco']['zx'][1], color='red', marker='x', alpha=1, s=100)
    scat(ax_zy, pts['reco']['zy'][0], pts['reco']['zy'][1], color='red', marker='x', alpha=1, s=100)
    scat(ax_xy, pts['reco']['xy'][0], pts['reco']['xy'][1], color='red', marker='x', alpha=1, s=100, label='Recon. Candidate Vertex')

    # Legend (dedup)
    handles, labels = ax_xy.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(list(uniq.values()), list(uniq.keys()),
               loc='lower center', ncol=2, bbox_to_anchor=(0.45,-0.08),
               columnspacing=1.5, fontsize=14, frameon=True)

    # Save using ONLY the treename into plots_<sample>/
    safe = safe_filename(name)
    base = outdir / safe
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches='tight', pad_inches=0.4)
    #fig.savefig(str(base) + ".pdf",            bbox_inches='tight', pad_inches=0.4)
    plt.close(fig)
    print(f"saved: {base}.png  (from {len(group['files'])} files)")

# ---------------- run ----------------
limit = 5 if options.testing_code else None
groups = collect_groups(pathlist, keyword, limit_files=limit)

if not groups:
    print("(no matching *_pos trees found)")
    sys.exit(0)

# Plot per selection (tree name)
for name, grp in sorted(groups.items()):
    n_reco = len(grp['pts']['reco']['xy'][0])
    n_ip   = len(grp['pts']['ip']['xy'][0])
    if n_reco + n_ip == 0:
        continue
    draw_one_merged(name, grp, outdir)
