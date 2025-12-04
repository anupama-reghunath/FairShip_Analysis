#!/bin/bash
#######################################################################################

ClusterId=$1
ProcId=$2
Lepton=$3
Mass=$4
PIDflag=$5     # --noPID or --wPID
UBTflag=$6     # --noUBT or --wUBT
SBTflag=$7     # --noSBT or --wSBT

#######################################################################################

source /cvmfs/ship.cern.ch/24.10/setUp.sh 
source /afs/cern.ch/user/a/anupamar/HTCondor/configfiles/config_ECN3_2024.sh #alienv load FairShip/latest-master-release > config_<version>.sh

echo 'config sourced'

# Map placeholder -> actual argument

if [ "$PIDflag" = "--noPID" ]; then
    PIDflag=""
fi

if [ "$UBTflag" = "--noUBT" ]; then
    UBTflag=""
fi

if [ "$SBTflag" = "--noSBT" ]; then
    SBTflag=""
fi
#######################################################################################

python /afs/cern.ch/user/a/anupamar/alt_v2/FairShip_Analysis/HNL_leptonrho_massrecon/HNLinvmass_wBG.py --lepton $Lepton --mass $Mass $PIDflag $UBTflag $SBTflag

