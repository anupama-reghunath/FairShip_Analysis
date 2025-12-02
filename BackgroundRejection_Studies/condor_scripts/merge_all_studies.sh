#!/bin/bash

EOSDIR=$1
SCRIPTDIR=$2
OUTPUTDIR=$3

#######################################################################################
source /cvmfs/ship.cern.ch/24.10/setUp.sh 
source /afs/cern.ch/user/a/anupamar/HTCondor/configfiles/config_ECN3_2024.sh #alienv load FairShip/latest-master-release > config_<version>.sh
echo 'config sourced'

#######################################################################################

echo "Merging muonDIS studies now.."

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --partialreco --all > "$OUTPUTDIR/muonDIS_partialreco_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --partialreco --vesselCase > "$OUTPUTDIR/muonDIS_partialreco_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --partialreco --heliumCase > "$OUTPUTDIR/muonDIS_partialreco_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --partialreco --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --partialreco --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --partialreco --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/muonDIS/partial" "muonDIS_partialreco" 

#--------------------------------------------------------------------------------------

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --fullreco --all > "$OUTPUTDIR/muonDIS_fullreco_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --fullreco --vesselCase > "$OUTPUTDIR/muonDIS_fullreco_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --fullreco --heliumCase > "$OUTPUTDIR/muonDIS_fullreco_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --fullreco --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --fullreco --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --fullreco --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/muonDIS_fullreco" "muonDIS_fullreco"

#--------------------------------------------------------------------------------------

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --leptonrho --all > "$OUTPUTDIR/muonDIS_leptonrho_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --leptonrho --vesselCase > "$OUTPUTDIR/muonDIS_leptonrho_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --muonDIS --leptonrho --heliumCase > "$OUTPUTDIR/muonDIS_leptonrho_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --leptonrho --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --leptonrho --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --muonDIS --leptonrho --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/muonDIS_leptonrho" "muonDIS_leptonrho" 

echo "completed."

#--------------------------------------------------------------------------------------
#exit 0
#--------------------------------------------------------------------------------------

echo "Merging neuDIS now .."

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --partialreco --all > "$OUTPUTDIR/neuDIS_partialreco_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --partialreco --vesselCase > "$OUTPUTDIR/neuDIS_partialreco_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --partialreco --heliumCase > "$OUTPUTDIR/neuDIS_partialreco_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --partialreco --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --partialreco --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --partialreco --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/neuDIS" "neuDIS_partialreco"

#--------------------------------------------------------------------------------------

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --fullreco --all > "$OUTPUTDIR/neuDIS_fullreco_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --fullreco --vesselCase > "$OUTPUTDIR/neuDIS_fullreco_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --fullreco --heliumCase > "$OUTPUTDIR/neuDIS_fullreco_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --fullreco --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --fullreco --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --fullreco --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/neuDIS_fullreco" "neuDIS_fullreco"

#--------------------------------------------------------------------------------------

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --leptonrho --all > "$OUTPUTDIR/neuDIS_leptonrho_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --leptonrho --vesselCase > "$OUTPUTDIR/neuDIS_leptonrho_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --neuDIS --leptonrho --heliumCase > "$OUTPUTDIR/neuDIS_leptonrho_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --leptonrho --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --leptonrho --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --neuDIS --leptonrho --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/neuDIS_leptonrho" "neuDIS_leptonrho" 

echo "completed."

#--------------------------------------------------------------------------------------
#exit 0 #REMOVE THIS IF SIGNAL STUDIES ARE ALSO TO BE MERGED
#--------------------------------------------------------------------------------------

echo "Merging Signal Studies now.."

echo "Merging mupi.."

python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --mupi --all > "$OUTPUTDIR/mupi_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --mupi --vesselCase > "$OUTPUTDIR/mupi_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --mupi --heliumCase > "$OUTPUTDIR/mupi_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --mupi --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --mupi --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --mupi --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/signalEventCalc/mupi" "mupi_EventCalc"

echo "completed."

echo "Merging mumuv now.."
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --mumuv --all > "$OUTPUTDIR/mumuv_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --mumuv --vesselCase > "$OUTPUTDIR/mumuv_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --mumuv --heliumCase > "$OUTPUTDIR/mumuv_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --mumuv --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --mumuv --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --mumuv --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/signalEventCalc/mumuv" "mumuv_EventCalc"

echo "completed."

echo "Merging erho now.."
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --erho --all > "$OUTPUTDIR/erho_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --erho --vesselCase > "$OUTPUTDIR/erho_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --erho --heliumCase > "$OUTPUTDIR/erho_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --erho --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --erho --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --erho --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/signalEventCalc/erho" "erho_EventCalc"

echo "completed."


echo "Merging murho now.."
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --murho --all > "$OUTPUTDIR/murho_all.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --murho --vesselCase > "$OUTPUTDIR/murho_vesselCase.txt"
python "$SCRIPTDIR/BackgroundRejection_Studies/merge.py" --murho --heliumCase > "$OUTPUTDIR/murho_heliumCase.txt"

#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --murho --all
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --murho --vesselCase
#python "$SCRIPTDIR/BackgroundRejection_Studies/surviving_xyzplots.py" --murho --heliumCase

#condor_scripts/condor_hadd.sh "$EOSDIR/signalEventCalc/murho" "murho_EventCalc"

echo "completed."

#------------------------------------------------------------------------------------------------------------

echo "All jobs completed"