#! /bin/bash

# Makes the augmented data for CWE415.
# Use this script as a reference for your own scripts
 
. /mnt/sdc/hector-pipeline/hector-virtualenv/bin/activate

# Declare list of CWEs
CWE=415
cd /mnt/sdd/datasets/All_CWEs/CWE${CWE}/graphs/
mkdir -p wild_aug/
echo "#### Augmenting Graphs ####"
FILES="wild/unlabeled/*.json"
seed=1
for p in $FILES
do
  f="${p##*/}"
  echo "Augmenting $f"
  eval "hector augmentation --margin 30 --inject-positive --max-positive-injections 1000 --seed $seed --min-path-length 3 --max-path-length 30 -o wild_aug/${f}.gz juliet/CWE${CWE}-juliet.json wild/unlabeled/$f &"
  seed=`expr $seed + 1`
done
wait
    
# Process the graphs
echo "#### Preprocessing Graphs ####"
cd /mnt/sdd/datasets/All_CWEs/CWE${CWE}/
mkdir -p proc_graphs/wild_real-labels/combined/
mkdir -p proc_graphs/wild_real-labels/individual/
mkdir -p proc_graphs/wild_augmented-labels/combined/
mkdir -p proc_graphs/wild_augmented-labels/individual/
    
FILES="graphs/wild/labeled/*.json"
for p in $FILES
do
  f="${p##*/}"
  echo "Processing $f"
  hector preprocess --training-indexes indexes-${CWE}.json --source-dir $PWD --cwe ${CWE} --output proc_graphs/wild_real-labels/individual/${f}.gz graphs/wild/labeled/$f
done

FILES="graphs/wild_aug/*.json.gz"
for p in $FILES
do
  f="${p##*/}"
  echo "Processing $f"
  hector preprocess --training-indexes indexes-${CWE}.json --source-dir $PWD --cwe ${CWE} --output proc_graphs/wild_augmented-labels/individual/$f graphs/wild_aug/$f
done
    
# Gather preprocessed graphs
echo "#### Gathering the Preprocessed Graphs (wild) ####"
cat proc_graphs/wild_real-labels/individual/*.json.gz > proc_graphs/wild_real-labels/combined/CWE${CWE}_wild_real-labels.json.gz
cat proc_graphs/wild_augmented-labels/individual/*.json.gz > proc_graphs/wild_augmented-labels/combined/CWE${CWE}_wild_augmented-labels.json.gz

# Validate combined datas
echo "#### Validating Data ####"
hector validate_data \
--output proc_graphs/wild_real-labels/combined/CWE${CWE}_wild_real-labels_clean.json.gz \
proc_graphs/wild_real-labels/combined/CWE${CWE}_wild_real-labels.json.gz

hector validate_data \
--output proc_graphs/wild_augmented-labels/combined/CWE${CWE}_wild_augmented-labels_clean.json.gz \
proc_graphs/wild_augmented-labels/combined/CWE${CWE}_wild_augmented-labels.json.gz
    
echo "#### Downsampling the Data ####"
hector sample_data --negative 1.0 \
proc_graphs/wild_real-labels/combined/CWE${CWE}_wild_real-labels_clean.json.gz \
proc_graphs/wild_real-labels/combined/CWE${CWE}_wild_real-labels_clean_1.0.json.gz
    
hector sample_data --negative 0.03  \
proc_graphs/wild_augmented-labels/combined/CWE${CWE}_wild_augmented-labels_clean.json.gz \
proc_graphs/wild_augmented-labels/combined/CWE${CWE}_wild_augmented-labels_clean_0.03.json.gz

# Generate normalization params
hector feature_stats --indexes indexes-${CWE}.json --depth-limit 40 proc_graphs/wild_augmented-labels/combined/CWE${CWE}_wild_augmented-labels_clean_0.03.json.gz


echo "DONE"


