echo "In this script we will use one of the provided CWE-190 models to search for integer overflopws in Avian."

while true; do
    read -p "Do you wish to run the model precition demo? We will execute a CWE-190 model on Avian (you must run the prerpocessing demo first) and then execute the mdeol directly on Avian's source code using lint. This may take time since this script uses cpu not gpu" yn
    case $yn in
        [Yy]* ) 
        
		
cd ~

echo ""
echo "Let's get the predictions for all potential manifestation points for each CWE."
echo ""
echo "First, let's use the feature_stats tool which takes in a preprocessed graph dataset. Since our Avian sample is not a full dataset, we will perform exec-only since there is not need for auc or roc plots. Remoce --device cpu if you have a cuda device avalaible."

hector stats --device cpu --predictions-csv avian-v1.2.0/avian-190-preds.csv --exec-only models/trained_on_aug_and_cve/CWE190/run6_doblog_275ep/model avian-v1.2.0/hector_build/hector-190-preproc.json 

echo ""
echo "Done"


* ) echo "Please answer yes or no.";;
    esac
done

