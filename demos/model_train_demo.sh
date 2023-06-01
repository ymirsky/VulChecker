echo "In this script we will train a CWE190 model using the dataset from the paper. The dataset uses 'clean' project from the wild with augmented labels from Juliet. The negative manifestation points in this dataset have been downsampled by 90% (memory limitations)."

while true; do
    read -p "Do you wish to run the model training demo?" yn
    case $yn in
        [Yy]* ) 
        
		
cd ~

echo ""
echo "First we need to extract the normalization params for the model"

hector feature_stats --indexes ~/dataset/CWE190/indexes-190.json --depth-limit 40 ~/dataset/CWE190/CWE190_augLabels_clean_0.1_1.0.json.gz 

echo ""
echo "Now we will train a model on the complete dataset."
echo "WARNING: this VM does not have cuda installed. Therefore, traning will take place on the CPU. Please use this as instructional code and not practical code. For actual training, change the device argument to the cuda ID (e.g., cuda:0) or simply remove it."
echo "Note: if you get an 'abort' or 'killed' then the VM has run our of memory"

hector train 190 model-CWE190/ ~/dataset/CWE190/CWE190_augLabels_clean_0.1_1.0.json.gz ~/dataset/CWE190/CWE190_realLabels_clean_0.005_1.0.json.gz --indexes ~/dataset/CWE190/indexes-190.json --embedding-dimensions 32 --learning-rate 0.0001 --keep-best --feature-stats ~/feature_stats.npz --patience 5 --epochs 200 --embedding-steps 50 --embedding-reduction mean --recursive-depth 9 --classifier-dimensions 32 --classifier-depth 7 --device cpu



break;;
[Nn]* ) exit;;
* ) echo "Please answer yes or no.";;
    esac
done

