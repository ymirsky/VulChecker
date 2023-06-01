echo "The following script is an example of how to (1) convert source code into an annotated PGD graph and then (2) preprocess it into a set of subgraphs, one for each manifiestation point. The final result is a single json file containing the subgraphs for the project. In this demo we will process the Avian source code only."

while true; do
    read -p "Do you wish to run the preprocessing demo?" yn
    case $yn in
        [Yy]* ) 
        
		
cd ~

echo ""
echo "First let's get a project with a known CVE, we will use avian. You can find more samples in our full dataset"
unzip VulChecker/dataset/avian.zip


cd avian-v1.2.0/

# Now we will use the hector tool to run LLVM to create the ePDG graph with labels in json format for each CWE
# Note: Here we chose binary_to_object to be the target, but there are 3 possible targets in Avian. To see all targets to choose from, run the same command but with empty quotes "" as the target instead.
hector configure --llap-lib-dir ~/llvm-project/llvm-build/lib --labels labels.json cmake src/tools/type-generator/type_generator 121 190 415 416

# use ninja to generate the json files from the build
cd hector_build/
ninja -f hector.ninja

echo ""
echo "There should now be 4 json files in the build directory; one for each category of CWE"
ls *.json -l

echo ""
echo "Use the hector tool to convert the ePDG json file into a set of ePDG subgraphs, one for each potential manifestation point. The final result will be a single json file that contains one subgraph per line. Here we must repeat the process for each CWE. During this process, we will incrementally update the traning indexes for this CWE that will be needed later."

## CWE190
hector preprocess \
    --training-indexes ~/indexes-190.json \
    --source-dir $PWD \
    --cwe 190 \
    --output hector-190-preproc.json \
    hector-190.json

## CWE191  (note that the 190 input works for 191)
hector preprocess \
    --training-indexes ~/indexes-191.json \
    --source-dir $PWD \
    --cwe 191 \
    --output hector-191-preproc.json \
    hector-190.json

## CWE121
hector preprocess \
    --training-indexes ~/indexes-121.json \
    --source-dir $PWD \
    --cwe 121 \
    --output hector-121-preproc.json \
    hector-121.json

## CWE122 (note that the 121 input works for 122)
hector preprocess \
    --training-indexes ~/indexes-122.json \
    --source-dir $PWD \
    --cwe 122 \
    --output hector-122-preproc.json \
    hector-121.json
    
## CWE415
hector preprocess \
    --training-indexes ~/indexes-415.json \
    --source-dir $PWD \
    --cwe 415 \
    --output hector-415-preproc.json \
    hector-415.json

## CWE416 (note that the 415 input works for 416)
hector preprocess \
    --training-indexes ~/indexes-416.json \
    --source-dir $PWD \
    --cwe 416 \
    --output hector-416-preproc.json \
    hector-415.json

echo ""
echo "There should now be 6 preprocessed json files in the build directory; one for each category of CWE"
ls *preproc.json -l

echo ""
echo "There should also be one training index for each CWE in the home dir. Note that you should reference these indexes when preprocesing more samples for the same traning set."
ls ~/*indexes*.json -l


echo ""
echo "All done!"
echo "Note: if you are making a training dataset, you would have to"
echo "(1) perform this process on many more projects, each time using the same index file"
echo "(2) concatenate the json files together into a single json file (per CWE)"
echo "      you can optionally augment the dataset using synthetic sampples using hector augmentation"
echo "(3) run hector validate_data on the result"
echo "(4) downsample the negatives with hector sample_data to avoid memory issues during training"
echo "(5) extract the normalization parameters for training from the dataset using hector feature_stats"

break;;
[Nn]* ) exit;;
* ) echo "Please answer yes or no.";;
    esac
done

