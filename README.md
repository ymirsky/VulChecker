# Overview

In this repository you will find a Python implementation of VulChecker; a tool for detecting vulnerabilties (CWE) in source code. From,

*Mirsky Y, Macon G, Brown M, Yagemann C, Pruett M, Downing E, Mertoguno S, Lee W. "VulChecker: Graph-based Vulnerability Localization in Source Code", USENIX Security 23*

If you use any derivative of this code in your work, please cite our publicaiton.

This implimentation supports cmake C/C++ projects only. It can be used to detect integer overflow (CWE-190), stack overflow (CWE-121), heap overflow (CWE-122), double free (CWE-415), and use-after-free (CWE-416) vulnerabilites.

# What is VulChecker?

VulChecker is a tool that can precisely locate vulnerabilities in source code (down to the exact instruction) as well as classify their type (CWE). This is useful for developers to locate potential security risks in their code *during* development, even before the project is complete and deployed. The tool converts cmake C/C++ projects into a graph-based program representation called and ePDG. For each potential manifestation point in the project, a subgraph is extracted by crawling the ePDG up from the potential manifestation point. Finally, a graph-based neural network called Structure2Vec is used to classify which subgraphs yeild actual vaulnerabilites. This is repeated for each CWE resulting in seperate a classifiers.  The figure below illustreates how Vulchecker works for a single CWE:

![image](https://github.com/ymirsky/VulChecker/assets/11553515/794361bf-f336-4d9e-b48a-28a1589c85bb)

The tool also provides a means for data augmetation: Although many labeld samples are required to train a robust model, it is hard to aquire many line-level labeled samples of vulnerabilites from the wild. Therefore, the tool lets you augment the ePDGs of "clean" projects from the wild with the ePDGs of synthetic vulnerbility datasets. In our research, we found that this is enough to train a model to detect vulnerabilites in the wild. However, whenever possible, it is reccomeneded to include real vulnerabilites from the wild in the training data as well.

## Contents
In this README you will find chapters on the following topics:
1. Installation instructions
2. Detailed usage instructions
3. Assets: How to access the assets (datasets, models, VM)
4. Developer Notes
5. Acknowledgements

# Installation

This tool uses a pipeline of many different components to go from a C/C++ project all the way to a predction from a deep learning model. For example, LLVM with a custom plugin is used to create the ePDGs with any provided labels. Setting up this pipeline is complex and takes a lot of time since LLVM must be compiled. Therefore, instead of performing a clean install (using the instructon below) we provide an Ubuntu VM with VulChecker preinstalled. On the VM's desktop you will find some demo scripts.

The VM can be downloaded from [here](https://bgu365-my.sharepoint.com/:f:/g/personal/yisroel_bgu_ac_il/Et1JFmXJFFREmBQewk-5GhIBxgFDxNihBYvTR7ZAOsC_Zg?e=btAJzX). 
Username: vulchecker, Password: vulchecker



## Clean Install

The following are instructions for a clean install on Linux (tested on `Ubuntu 20.04` and `python 3.8.10`)

### Quick Start
You can use the install script in this repository (`demos/`) as a guide. However, we reccomend that you read below for better instruction.

### Components

VulChecker uses a number of components that must be installed. Here is a list of components of Vulchecker which we maintain in seperate repositories:

- `VulChecker`: the core library for processing data and training models. All operations with this library are through a command line tool called `hector`. https://github.com/ymirsky/VulChecker.git
- `LLAP`: a plugin to LLVM for extracting ePDGs from cmake C\C++ projects. https://github.com/michaelbrownuc/llap
- `Structure2Vec`: our pyTorch implimentation of the graph-based neural network by Dai et al. https://github.com/gtri/structure2vec
- `vulchecker-misc`: a collection of helpful (optional) scripts, such as automatic labeling Juliet samples. https://github.com/michaelbrownuc/vulchecker-misc


### Step 1: Install the Python Libraries

It is reccomended that you create and activate a python environment before installing any of the libraries to avoid conflicts.

First get VulChecker (`hector`) and Structure2Vec:
```bash
git clone https://github.com/ymirsky/VulChecker.git
git clone https://github.com/gtri/structure2vec.git
```

Install them and Cython for optimized graph manipulation:
```bash
# install them
python3 -m pip install -U pip setuptools wheel
python3 -m pip install cython cmake
python3 -m pip install ./structure2vec 
python3 -m pip --no-cache-dir install ./VulChecker
```

Check that VulChecker installed correctly by accessing the help option of the `hector` tool
```bash
~$ hector --help

Usage: hector [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  augmentation       Augment a real-world program with Juliet...
  compile_for_train
  configure          Configure a codebase to be analyzed by HECTOR.
  cross_validation
  feature_stats
  hyperopt           Optimize hyperparameters.
  lint               Lint-check a codebase using HECTOR.
  predict
  preprocess         Preprocess Program Dependence Graphs.
  sample_data        Downsample manifestation points.
  stats
  train
  train_test_split
  validate_data
  visualize
```

Note :memo:: be sure to use `--help` on the commands to get further otpions and hints. E.g., `hector preprocess --help`

### Step 2: Get LLVM and ninja

Next we need to obtain v10.0.0 of the LLVM compiler and ninja to work with the source code.

Install ninja:
```bash
sudo apt-get install -y ninja-build
```

Download LLVM:
```bash
cd ~
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/llvm-project-10.0.0.tar.xz
tar xvf llvm-project-10.0.0.tar.xz
mv llvm-project-10.0.0 llvm-project
```

### Step 3: Install LLVM with LLAP

Now we need to install the VulChecker plugin to the LLVM compiler (LLAP) which enables us to generate ePDGs from source code.

Download LLAP:
```bash
git clone https://github.com/michaelbrownuc/llap.git
```

Add the LLAP plugin to LLVM:
```bash
cp -R llap-master/src/* llvm-project/llvm/lib/Transforms/
```

Compile LLVM *...and go get some coffee* :coffee:. *It will take a while*):
```bash
cd llvm-project/
cmake -S ./llvm/ -B llvm-build -DCMAKE_BUILD_TYPE=Release
make -C llvm-build -j 16
make -C llvm-build install 
cmake -S ./clang/ -B clang-build -DCMAKE_BUILD_TYPE=Release
make -C clang-build -j 16
make -C clang-build install
```

Important ⚠️: When using the `hector` tool, you will be asked to provide the path to LLAP to execute certain commands. If you installed LLVM in the home dir (as above) then the path to LLAP is:
```
~/llvm-project/llvm-build/lib
```


# Usage

VulChecker follows a pipeline approach consisting of three segments:

1. Data Preparation
2. Model Training
3. Execution

`Data Preperation` involves (1) prepairing line-level labels for a C/C++ cmake project [optional], (2) converting the project into an ePDG using LLVM, (3) processing the ePDG by converting it into a collection of potential manifestaion point subgraphs, and (4) collecting the processed projects into singular dataset files [optional]. 

`Model Training` involves (1) extracting normalization parameters from the training dataset, (2) training a *Structure2Vec* model on the dataset, and (3) evaluating the model on a test set.

`Execution` involves (1) executing a trained model on a project and (2) aquiring the results. The project must be preprocessed first similar to the steps in `Data Preparation`.

You can execute each of these steps using our command line tool called `hector`.

Important ⚠️: When executing each part of the pipeline, you **must indicate which CWE** the final model will be detecting. This is because the features and manifestation points are different for each CWE. This means, if you want to use C++ project for all six CWEs then you will need to make six seperate ePDGs of the project, etc.

## Running the Pipeline
Below is a detailed illustration of how the pipeline is used for a single CWE 'X':

![image](https://github.com/ymirsky/VulChecker/assets/11553515/38b9fa89-d7d3-4bf7-8cfe-5cd7f98ab5c3)

In this dataflow diagram, we show how to (1) setup a training dataset that uses labels from a sythetic vulnerability dataset (e.g., Juliet), (2) evaluate the model on a labeled CVE dataset, and (3) execute other projects from the wild on the same model. Note, although not required, a good model will also use samples from the wild labeled vulnerabilites (and not just sythnthetic vulenrabilites).

As examples of how to execute parts of this pipeline, you can take a look at the demo scripts in this repo (`demos/`) which show you how to process data, train models and make predicitons. The demos are written to run on the provided VM.

We will now explain in detail how to perform each of these steps

### (1) Collect Source Code
The first thing you need to do is collect C/C++ cmake projects for training and testing the model. You may already have a model (e.g., the ones we provide) and want execute them on new projects as well. The source code to each project should be in a seperate directory (e.g., `cmake_proj/`). 

### (2) Label the Projects
The projects you will be using for training and testing the model will have some labels. To label project `cmake_proj/`, you will need to make a file that indicates where the vulnerabilites manifest themselves in the source code. The labels file is a JSON array of objects (e.g., `cmake_proj/labels.json`).
Each object has three keys, `filename`, `line_number`, and `label`.
The recognized labels are:

CWE | Root Cause Label | Manifestation Label
--- | ---------------- | -------------------
121 (Stack Overflow) | `declared_buffer` | `stack_overflow`
122 (Heap Overflow) | `declared_buffer` | `heap_overflow`
190 (Integer Overflow | `overflowed_variable` | `overflowed_call`
191 (Integer Underflow | `underflowed_variable` | `underflowed_call`
415 (Double Free) | `first_free` | `second_free`
416 (Use After Free) | `freed_variable` | `use_after_free`

For example,
the labels file might contain:

```json
[
    {"filename": "src/foo.c", "line_number": 27, "label": "declared_buffer"},
    {"filename": "src/foo.c", "line_number": 37, "label": "stack_overflow"}
]
```

### (3) Convert each Project into an ePDG (configure step)
Next, we pass each project through LLAP to generate its ePDG file. A project does not need a label file to be processed. 

To generate the ePDG for project `cmake_proj/*`, use the `configure` option of the `hector` tool. For example:
```bash
cd ~/cmake_proj
hector configure --llap-lib-dir ~/llvm-project/llvm-build/lib --labels labels.json cmake <path-to-target-in-project> 121 190 415 416
cd hector_build
ninja -f hector.ninja
```

This will produce four files named `hector-{121,190,415,416}.json`. The file `hector-190.json` works for both CWE-190 and CWE-191. Similarly, the file `hector-121.json` works for both CWE-121 and CWE-122.

Important :warn:: At this point, the workflow splits according to the CWE you are working on. 
This means that 
- you cannot mix different CWEs in the same dataset (the end model must receive files processed for one CWE)
- you must indicate which CWE you are processing in future steps (see later on)

Note :memo:: If you don't know what the target is for your project, then simply run the command with "" as the target. The command will return with a list of options you can use as targets.

Note :memo:: The `hector configure` command can be run on many projects in parallel to save time (if you have enough RAM)

Note :memo:: not all projects used for training need to have labels. For example, in the figure above we create a training set by augmenting clean labeless projects from the wild with the Juliet projects which have labels.

Note :memo:: The entire pipeline supports json files compressed using gzip (`*.json.gz`). At this point, you can compress your json files to save a significant amount of disk space. All other steps in the pipeline will accept `*.json.gz` files and output the same format respectivly.


### (4) Optional - Create augmented ePDGs
If you want to make a robust model but only have a few real projects with vulnerabilities, or none at all, then you can perform augmentation. Augmentaton takes the ePDG a real project (assumedly clean of vulnerabilites) and injects labeled vulnerabiles from differnt ePDGs. In our research, we found that the source for can be a sythetic dataset such as [Juliet](https://samate.nist.gov/SARD/test-suites/112). Doing so expands your training data and helps the model better idenitfy vulnerabilites in the wild. The figure below illustrates how an ePDG is augmented with one example of a labeled vulnerable ePDG.

![image](https://github.com/ymirsky/VulChecker/assets/11553515/af867247-48ff-4b04-adfd-07803e2ba97a)

To augment a single project you will need to collect all of the ePDGs of the labled projects into a single file, one per line.
For example, for CWE-121, we can collect samples from the provided Juliet dataset into a single file:

```bash
find CWE121/labeled_graphs -name '*.json' | xargs cat > juliet-121-pdgs.nljson
```

Then, you can augment a project using the `hector augmentation` command as follows:
```bash
~$ hector augmentation --help
Usage: hector augmentation [OPTIONS] JULIET REAL_WORLD

  Augment a real-world program with Juliet vulnerabilities.

  For each PDG in REAL_WORLD, random control flow paths between --min-path-
  length and --max-path-length are chosen and a vulnerable path from JULIET
  is inserted into the control flow split into two parts at the beginning
  and end of the chosen path. This continues until there are no more JULIET
  examples or until there are no more suitable paths.

  You must specify at least one of --inject-positive or --inject-negative
  either directly or implicitly via --max-{positive,negative}-injections.

  Positive and negative examples are injected with equal probability until
  one set is exhausted (or the max for that type is reached). After that,
  the other type is injected unconditionally until it is exhausted (or its
  max is reached).

Options:
  --seed INTEGER                  Random seed for reproducibility.
  -o, --output FILE               Location where selected paths will be
                                  written.

  --min-path-length INTEGER RANGE
                                  Minimum path length to augment
  --max-path-length INTEGER RANGE
                                  Maximum path length to augment
  --margin INTEGER                Minimum graph distance between inserted
                                  paths

  --max-positive-injections INTEGER
                                  Maximum number of vulnerable examples to
                                  insert into a single graph (implies
                                  --inject-positive).

  --max-negative-injections INTEGER
                                  Maximum number of not-vulnerable examples to
                                  insert into a single graph (implies
                                  --inject-negative).

  --inject-positive               Inject vulnerable examples.
  --inject-negative               Inject not-vulnerable examples.
  --help                          Show this message and exit.
```
For example, the following would augment an ePDG of `cleanProj/*` with samples from Juliet in `~/juliet-121.nljson`

```bash
hector augmentation \
    --margin 30 --inject-positive --max-positive-injections 1000 \
    --seed 5 --min-path-length 3 --max-path-length 30 \
    --output hector-121-augmented.json \
    ~/juliet-121.nljson \
    ~/cleanProj/hector_build/hector-121.json
```

Note :memo:: Augmentation using synthetic labels alone can make an effective model. However, it is always prefferred to add as many real examples of vulnerabiites from the wild as possible.


### (5) Extract sub-ePDGS (potential manifestation points)
Now we have a collection of ePDGs one for each project. At this stage we need to extract sub-graphs from each ePDG where each sub-graph captures a potential manifestation point. To do this, we can run the `hector preprocess` command on each ePDG. For example, the following command takes in the json of the ePDG for `cmake_proj` and outputs a json containing all of its sub-ePDGs. 

```bash
 hector preprocess \
    --training-indexes indexes-121.json --source-dir ~/ --cwe 121 \
    --output ~/proc_graphs/CWE121/cmake_proj.json \
    ~/cmake_proj/hector_build/hector-121.json
```

While processing an ePDG, the tool dynamically builds an index of all the functions and operations found in the code. This index is needed to determine the values of the nominal features for training. The `--training-indexes` argument is used to indicate where this file should be saved. If the file already exists then it will be updated. When preprocessing many projects under the same CWE, the same index should be passed to each subsequent call.

Warning ⚠️: You cannot run this command in parallel on multiple ePDGs for the same CWE. This is because each subsequent call to `hector preprocess` updates the current index.

Tip 📝: give the filename for the argument `--output` the extention `json.gz` to have the tool compress the outout for you

### (6) Create Datasets (gather subgraph collections)

At this point you will have multiple sub-graph files, one for each project. To creat a dataset for training or testing, simply concatenate the files you want in each dataset. For example, you can execute
```bash
cat ~/proc_graphs/CWE121/*.json >   \
    ~/proc_graphs/CWE121/combined/dataset.json
```

Note :memo:: As noted earlier, you can pass `hector` json files in gzip format (`*.json.gz`). A useful tip is that gz files can be directly concatenated (e.g., run `cat *json.gz > dataset.json.gz`)


### (7) Optional - Validate the Datasets

Sometimes there are issues in generating the ePDGs. For example, labels are not assigned due mismapping and potential bugs. The `hector validate` command check whether a sub-ePDG file (or dataset) is correct and ready for use. The tool  will warn you about any graphs with issues that will cause problems further along. The most common issue is that a program might not have any labeled nodes
even if there were labeled lines of source.
There are several reasons this may happen,
but they have to be investigated one at a time.

After validation, the tool outputs the fixed version. Below is an example of how to validate a dataset.

```bash
hector validate_data \
    --check-labels \
    --output ~/proc_graphs/CWE121/combined/dataset_clean.json \
    ~/proc_graphs/CWE121/combined/dataset.json
```
The argument `check-labels` can be omitted if the dataset under validation intentionally has no labels (e.g., a project being analyzed during production)

### (8) Optional - Downsampling
In some cases there may be too many potential manifestation points for the model to handle. In order to avoid severe class imbalance and to make training more efficient, you can down sample the number of negative cases in the data. In this example, we are only retaining 10% of the negatives
```bash
hector sample_data --negative 0.1 \
~/proc_graphs/CWE121/combined/dataset_clean.json \
~/proc_graphs/CWE121/combined/dataset_clean_0.1.json
```
You can also add an argument to downsample the positive cases if needed.

### (9) Get Normalization Parameters
Before training a model, we need to extract some statistics which will help us normalize the data before training. This meta data will be stored in the final model after training.
To extract these parameters use the `hector feature_stats` tool. For example

```bash
hector feature_stats --indexes ~\indexes-121.json --depth-limit 40 ~/proc_graphs/CWE121/combined/dataset_clean_0.1.json
```
This will result in a file called `feature_stats.npz` which will be written to the local directory.

### (10) Train a Model
Now we are finally ready to train a structure2vec model for each CWE dataset that has been prepaired. 
There are many options you can supply `hector train` to controlling the model's hyperparameters. You can use `hector hyperopt` to help you find the best set of parameters.
By running `hector train --help` we can see the available options. 
```bash
$ hector train --help
Usage: hector train [OPTIONS] CWE OUTPUT_DIR TRAINING_GRAPHS TESTING_GRAPHS

Options:
  --device DEVICE                 Device on which to run.
  --indexes FILE                  File where feature dictionaries are stored.
                                  [default: indexes.json]

  --feature-stats FILE            File where feature statistics are stored.
                                  [default: feature_stats.npz]

  --embedding-dimensions INTEGER RANGE
                                  Dimensionality of graph embedding.
                                  [default: 16]

  --embedding-steps INTEGER RANGE
                                  Iterations of embedding algorithm.
                                  [default: 4]

  --embedding-reduction [sum|mean|first]
                                  Reduction method to use at end of embedding.
                                  [default: first]

  --recursive-depth INTEGER RANGE
                                  Depth of embedding DNN.  [default: 2]
  --classifier-dimensions INTEGER RANGE
                                  Dimensionality of classifier DNN.  [default:
                                  16]

  --classifier-depth INTEGER RANGE
                                  Depth of classifier DNN.  [default: 2]
  --batch-size INT                Training batch size  [default: 50]
  --epochs INT                    Training epochs  [default: 50]
  --patience INT                  Earlystopping Patience  [default: 10]
  --learning-rate FLOAT           Learning rate for Adam optimizer.  [default:
                                  0.001]

  --betas <FLOAT FLOAT>...        Gradient running average decays for Adam
                                  optimizer.  [default: 0.9, 0.999]

  --fine-tune                     Fine-tune an existing model.
  --existing DIRECTORY            Model path to load (default: same as
                                  output).

  --keep-best                     Keep the best model instead of the last one.
  --eager-dataset / --lazy-dataset
                                  Load entire dataset into memory in advance.
                                  [default: True]
```
The basic command has the following form (for CWE-121):
```bash
hector train \
    --indexes ~/indexes-121.json \
    -- ~/feature_stats.npz
    121 \
    ~/models/CWE121 \
    ~/proc_graphs/CWE121/combined/trainset_clean_0.1.json \
    ~/proc_graphs/CWE121/combined/testset_clean_0.1.json 
```

Once training is complete, the model files will be written to the indicated directory. Note :warn:, the directory must be initially empty or non-existant otherwise the code will halt.

The model serialization format consists of two files:
- a PyTorch weights checkpoint
- a metadata file with additional information needed to make predictions

Note :memo:: The parameters we used in our paper can be found in a txt file next to the provided models (see assets below).
Note :memo:: Early stopping is not implimented in this version
Note :memo:: If you want to train without a testset, simply use the train set as the test set as well

### (11) Evaluate a Model
After training, you can extract statistics, roc plots, and raw predictions in bulk on your test set or other datasets (labeled or unlabled). To do this, use the `hector stats` command as follows:
```bash
~$ hector stats --help
Usage: hector stats [OPTIONS] OUTPUT_DIR TESTING_GRAPHS

Options:
  --device DEVICE         Device on which to run.
  --batch-size INT        Training batch size  [default: 50]
  --predictions-csv FILE  File where CSV prediction information will be
                          written.

  --dump FILE             File where outputs will be written.
  --source-dir DIRECTORY  Directory containing original source files.
  --roc-file FILE         File where ROC plot will be saved.
  --exec-only             For making predicitons on data with no labels.
  --help                  Show this message and exit.
```
For example
```bash
hector stats \
    --dump CWE121-testset.npz \
    --roc-file CWE121-testset_roc.png \
    --predictions-csv CWE121-testset.csv \
    ~/models/CWE121  \
    ~/proc_graphs/CWE121/combined/testset_clean_0.1.json 
```

Note :memo:: The csv file contains the predicted scores for every potential manifestation point in unsorted order. There may be duplicate rows since there can be multiple potential manifestaion points (instructions) in a line of source code. Therefore, it is reccommended to perform duplicate elimination (while retaining the highest score from of each set of duplicates).

### (12) Use a Model
If you have a trained model in production, you will want to execute it on projects without going through this massive pipeline. All you need to do is use `hector lint` on the cmake project with the target model. For example, if you want to check for CWEs 121, 122 and 416 then run `hector lint` three times; once for each CWE model. The command is used as follows:
```bash
~$ hector lint --help
Usage: hector lint [OPTIONS] [SOURCE_DIR] TARGET MODEL_DIR

  Lint-check a codebase using HECTOR.

Options:
  --device DEVICE             Device on which to run.
  --llap-lib-dir DIRECTORY    Directory containing HECTOR opt passes.
                              [default: /usr/local/lib]

  --threshold FLOAT RANGE     Decision threshold probability.  [default: 0.5]
  --top K                     Show only K most-likely vulnerabilities (per
                              CWE).

  --output FILENAME           File where output will be written.  [default: -]
  --output-format [lint|csv]  Output style  [default: lint]
  --help                      Show this message and exit.
```
By default, output is written to standard output in a lint-like text format.
You can alternatively request CSV output by passing `--output-format csv`.
You can also send the output to a file by passing `--output path/to/file`.



### More Tips 👍:
- Each step in this pipeline can be done in parallel on each project except for (5) since each run of the preprocessor requires exclusive access to the shared index file.
- `hector` has some other useful tools. For example, you can use `hector train_test_split` to create a random train/test datasets from a sub-ePDG dataset (e.g., juliet). You can also ise `hector hyperopt` to use hyperparameter optimization to find the best configuration for your model on your CWE dataset.
- The VulChecker pipeline was written for research and not production. This means that there are many ways in which the code can be revised to run faster. For example, the loading of ePDGs is incredibly slow and could be expidited if stored in raw serialized format. In general, the loading and storing of the data to/from disk at each step should be eliminated wherever possible. Other optimizations could be made in model training to limit the amount of data loaded into memory at a time. Also, computing the betweeness centrality measure takes the majority of the time when extracting features (this could be replaced with approximations).


# Assets
For reproducability, we provide the datasets and models used in our paper. The models are hosted in this repository. The VM and datasets are hosted on OneDrive for the time being.

## Datasets
We provide the cmake projects used in our paper in both source code format, ePDG format, processed as subgraph ePDGs and as the final datasets used to train our models.
This data can be accessed via OneDrive: source code and processed graphs

There are two folders, `Origional Projects` and `Processed Graphs`:

### Origional Projects
We provide the source code to both the Juliet and Wild Labeled (Github with CVE) projects.
- Juliet can be found on OneDrive [here](https://bgu365-my.sharepoint.com/:f:/g/personal/yisroel_bgu_ac_il/EuvGBQXY-WBIsZcRhYoO1dwBtw4CoQVlWx12BhL_pBdtOg?e=vuyajw_
- Wild Labeled can be found on OneDrive [here](https://bgu365-my.sharepoint.com/:u:/g/personal/yisroel_bgu_ac_il/EWtBXjeUMyZFoQyh-QudRKkBNQzLnDRLftgavWTOSALIMQ?e=WVbRiG)
If a project is labeled then it will contain label.json file in its directory. 

The Juliet samples have already gone though some preprocessing. Their directory contains a zip of labeled files from the Juliet dataset. Each directory has the contents of a single CVE and includes labeled graphs produced by llap (labeled\_graphs), LLVM IR files (ll_files), preprocessed source files from Juliet (source\_files), and labels produced from comments in the source files (source\_labels). Files that end in `omitgood` correspond to test cases that contain vulnerabilities, while files that end in `omitbad` contain no vulerabilities.

### Processed Graphs
We provide the processed graphs (at differnt levels of prerpocessing). These files can be found on OneDrive [here](https://bgu365-my.sharepoint.com/:f:/g/personal/yisroel_bgu_ac_il/EpTfUBgdSTVGiJm4-GEODUEBFXuOc9xfZHkOIkau--42kA?e=FwxeGW)
The directory contains:

- All of the ePDG files (before and after subgraph processing) 
- The paper's final train and test sets (the concatenation of the relevant graphs with downsampling)
- The data nomalization parameters used in the paper on these datasets (feature_stats)
- The indexes used in the paper, from the respective datasets

Structure:

- `/` The data is organized by CWE. C/C++ projects with multiple CWEs will appear in multiple directories (however, they will have been processed by different CWE pipelines). Note, although there are CWE191 samples, they were not included in the paper evalaution 
- `/CWE<id>/` contains all of the data for CWE `<id>`. It also contains the normalization parameters (`feature_stats_nd.npz`) and indexes (`indexes-190.json`) needed for a model that would be trained on the augmented dataset for this CWE.
- `/CWE<id>/graphs` contains the complete ePDGs for each project
- `/CWE<id>/graphs/juliet` contains a single json with all of the ePDGs in Juliet (one per row)
- `/CWE<id>/graphs/oob` contains the ePDGs for the unlabled projects left out of train and test
- `/CWE<id>/graphs/wild` contains the ePDGs for graphs that have CVE labels for the CWE `<id>` (`./labeled`) and that do not have labels for the given CWE (`./unabeled`). Some projects in `./unlabeled` were not used in the final dataset due to their size or number of potential manifestaion points. These ePDGs are in the subdir `./labeled/omitted`
- `/CWE<id>/graphs/wildaug` contains the ePDGs of augmented graphs. Augmentation is done by taking the juliet ePDGs from `/CWE<id>/graphs/juliet` and injecting them into each ePDG in `/CWE<id>/graphs/wild/unlabled` 

- `/CWE<id>/proc_graphs` contains json files of the processed ePDGs from the preprocessing step (each project is a single json where each line is a subgraph from the origional ePDG). It also contains the final datasets before and after downsampling the number of negative manifestation points. In this dir you will find `CWE190_augANDcve.json.gz` which is the final dataset if you inted to train a model on both augmented data and the labeled data (no testset).
- `/CWE<id>/proc_graphs/synth_real-labels` contains the Juliet preprocesed graphs
- `/CWE<id>/proc_graphs/wild_augmented-labels` contains the augmented projects' preprocesed graphs in both `./individual` and `./combined` formats. A json in `./combined` is a complete and final dataset for training or testing. The `./individual` files are useful if you want to create custom dataset splits among projects or evalaute a specific project alone. 
- `/CWE<id>/proc_graphs/wild_no-labels` is the preprocessed graphs from projects that did not have a labels for CVEs for the given CWE.
- `/CWE<id>/proc_graphs/wild_real-labels` is the preprocessed graphs from projects that do have have a labels for CVEs for the given CWE. This data was used as the test set in the paper.
- Files in `/CWE<id>/proc_graphs/*/combined` have the following format: `CWE<id>_*.json.gz` is the complete and origional set of subgraphs from all projects, `CWE<id>_*_clean.json.gz` is the same dataset after validation (some subgraphs may have been removed), and `CWE<id>_*_clean_<N>_<P>.json.gz` is the subsequent dataset after removing a ratio of `<N>` negative and `<P>` positive manifestation points (the actual dataset used in the paper).




## Models
The `models/` directory [in this repository](https://github.com/ymirsky/VulChecker/tree/main/models) contains two set of models
- `models/trained_on_aug`: Models trained on augmented samples only (Juliet mixed into 'clean' GitHub Projects)
- `models/trained_on_aug_and_cve`: Models trained on both augmented samples and CVE samples from Github.

In each of these directories you will find the models organized by CWE (190, 121, 122, 415 and 416). The subdirectories contain one or two different versions of a model trained. For each model, we provide:
- `model/`: the model itself
- `params.txt`: the parameters (full command) used to train this model
- csv and npz dump of the performance on the testset (where applicable)
- csv and npz dump of the performance on the holdout set (where applicable)
- roc plots of the evalautions

## VM
As mentioned earlier, you can get immediate access to a fully operational VulChecker by downloading our VM.
The VM is [hosted on OneDrive](https://bgu365-my.sharepoint.com/:f:/g/personal/yisroel_bgu_ac_il/Et1JFmXJFFREmBQewk-5GhIBxgFDxNihBYvTR7ZAOsC_Zg?e=btAJzX).

Note 📝: The VM is not configure to use a GPU so model training and execution may be very slow.


# Developer Notes
In this section we give low level technical information for those who need to modify or extend parts of the code. 

## Quick Start
`hector` depends on [NetworKit](https://networkit.github.io/),
which uses Cython but doesn't declare that according to PEP 518.
You must have Cython installed before attempting to resolve the NetworKit dependency.
We can work around this by pre-building wheels and
setting the `PIP_FIND_LINKS` environment variable:

```bash
for py_interp in python3.6 python3.7 python3.8; do
    $py_interp -m venv build-env
    . build-env/bin/activate
    python -m pip install -U pip setuptools wheel
    python -m pip install cython cmake
    python -m pip wheel -w wheelhouse networkit
    deactivate
    rm -rf build-env
done
export PIP_FIND_LINKS="$PWD/wheelhouse"
```

You will also need to add a wheel for
[`structure2vec`](https://github.gatech.edu/HECTOR/structure2vec)
to the `wheelhouse` directory.
Don't forget to set `PIP_FIND_LINKS` each time you start a new shell.

With the project cloned and a virtual environment active:

```bash
pip install -e .[dev,tests,docs]
```

You should configure [pre-commit](https://pre-commit.com/) to check your code before you commit:

```bash
pre-commit install
```

To run the tests, you will need all supported versions of Python installed.
On Ubuntu, you can use the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).
In other places, you can use [pyenv](https://github.com/pyenv/pyenv).
You can run the automated tests by saying:

```bash
tox
```

## Performance Critical Code (Cython)

hector_ml uses [Cython](https://cython.org/) for performance-critical functions.
The Cython files are named `_foo.pyx`,
and should be imported in the corresponding `foo.py` file.
It's also OK to `cimport` the Cython objects from other Cython source files.

### Graphs to Matrices

When training models,
we discovered that it took a very long time to load the data into memory.
Profiling just the data loading part revealed that
converting the graphs to matrix representation was taking most of the time.
I therefore converted that code (`mean_field_from_node_link_data`) to Cython.

The `feature_row` function was taking up a plurality of the internal time of that function,
so I reduced the dynamism by creating a "compiled feature" Cython extension class
that remembers the number of columns for each feature
(I call this the feature's "width").
There's a concrete class for each feature kind,
so the dynamic dispatch into the individual handlers
becomes an indirect function call at the C level.

## Features
The following features are used to train structure2vec models:

## Node Features

| Feature Name | Feature Identifier | Computed By | Comment |
| ------------ | ------------------ | ----------- | ------- |
| Static Value | `static_value` | | |
| Operation | `operation` | LLAP | |
| Basic Function | `function` | | name of function defined in other compilation unit |
| Output dtype | `dtype` | LLAP | |
| Part of "if" clause | `condition` | LLAP | |
| Number of data dependents | `def_use_out_degree` | HECTOR | |
| Number of control dependents | `control_flow_out_degree` | HECTOR | |
| Betweenness | `betweenness` | HECTOR | |
| Distance to manifestation point | `distance_manifestation` | HECTOR | |
| Distances to nearest root cause point | `distance_root_cause` | HECTOR | |
| Operation of nearest root cause point | `nearest_root_cause_op` | HECTOR | `call` or plurality or uniform random |
| Node tag | `tag` | LLAP | list-set of {`root_cause`, `manifestation`} |

### Node Metadata

| Metadata Description | Metadata Identifier |
| -------------------- | ------------------- |
| Containing function | `containing_function` |
| Source file | `file` |
| Source line | `line_number` |
| Training label | `label` |

### Edge Features

| Feature Name | Feature Identifier | Computed By | Comment |
| ------------ | ------------------ | ----------- | ------- |
| dtype | `dtype` | LLAP | |
| edge type | `type` | LLAP | |

## Input Graph Structure

Input should be in the node-link JSON format.
That looks like this:

``` json
{
    "graph": {},
    "nodes": [
        {
            "id": 0,

            "static_value": null,
            "operation": "add",
            "function": null,
            "dtype": "int64",
            "condition": false,
            "tag": [],
            "file": "foo.c",
            "line_number": 27,
            "containing_function": "foo",
            "label": "negative"
        }
    ],
    "links": [
        {
            "source": 0,
            "target": 0,

            "type": "def_use",
            "dtype": "int64"
        }
    ]
}
```

Every node needs a unique ID in order to match the edges to the nodes.
The unique ID has no semantic meaning,
and so you can simply assign sequential numbers.
The objects for `graph`, `nodes`, and `links` can contain arbitrary additional data.
Only the `id`, `source`, and `target` keys are reserved.

Here are some more notes on the implimentation:
When processing unlabeled input, omit the `label` key from the node data.

The graph-structure features
(betweenness,
distance to manifestation,
distance to nearest root-cause,
operation of nearest root-cause) are computed in hector's code.

For categorical features,
two passes over the data are made:
once to find out what all the possible values are,
and again to produce one-hot vectors of the appropriate size.
That means it doesn't matter what exact values are produced.

Hector specially handles some categorical values.
It recognizes
`"tag": ["manifestation"]` and `"tag": ["root_cause"]`
for producing graph features.
It recognizes
`"operation": "call"`
for breaking ties on the operation of the nearest root cause.
If those aren't the most natural values, then they can be swapped them out for something else.

For ease of combining multiple outputs into a data set,
the JSON should be output in minified form;
specifically, it should be on a single line with a trailing newline.

# Citations
If you use any derivative of the code or datasets from our work, please cite our publicaiton:
```
@inproceedings{mirskyvulchecker,
  title={VulChecker: Graph-based Vulnerability Localization in Source Code},
  author={Mirsky, Yisroel and Macon, George and Brown, Michael and Yagemann, Carter and Pruett, Matthew and Downing, Evan and Mertoguno, Sukarno and Lee, Wenke}
  booktitle={USENIX Security},
  year={2023}
}
```

## Acks
Special thanks to the lead developers: Michael Brown for his work on LLAP and George Macon for his work on `hector`! 🍻


