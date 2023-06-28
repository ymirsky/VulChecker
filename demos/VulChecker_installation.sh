# The following script installs VulChecker and all of it's dependencies
# *** Do not run this script on this VM (these actions have already been performed)

while true; do
    read -p "Do you wish to install VulChecker? Do not run this on the supplied VM!" yn
    case $yn in
        [Yy]* ) 
        
		
	cd ~

	# Get VulChecker, Structure2Vec, and the preprocessing pipeline LLAP
	git clone https://github.com/ymirsky/VulChecker.git
	git clone https://github.com/gtri/structure2vec.git
	git clone https://github.com/michaelbrownuc/llap.git

	# install requite linux libs
 	sudo apt install cmake
	sudo apt install python3-pip

	# install them
	python3 -m pip install -U pip setuptools wheel
	python3 -m pip install cython cmake
	python3 -m pip install ./structure2vec 
	python3 -m pip --no-cache-dir install ./VulChecker

	# check if path to hector works
	hector --help

	# install ninja
	sudo apt-get install -y ninja-build

	# install LLVM v10
	wget https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/llvm-project-10.0.0.tar.xz
	tar xvf llvm-project-10.0.0.tar.xz
	mv llvm-project-10.0.0 llvm-project

	# add the llap plugins to LLVM
	cp -R llap-master/src/* llvm-project/llvm/lib/Transforms/

	# compile LLVM
	cd llvm-project/
	cmake -S ./llvm/ -B llvm-build -DCMAKE_BUILD_TYPE=Release
	make -C llvm-build -j 16
	make -C llvm-build install 
	cmake -S ./clang/ -B clang-build -DCMAKE_BUILD_TYPE=Release
	make -C clang-build -j 16
	make -C clang-build install
        
        
        break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

