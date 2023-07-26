CURR_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
all : boost gurobi xtl xtensor xtensor-blas

boost:
	cd $(CURR_DIR); $(info installing boost) wget -q https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.tar.gz/download -O boost_1_68_0.tar.gz 
	cd $(CURR_DIR); tar xzvf boost_1_68_0.tar.gz >> /dev/null; rm boost_1_68_0.tar.gz
	mkdir -p $(CURR_DIR)/boost_1_68_0/installed
	cd $(CURR_DIR)/boost_1_68_0; ./bootstrap.sh --prefix=$(CURR_DIR)/boost_1_68_0/installed --with-libraries=program_options,timer,chrono,thread >> /dev/null
	cd $(CURR_DIR)/boost_1_68_0; ./b2 cxxflags=-fPIC link=static install >> /dev/null

gurobi: $(CURR_DIR)/gurobi912/linux64/src/build/libgurobi_c++.a

$(CURR_DIR)/gurobi912/linux64/src/build/libgurobi_c++.a:
	cd $(CURR_DIR); $(info installing gurobi) wget -q https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz -O gurobi9.1.2_linux64.tar.gz
	cd $(CURR_DIR); tar -xvf gurobi9.1.2_linux64.tar.gz >> /dev/null; rm gurobi9.1.2_linux64.tar.gz
	cd $(CURR_DIR)/gurobi912/linux64/src/build; make
	cp -f $(CURR_DIR)/gurobi912/linux64/src/build/libgurobi_c++.a $(CURR_DIR)/gurobi912/linux64/lib

xtl:
	cd $(CURR_DIR); git clone -b 0.7.4 https://github.com/xtensor-stack/xtl.git
	mkdir -p $(CURR_DIR)/xt-build
	cd $(CURR_DIR)/xtl; cmake -DCMAKE_PREFIX_PATH=$(CURR_DIR)/xt-build -DCMAKE_INSTALL_PREFIX=$(CURR_DIR)/xt-build
	cd $(CURR_DIR)/xtl; make install

xtensor:
	cd $(CURR_DIR); git clone -b 0.24.2 https://github.com/xtensor-stack/xtensor.git
	mkdir -p $(CURR_DIR)/xt-build
	cd $(CURR_DIR)/xtensor; cmake -DCMAKE_INSTALL_PREFIX=$(CURR_DIR)/xt-build
	cd $(CURR_DIR)/xtensor; make install

xtensor-blas:
	cd $(CURR_DIR); git clone -b 0.20.0 https://github.com/xtensor-stack/xtensor-blas.git
	mkdir -p $(CURR_DIR)/xt-build
	cd $(CURR_DIR)/xtensor-blas; cmake -DCMAKE_INSTALL_PREFIX=$(CURR_DIR)/xt-build
	cd $(CURR_DIR)/xtensor-blas; make install

openblas:
	cd $(CURR_DIR); wget -q https://github.com/xianyi/OpenBLAS/releases/download/v0.3.19/OpenBLAS-0.3.19.tar.gz -O OpenBLASv0.3.19.tar.gz
	cd $(CURR_DIR); tar -xzf OpenBLASv0.3.19.tar.gz >> /dev/null; rm OpenBLASv0.3.19.tar.gz
	cd $(CURR_DIR)/OpenBLAS-0.3.19; make NO_SHARED=1 CBLAS_ONLY=1 USE_THREAD=1 >> /dev/null
	mkdir -p $(CURR_DIR)/OpenBLAS-0.3.19/installed
	cd $(CURR_DIR)/OpenBLAS-0.3.19; make PREFIX=$(CURR_DIR)/OpenBLAS-0.3.19/installed NO_SHARED=1 install >> /dev/null
