git = git -c user.name="Auto" -c user.email="auto@auto.com" 

DREFINE=drefine
HOME_INSTALLED=~/installed
BUILDDIR = $(PWD)/build
SRCDIR = $(PWD)/src
DEEPPOLYDIR = $(PWD)/deeppoly
DEEPPOLY=deeppoly
export GUROBI_HOME=$(BUILDDIR)/gurobi912/linux64
export XT_HOME = $(BUILDDIR)/xt-build
XTENSOR_VERSION=0.3.0  # dummy now
XTL_VERSION=0.4.0 # dummy now
LD=ld.gold

all : release

.PHONY : release debug run clean test deepclean deeppoly deeppolydebug

release : $(BUILDDIR)/buildr/Makefile
	+make -C $(BUILDDIR)/buildr
	cp -f $(BUILDDIR)/buildr/$(DREFINE) $(DREFINE)

debug :  $(BUILDDIR)/buildd/Makefile
	+make -C $(BUILDDIR)/buildd
	rm -rf $(DREFINE)
	ln -s $(BUILDDIR)/buildd/$(DREFINE) $(DREFINE)


$(BUILDDIR)/buildr/Makefile: $(BUILDDIR)/gurobi912/linux64/src/build/libgurobi_c++.a xt_build
	mkdir -p $(BUILDDIR)/buildr
	cd $(BUILDDIR)/buildr; cmake -DCMAKE_BUILD_TYPE=Release -DDREFINE=$(DREFINE) $(SRCDIR) 

$(BUILDDIR)/buildd/Makefile: $(BUILDDIR)/gurobi912/linux64/src/build/libgurobi_c++.a xt_build
	mkdir -p $(BUILDDIR)/buildd
	cd $(BUILDDIR)/buildd; cmake -DCMAKE_BUILD_TYPE=Debug -DINSTALLED=$(HOME_INSTALLED) -DDREFINE=$(DREFINE) $(SRCDIR)

clean :
	rm -rf $(BUILDDIR)/buildr
	rm -rf $(BUILDDIR)/buildd
	rm -f $(DREFINE) 
	rm -rf $(BUILDDIR)/deeppoly/buildr
	rm -rf $(BUILDDIR)/deeppoly/buildd
	rm -f $(DEEPPOLYDIR)/$(DEEPPOLY)
	find -name "*~"| xargs rm -rf

# removes all the downloaded libs and all the installs
deepclean: clean
	rm -rf $(BUILDDIR)/*

#-----------------------------------------------------------------------------
# Z3 fetch and install

$(BUILDDIR)/z3/README.md :
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR);$(git) clone https://github.com/Z3Prover/z3.git

$(BUILDDIR)/z3/buildr/libz3.so : $(BUILDDIR)/z3/README.md
	rm -rf $(BUILDDIR)/z3/buildr
	mkdir -p $(BUILDDIR)/z3/buildr
	cd $(BUILDDIR)/z3/buildr; cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../
	+make -C $(BUILDDIR)/z3/buildr

# $(BUILDDIR)/z3/buildr/libz3.a : $(BUILDDIR)/z3/buildr/libz3.so

$(BUILDDIR)/z3/buildd/libz3.so : $(BUILDDIR)/z3/README.md
	rm -rf $(BUILDDIR)/z3/buildd
	mkdir -p $(BUILDDIR)/z3/buildd
	cd $(BUILDDIR)/z3/buildd; cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ../
	+make -C $(BUILDDIR)/z3/buildd


deeppolydebug: $(BUILDDIR)/deeppoly/buildd/Makefile 
	+make -C $(BUILDDIR)/deeppoly/buildd
	cp -f $(BUILDDIR)/deeppoly/buildd/$(DEEPPOLY) $(DEEPPOLYDIR)/$(DEEPPOLY)

$(BUILDDIR)/deeppoly/buildd/Makefile: $(BUILDDIR)/gurobi912/linux64/src/build/libgurobi_c++.a
	mkdir -p $(BUILDDIR)/deeppoly/buildd
	cd $(BUILDDIR)/deeppoly/buildd; cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DDEEPPOLY=$(DEEPPOLY) $(DEEPPOLYDIR)

deeppoly: $(BUILDDIR)/deeppoly/buildr/Makefile 
	+make -C $(BUILDDIR)/deeppoly/buildr
	cp -f $(BUILDDIR)/deeppoly/buildr/$(DEEPPOLY) $(DEEPPOLYDIR)/$(DEEPPOLY)

$(BUILDDIR)/deeppoly/buildr/Makefile: $(BUILDDIR)/gurobi912/linux64/src/build/libgurobi_c++.a
	mkdir -p $(BUILDDIR)/deeppoly/buildr
	cd $(BUILDDIR)/deeppoly/buildr; cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DDEEPPOLY=$(DEEPPOLY) $(DEEPPOLYDIR)

$(BUILDDIR)/gurobi912/linux64/src/build/libgurobi_c++.a:
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR); wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz; tar -xvf gurobi9.1.2_linux64.tar.gz; rm gurobi9.1.2_linux64.tar.gz
	cd $(BUILDDIR)/gurobi912/linux64/src/build; make
	cp -f $(BUILDDIR)/gurobi912/linux64/src/build/libgurobi_c++.a $(BUILDDIR)/gurobi912/linux64/lib

$(BUILDDIR)/xtl/README.md:
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR);$(git) clone https://github.com/xtensor-stack/xtl.git

$(BUILDDIR)/xtensor-blas/README.md:
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR);$(git) clone https://github.com/xtensor-stack/xtensor-blas.git

$(BUILDDIR)/xtensor/README.md:
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR);$(git) clone https://github.com/xtensor-stack/xtensor.git

$(BUILDDIR)/xt-build/include/xtl/xtl_config.hpp: $(BUILDDIR)/xtl/README.md
	mkdir -p $(BUILDDIR)/xt-build
	cd $(BUILDDIR)/xtl;cmake -DCMAKE_INSTALL_PREFIX=$(BUILDDIR)/xt-build
	cd $(BUILDDIR)/xtl; make install

$(BUILDDIR)/xt-build/include/xtensor/xtensor_config.hpp: $(BUILDDIR)/xtensor/README.md
	mkdir -p $(BUILDDIR)/xt-build
	cd $(BUILDDIR)/xtensor;cmake -DCMAKE_INSTALL_PREFIX=$(BUILDDIR)/xt-build
	cd $(BUILDDIR)/xtensor; make install

$(BUILDDIR)/xt-build/include/xtensor-blas/xblas_config.hpp: $(BUILDDIR)/xtensor-blas/README.md
	mkdir -p $(BUILDDIR)/xt-build
	cd $(BUILDDIR)/xtensor-blas;cmake -DCMAKE_INSTALL_PREFIX=$(BUILDDIR)/xt-build
	cd $(BUILDDIR)/xtensor-blas; make install

$(BUILDDIR)/boost_1_78_0/boost/program_options.hpp:
	cd $(BUILDDIR); wget https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.bz2; tar -xvf boost_1_78_0.tar.bz2; rm boost_1_78_0.tar.bz2
	cd $(BUILDDIR)/boost_1_78_0; ./bootstrap.sh --prefix=$(BUILDDIR)/boost_1_78_0/installed --with-libraries=program_options,timer,chrono,thread; mkdir -p $(BUILDDIR)/boost_1_78_0/installed; ./b2 cxxflags=-fPIC link=static install

$(BUILDDIR)/lapack-release/installed/lib/libblas.a:
	cd $(BUILDDIR); git clone -b lapack-3.10.0 https://github.com/Reference-LAPACK/lapack-release.git
	mkdir -p $(BUILDDIR)/lapack-release/build
	mkdir -p $(BUILDDIR)/lapack-release/installed
	cd $(BUILDDIR)/lapack-release/build; cmake -DCMAKE_INSTALL_PREFIX=$(BUILDDIR)/lapack-release/installed -DCBLAS=ON ..; cmake --build . -j --target install


xt_build: $(BUILDDIR)/xt-build/include/xtl/xtl_config.hpp $(BUILDDIR)/xt-build/include/xtensor/xtensor_config.hpp $(BUILDDIR)/xt-build/include/xtensor-blas/xblas_config.hpp

test: release runtest
