git = git -c user.name="Auto" -c user.email="auto@auto.com" 

DREFINE=drefine
HOME_INSTALLED=~/installed
BUILDDIR = $(PWD)/build
SRCDIR = $(PWD)/src
XTENSOR_VERSION=0.3.0  # dummy now
XTL_VERSION=0.4.0 # dummy now
LD=ld.gold

all : release

.PHONY : release debug run clean test deepclean

release : $(BUILDDIR)/buildr/Makefile
	+make -C $(BUILDDIR)/buildr
	cp -f $(BUILDDIR)/buildr/$(DREFINE) $(DREFINE)

debug :  $(BUILDDIR)/buildd/Makefile
	+make -C $(BUILDDIR)/buildd
	rm -rf $(DREFINE)
	ln -s $(BUILDDIR)/buildd/$(DREFINE) $(DREFINE)


$(BUILDDIR)/buildr/Makefile: $(BUILDDIR)/z3/buildr/libz3.so
	mkdir -p $(BUILDDIR)/buildr
	cd $(BUILDDIR)/buildr; cmake -DCMAKE_BUILD_TYPE=Release -DDREFINE=$(DREFINE) $(SRCDIR)

$(BUILDDIR)/buildd/Makefile: $(BUILDDIR)/z3/buildd/libz3.so 
	mkdir -p $(BUILDDIR)/buildd
	cd $(BUILDDIR)/buildd; cmake -DCMAKE_BUILD_TYPE=Debug -DINSTALLED=$(HOME_INSTALLED) -DDREFINE=$(DREFINE) $(SRCDIR)

clean :
	rm -rf $(BUILDDIR)/buildr
	rm -rf $(BUILDDIR)/buildd
	rm -f $(DREFINE) 
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


#-----------------------------------------------------------------------------
# Xtensor-blas fetch and install



test: release runtest
