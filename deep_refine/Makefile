git = git -c user.name="Auto" -c user.email="auto@auto.com" 

DREFINE=drefine
HOME_INSTALLED=~/installed
BUILDDIR = $(PWD)/build
SRCDIR = $(PWD)/src
DEEPPOLYDIR = $(PWD)/deeppoly
DEEPPOLY=deeppoly
CMAKE_DIR=$(PWD)

all : release

.PHONY : release debug run clean test deepclean deeppoly deeppolydebug

release : $(BUILDDIR)/buildr/Makefile
	+make -C $(BUILDDIR)/buildr
	cp -f $(BUILDDIR)/buildr/$(DREFINE) $(DREFINE)

debug :  $(BUILDDIR)/buildd/Makefile
	+make -C $(BUILDDIR)/buildd
	rm -rf $(DREFINE)
	ln -s $(BUILDDIR)/buildd/$(DREFINE) $(DREFINE)


$(BUILDDIR)/buildr/Makefile:
	mkdir -p $(BUILDDIR)/buildr
	cd $(BUILDDIR)/buildr; cmake -DCMAKE_BUILD_TYPE=Release -DDREFINE=$(DREFINE) $(CMAKE_DIR) 

$(BUILDDIR)/buildd/Makefile:
	mkdir -p $(BUILDDIR)/buildd
	cd $(BUILDDIR)/buildd; cmake -DCMAKE_BUILD_TYPE=Debug -DINSTALLED=$(HOME_INSTALLED) -DDREFINE=$(DREFINE) $(CMAKE_DIR)

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
	rm -rf $(PWD)/ex_tools/boost_1_68_0
	rm -rf $(PWD)/ex_tools/gurobi912
	rm -rf $(PWD)/ex_tools/xtl
	rm -rf $(PWD)/ex_tools/xtensor
	rm -rf $(PWD)/ex_tools/xtensor-blas
	rm -rf $(PWD)/ex_tools/xt-build


#-----------------------------------------------------------------------------

test: release runtest
