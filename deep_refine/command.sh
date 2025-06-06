PROJECT_DIR=/home/afzal/tools/VeriNN
BIN_NAME=drefine
g++ -g \
  deeppoly/*.cc src/main.cpp src/lib/*.cc k/findk.cc \
  -I${PROJECT_DIR}/deep_refine/ex_tools/gurobi912/linux64/include \
  -I${PROJECT_DIR}/deep_refine/ex_tools/xt-build/include \
  -I${PROJECT_DIR}/deep_refine/ex_tools/boost_1_68_0/installed/include \
  -L${PROJECT_DIR}/deep_refine/ex_tools/gurobi912/linux64/lib \
  -L${PROJECT_DIR}/deep_refine/ex_tools/boost_1_68_0/installed/lib \
  -L${CONDA_PREFIX}/lib \
  -lblas \
  -lboost_program_options \
  -lpthread \
  -lgurobi_c++ \
  -lgurobi91 \
  -o ${BIN_NAME}

export LD_LIBRARY_PATH=${PROJECT_DIR}/deep_refine/ex_tools/gurobi912/linux64/lib:${PROJECT_DIR}/deep_refine/ex_tools/boost_1_68_0/installed/lib:${CONDA_PREFIX}/lib
