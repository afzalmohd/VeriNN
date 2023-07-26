PROJECT_DIR=/home/smit/mtp/VeriNN
BIN_NAME=drefine-temp
g++ -g deeppoly/*.cc src/main.cpp src/lib/*.cc -I${PROJECT_DIR}/deep_refine/ex_tools/gurobi912/linux64/include -I${PROJECT_DIR}/deep_refine/ex_tools/xt-build/include -L${PROJECT_DIR}/deep_refine/ex_tools/gurobi912/linux64/lib -lblas -lboost_program_options -lpthread -lgurobi_c++ -lgurobi91 -o ${BIN_NAME}