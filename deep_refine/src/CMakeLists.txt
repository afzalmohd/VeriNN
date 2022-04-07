cmake_minimum_required(VERSION 2.8.7)

project(${DREFINE})

# find_package (BISON REQUIRED)
# find_package(FLEX REQUIRED)

#set(CMAKE_POSITION_INDEPENDENT_CODE ON) # requires version 2.8.9

# BISON_TARGET(MyParser inv2smt/inv2smt.y ${CMAKE_CURRENT_BINARY_DIR}/inv2smt.y.cpp)
# FLEX_TARGET(MyScanner inv2smt/inv2smt.l  ${CMAKE_CURRENT_BINARY_DIR}/inv2smt.l.cpp)
# ADD_FLEX_BISON_DEPENDENCY(MyScanner MyParser)

# SET (LLVM_DEBUG FALSE CACHE BOOL "Use Locally compiled and debug enabled LLVM")
SET (Z3_DEBUG FALSE CACHE BOOL "Link debug verion of Z3")
SET (CMAKE_VERBOSE_MAKEFILE OFF)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message("debug mode")
  IF(NOT DEFINED LLVM_DEBUG)
    SET( LLVM_DEBUG TRUE )
  ENDIF(NOT DEFINED LLVM_DEBUG)
  SET (Z3_DEBUG TRUE)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../gurobi_cmake")
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../build/xtl")
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../build/xtensor")
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../build/xtensor-blas")
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# find_package(xtensor REQUIRED CONFIG)
# find_package(xtl REQUIRED CONFIG)
find_package(GUROBI REQUIRED)
if(GUROBI_FOUND)
  message(STATUS "GUROBI FOUND")
else(GUROBI_FOUND)
  message(FATAL_ERROR "!!! GUROBI NOT FOUND !!!")
endif(GUROBI_FOUND)

#find_package(boost REQUIRED CONFIG)
# include_directories()
include_directories(${GUROBI_INCLUDE_DIRS})
#include_directories(${xtl_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}/../build/xt-build/include")
# include_directories("${CMAKE_SOURCE_DIR}/../build/boost_1_78_0/installed/include")
# include_directories("${CMAKE_SOURCE_DIR}/../build/lapack-release/installed/include")

message(STATUS "Found XTENSOR ${xtensor_FOUND}")
message(STATUS "Found XTL ${XTENSOR_PACKAGE_VERSION}")

# message(STATUS "DEBUG : LLVM - ${LLVM_DEBUG}")

# IF(LLVM_DEBUG)
#   SET( LLVM_DIR "${INSTALLED}/llvm-${LLVM_VERSION}")
#   SET( CLANG_DIR "${INSTALLED}/llvm-${LLVM_VERSION}")
#   find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH HINTS "${LLVM_DIR}/lib/cmake/llvm/")
#   find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH HINTS "${CLANG_DIR}/lib/cmake/clang/")
# ELSE(LLVM_DEBUG)
#   # change here to modify the relase version of LLVM
#   find_package(LLVM 10 REQUIRED CONFIG)
#   #find_package(Clang REQUIRED CONFIG)
#   # find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH HINTS "/usr/lib/llvm-10/lib/cmake/clang")
#   find_package(Clang REQUIRED CONFIG NO_DEFAULT_PATH HINTS "/usr/lib/cmake/clang-10/")
#   message(STATUS "LLVM INCLUDE DIR - ${LLVM_INCLUDE_DIRS}")
#   message(STATUS "CLANG INCLUDE DIR - ${CLANG_INCLUDE_DIRS}")
#   # find_package(Clang REQUIRED clangTooling libClang)
# ENDIF(LLVM_DEBUG)

# message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
# message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

SET(Boost_USE_STATIC_LIBS    OFF)
SET(Boost_USE_STATIC_RUNTIME OFF)

# find_package(Boost 1.58.0 COMPONENTS regex program_options system filesystem REQUIRED) 

# IF(Z3_DEBUG)
#   LINK_DIRECTORIES("${PROJECT_BINARY_DIR}/../z3/buildd")
# ELSE(Z3_DEBUG)
#   LINK_DIRECTORIES("${PROJECT_BINARY_DIR}/../z3/buildr")
# ENDIF(Z3_DEBUG)

LINK_DIRECTORIES("${GUROBI_LIBRARY_PATH}")
# LINK_DIRECTORIES("${CMAKE_SOURCE_DIR}/../build/boost_1_78_0/installed/lib")
# LINK_DIRECTORIES("${CMAKE_SOURCE_DIR}/../build/OpenBLAS-0.3.9/installed/lib")
# LINK_DIRECTORIES("${CMAKE_SOURCE_DIR}/../build/lapack-release/installed/lib")

ADD_DEFINITIONS(-std=c++14 -Wall)

include_directories (${CMAKE_CURRENT_BINARY_DIR} "${PROJECT_SOURCE_DIR}") #"${PROJECT_BINARY_DIR}/../z3/src/api" "${PROJECT_BINARY_DIR}/../z3/src/api/c++")
#SET(SOURCES lib/backprop.cc lib/network.cc lib/z3expr.cc lib/configuration.cc main.cpp) 
SET(SOURCES main.cpp lib/drefine_driver.cc lib/pullback.cc lib/decision_making.cc lib/milp_refine.cc lib/milp_mark.cc ../deeppoly/analysis.cc ../deeppoly/deeppoly_configuration.cc ../deeppoly/deeppoly_driver.cc ../deeppoly/helper.cc ../deeppoly/interval.cc ../deeppoly/network.cc ../deeppoly/optimizer.cc ../deeppoly/parser.cc)
add_executable(${DREFINE} ${SOURCES})

SET(LIBZ3 z3)
SET(xtensor_libs blas)
SET(boost_lib boost_program_options)
SET(pthread_lib pthread)
SET(gurobi_lib gurobi_c++ gurobi91)
SET(cblas_lib cblas)
SET(gfortran_lib gfortran)

target_link_libraries (${DREFINE} ${xtensor_libs} ${boost_lib} ${pthread_lib} ${gurobi_lib})
