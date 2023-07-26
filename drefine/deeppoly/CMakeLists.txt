cmake_minimum_required(VERSION 2.8.7)

project(${DEEPPOLY})

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
#message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(xtensor REQUIRED CONFIG)
find_package(xtl REQUIRED CONFIG)
find_package(GUROBI REQUIRED)

if(GUROBI_FOUND)
  message(STATUS "GUROBI FOUND")
else(GUROBI_FOUND)
  message(FATAL_ERROR "!!! GUROBI NOT FOUND !!!")
endif(GUROBI_FOUND)
#message("Check......")
#find_package(boost REQUIRED CONFIG)
# include_directories()
include_directories(${xtl_INCLUDE_DIRS})
include_directories(${GUROBI_INCLUDE_DIRS})
#message("GUROBI_LIBRARY: ${GUROBI_LIBRARY}")

message(STATUS "Found XTENSOR ${xtensor_FOUND}")
message(STATUS "Found XTL ${XTENSOR_PACKAGE_VERSION}")


SET(Boost_USE_STATIC_LIBS    OFF)
SET(Boost_USE_STATIC_RUNTIME OFF)

# find_package(Boost 1.58.0 COMPONENTS regex program_options system filesystem REQUIRED) 

IF(Z3_DEBUG)
  LINK_DIRECTORIES("${PROJECT_BINARY_DIR}/../z3/buildd")
ELSE(Z3_DEBUG)
  LINK_DIRECTORIES("${PROJECT_BINARY_DIR}/../z3/buildr")
ENDIF(Z3_DEBUG)

LINK_DIRECTORIES("${GUROBI_LIBRARY_PATH}")

ADD_DEFINITIONS(-std=c++14 -Wall)

include_directories (${CMAKE_CURRENT_BINARY_DIR} "${PROJECT_SOURCE_DIR}" )
# include_directories(${Boost_INCLUDE_DIRS})
# LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})


SET(SOURCES analysis.cc deeppoly_configuration.cc interval.cc network.cc parser.cc helper.cc optimizer.cc deeppoly_driver.cc) 

add_executable(${DEEPPOLY} ${SOURCES})

#SET(LIBZ3 z3)
SET(xtensor_libs blas)
SET(boost_lib boost_program_options)
SET(pthread_lib pthread)
SET(gurobi_lib gurobi_c++ gurobi91)
target_link_libraries (${DEEPPOLY} ${LIBZ3} ${xtensor_libs} ${boost_lib} ${pthread_lib} ${gurobi_lib})
