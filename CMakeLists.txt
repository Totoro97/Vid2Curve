cmake_minimum_required(VERSION 3.5)
project(ARAP CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# ceres-solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# OpenCV
find_package(Ade QUIET) # Work around linking issue caused by OpenCV4's cmake configuration, at least with the MacPorts release
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP)
message(STATUS "OpenMP flags = ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
include_directories(${OpenMP_C_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(ALL_LIBS ${OpenMP_libomp_LIBRARY} ${OpenCV_LIBS} ${CERES_LIBRARIES})

# Pangolin
if(DEFINED USE_GUI)
  add_definitions(-DUSE_GUI)
  find_package(Pangolin REQUIRED)
  include_directories(${Pangolin_INCLUDE_DIRS})
  set(ALL_LIBS ${ALL_LIBS} ${Pangolin_LIBRARIES})
endif()

file(GLOB LSQR_FILES src/Utils/LSQR/*)
file(GLOB OF_DIS_FILES src/Utils/FastOpticalFlow/*)
file(GLOB CORE_FILES src/Core/*)
file(GLOB UTILS_FILES src/Utils/*)
set(SOURCE_FILES ${OF_DIS_FILES} ${LSQR_FILES} ${CORE_FILES} ${UTILS_FILES})

add_executable(Display ${SOURCE_FILES} src/DisplayMain.cpp src/Reconstructor.cpp)
set_target_properties (Display PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET Display APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1") # use grey-valued image
target_link_libraries(Display ${ALL_LIBS})

# Test programs.
# include(AddTests.cmake)
