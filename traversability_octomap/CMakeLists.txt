cmake_minimum_required(VERSION 2.8.3)
project(traversability_octomap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
endif()

find_package(OpenCV REQUIRED)

## Enable OpenMP
add_definitions(-DOPENMP)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    visualization_msgs
    nav_msgs
    pcl_ros
    roscpp
    tf_conversions
    # octomap_utils
    tf2
    tf2_ros
    tf2_geometry_msgs
    grid_map_ros
    grid_map_msgs
)

catkin_package(
  INCLUDE_DIRS include
#  CATKIN_DEPENDS geometry_msgs pcl_ros roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  include/common
  include/mapping
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(bgkoctomap
#   src/${PROJECT_NAME}/bgkoctomap.cpp
# )

# Common source files
set(COMMON_SOURCE_FILES src/common/point3f)

# Files for SEMANTIC BGK source
set(MAPPING_SOURCE_FILES
    src/mapping/bkiblock.cpp
    src/mapping/bkioctomap.cpp
    src/mapping/bkioctree_node.cpp
    src/mapping/bkioctree.cpp)

add_executable(octomap_node
  ${MAPPING_SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  src/main.cpp
  src/octomap_node.cpp
)

## Specify libraries to link a library or executable target against

target_link_libraries(octomap_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
