cmake_minimum_required(VERSION 2.8.3)
project(rrt_star)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  grid_map_ros
  grid_map_msgs
  nav_msgs
  geometry_msgs
)

find_package(Eigen3 REQUIRED)

# Explicitly set OMPL path if needed
set(CMAKE_PREFIX_PATH "/opt/ros/noetic/share/ompl/cmake/" ${CMAKE_PREFIX_PATH})

# Find OMPL
find_package(ompl REQUIRED)

# Ensure OMPL is found
if(OMPL_FOUND)
  message(STATUS "Found OMPL version ${OMPL_VERSION}")
else()
  message(FATAL_ERROR "OMPL not found. Please install OMPL or specify the correct path.")
endif()

## catkin specific configuration ##
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp grid_map_ros grid_map_msgs nav_msgs geometry_msgs
  DEPENDS EIGEN3 OMPL  
)

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
)

## Declare a cpp executable
add_executable(rrt_node 
  src/main.cpp  
  src/rrt_node.cpp
)

## Add target dependencies
add_dependencies(rrt_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(rrt_node 
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${OMPL_LIBRARIES} 
)
