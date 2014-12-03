cmake_minimum_required(VERSION 2.8.3)
project(objectify)

SET(ALIAS objectify)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs  
  roscpp
  sensor_msgs
  stage_simulation
  message_filters
  dynamic_reconfigure 
)


## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)

#find_package(libpointmatcher 1.1.0 REQUIRED)
#include_directories("${libpointmatcher_INCLUDE_DIRS}")
#message(STATUS "Using libpointmatcher version ${libpointmatcher_VERSION}")

SET(CMAKE_BUILD_TYPE Debug)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   nav_msgs#   sensor_msgs
# )

#add dynamic reconfigure api
generate_dynamic_reconfigure_options(
  cfg/objectify_param.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES exercise1
#  CATKIN_DEPENDS geometry_msgs nav_msgs opencv2 roscpp sensor_msgs stage_simulation
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
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(exercise1
#   src/${PROJECT_NAME}/exercise1.cpp
# )

file(GLOB UTILS_HEADERS "include/utils/*.h")
file(GLOB DP_HEADERS "include/data_processing/*.h")
file(GLOB VIS_HEADERS "include/visual/*.h")

add_library(${ALIAS}_utils_lib 
  src/utils/base_classes.cpp
  src/utils/math.cpp 
  src/utils/iterators.cpp
  src/utils/kalman.cpp
  src/utils/kalman_base.cpp
  src/utils/kalman_predict.cpp
  src/utils/kalman_update.cpp
  src/utils/convolution.cpp
  ${UTILS_HEADERS}
)
target_link_libraries(${ALIAS}_utils_lib
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)

add_library(${ALIAS}_visual_lib 
  src/visual/plot.cpp 
  src/visual/plot_data.cpp
  src/visual/plot_convolution.cpp
  ${VIS_HEADERS}
)
target_link_libraries(${ALIAS}_visual_lib
  ${ALIAS}_utils_lib
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)

add_library(${ALIAS}_data_processing_lib 
  src/data_processing/dp.cpp 
  src/data_processing/correlation.cpp
  src/data_processing/preprocessing.cpp 
  src/data_processing/segmentation
  ${DP_HEADERS}
)
target_link_libraries(${ALIAS}_data_processing_lib
  ${ALIAS}_utils_lib
  ${ALIAS}_data_processing_lib
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)




add_executable(${ALIAS} 
  src/data_processing/dp_node.cpp
  ${DP_HEADERS}
)
target_link_libraries(${ALIAS}
  ${ALIAS}_data_processing_lib
  ${ALIAS}_utils_lib
  ${ALIAS}_visual_lib
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)


## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(${ALIAS} ${ALIAS}_data_processing_lib ${ALIAS}_utils_lib ${ALIAS}_visual_lib)

set_target_properties(${ALIAS}_data_processing_lib PROPERTIES COMPILE_FLAGS "-std=c++0x")


## Specify libraries to link a library or executable target against




#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS exercise1 exercise1_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_exercise1.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
