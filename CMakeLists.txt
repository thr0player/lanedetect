cmake_minimum_required(VERSION 2.8.3)
project(lanedetection)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        geometry_msgs
        pcl_ros
        pcl_conversions
        tf
        message_filters
        )

catkin_package(CATKIN_DEPENDS
        roscpp
        std_msgs
        sensor_msgs
        geometry_msgs
        pcl_ros
        pcl_conversions
        tf
        message_filters
        )

add_definitions(-std=c++11)
SET(CMAKE_CXX_FLAGS "-O3 -g -Wall ${CMAKE_CXX_FLAGS}")

#set(OpenCV_DIR "/usr/local/share/OpenCV")

#find_package(Qt5Core REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(OpenMP)

message(STATUS "show test work")
message(STATUS "------->pcl found: ${PCL_INCLUDE_DIRS}")
message(STATUS "------->opencv found: ${OpenCV_VERSION}")
message(STATUS "------->opencv found: ${OpenCV_LIBRARIES}")
message(STATUS "------->opencv inc found: ${OpenCV_LIBRARIES}")


###################################
## build                         ##
###################################
include_directories(
        ${catkin_INCLUDE_DIRS}
        include
        include/utils
        src
)

link_directories(${PCL_LIBRARY_DIRS})

# Ring Ground Filter
add_definitions(${PCL_DEFINITIONS})

add_executable(lanedetection
        main.cpp
        src/aggregation.cc
        include/cca.h
  src/SVMClassifier.cpp include/SVMClassifier.h)

target_include_directories(lanedetection PRIVATE
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        )

target_link_libraries(lanedetection
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        )
add_dependencies(lanedetection ${catkin_EXPORTED_TARGETS} )

#### detect test
add_executable(detectdemo
  main_detect.cpp
  include/cca.h
  src/SVMClassifier.cpp include/SVMClassifier.h
  )

target_include_directories(detectdemo PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  )

target_link_libraries(detectdemo
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  )
add_dependencies(detectdemo ${catkin_EXPORTED_TARGETS} )



#
## Ray Ground Filter
#add_library(ray_ground_filter_lib SHARED
#        ray_ground_filter/ray_ground_filter.cpp)
#
#if (OPENMP_FOUND)
#    set_target_properties(ray_ground_filter_lib PROPERTIES
#            COMPILE_FLAGS ${OpenMP_CXX_FLAGS}
#            LINK_FLAGS ${OpenMP_CXX_FLAGS}
#            )
#endif ()
#
#target_include_directories(ray_ground_filter_lib PRIVATE
#        ${OpenCV_INCLUDE_DIRS}
#        ${PCL_INCLUDE_DIRS}
#        ray_ground_filter/include
#        )
#
#target_link_libraries(ray_ground_filter_lib
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        ${OpenCV_LIBRARIES}
#        )
#
#add_executable(ray_ground_filter
#        ray_ground_filter/ray_ground_filter_main.cpp
#        )
#
#target_include_directories(ray_ground_filter PRIVATE
#        ray_ground_filter/include)
#
#target_link_libraries(ray_ground_filter
#        ray_ground_filter_lib)
#
#add_dependencies(ray_ground_filter ${catkin_EXPORTED_TARGETS})
