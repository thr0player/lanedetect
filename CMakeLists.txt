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
SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

#set(OpenCV_DIR "/usr/local/share/OpenCV")

#find_package(Qt5Core REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
#find_package(OpenMP)

message(STATUS "show test work")
message(STATUS "------->pcl found: ${PCL_INCLUDE_DIRS}")
message(STATUS "------->opencv found: ${OpenCV_LIBRARIES}")
message(STATUS "------->opencv inc found: ${OpenCV_LIBRARIES}")


###################################
## build                         ##
###################################
include_directories(
        ${catkin_INCLUDE_DIRS}
        include
)

SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

link_directories(${PCL_LIBRARY_DIRS})

# Ring Ground Filter
add_definitions(${PCL_DEFINITIONS})

add_executable(lanedetection
        main.cpp
        )

target_include_directories(lanedetection PRIVATE
        ${PCL_INCLUDE_DIRS}
        )

target_link_libraries(lanedetection
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        )
add_dependencies(lanedetection ${catkin_EXPORTED_TARGETS})