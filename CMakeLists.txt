cmake_minimum_required(VERSION 3.2)
project(fast_guide_filter)

find_package(OpenCV)

include_directories(${OpenCV_DIR})

add_executable(boxFilter main.cpp)
target_link_libraries(boxFilter ${OpenCV_LIBS})