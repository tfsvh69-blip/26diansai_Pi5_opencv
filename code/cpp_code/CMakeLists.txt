cmake_minimum_required(VERSION 3.10)
project(undistort_view CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)  # -O2/-O3，性能敏感项目默认发布优化
endif()

find_package(OpenCV REQUIRED)

add_executable(undistort_view undistort_view.cpp)
target_link_libraries(undistort_view ${OpenCV_LIBS})
target_include_directories(undistort_view PRIVATE ${OpenCV_INCLUDE_DIRS})
