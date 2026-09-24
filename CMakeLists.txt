cmake_minimum_required(VERSION 3.10)
project(shape_match_lib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- 依赖 ----
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

# ---- cuda_icp（CPU 版）----
add_subdirectory(third_party/cuda_icp)

# ---- 核心库 ----
add_library(shape_match_lib STATIC
    src/shapematch.cpp
    src/line2Dup.cpp
)

target_include_directories(shape_match_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MIPP>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(shape_match_lib
    PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        cuda_icp
)

# ---- 安装规则（可选）----
install(TARGETS shape_match_lib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
