cmake_minimum_required(VERSION 3.5)
project(shape_match_lib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ===== 平台编译选项 =====
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    SET(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
ELSE()
    SET(PLATFORM_COMPILE_FLAGS "-march=native")
ENDIF()
SET(COMMON_COMPILE_FLAGS "-Wall -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS}")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ===== 依赖 =====
find_package(OpenCV 3 REQUIRED)
find_package(OpenMP REQUIRED)

# ===== 核心库 =====
add_library(shape_match_lib STATIC
    src/shapematch.cpp
    src/line2Dup.cpp
)

target_include_directories(shape_match_lib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/MIPP>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/MIPP>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(shape_match_lib
    PUBLIC
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)

# ===== 示例 =====
add_executable(demo sample/demo.cpp)
target_link_libraries(demo PRIVATE shape_match_lib)

# ===== 安装规则 =====
install(TARGETS shape_match_lib
        EXPORT shape_match_libTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT shape_match_libTargets
        FILE shape_match_libConfig.cmake
        NAMESPACE shape_match_lib::
        DESTINATION lib/cmake/shape_match_lib)
