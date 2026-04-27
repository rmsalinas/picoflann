cmake_minimum_required(VERSION 3.10)
project(picoflann_performance)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(OpenCV REQUIRED)

add_executable(testperformance testperformance.cpp)

target_include_directories(testperformance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(testperformance ${OpenCV_LIBS})

target_compile_options(testperformance PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

add_executable(test_accuracy test_accuracy.cpp)

target_include_directories(test_accuracy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_accuracy ${OpenCV_LIBS})

target_compile_options(test_accuracy PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)
