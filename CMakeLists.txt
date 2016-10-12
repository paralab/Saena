cmake_minimum_required(VERSION 2.8)
project(Saena)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
        include
)


set(SOURCE_FILES
        src/main.cpp
        include/cscmatrix.h
        src/cscmatrix.cpp
        )

add_executable(Saena ${SOURCE_FILES})