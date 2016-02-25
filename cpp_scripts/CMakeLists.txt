cmake_minimum_required(VERSION 3.3)
project(Axa_Data_Challenge)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES filter_france.cpp)
add_executable(Axa_Data_Challenge ${SOURCE_FILES})