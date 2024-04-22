cmake_minimum_required(VERSION 3.0)
project(network)

add_executable(network_test Layer.cpp Network.cpp Penalty.cpp Sigmoid.cpp tests.cpp
        Rand.cpp
        usings.h
)
target_include_directories(network_test PUBLIC eigen EigenRand)