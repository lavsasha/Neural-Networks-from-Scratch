cmake_minimum_required(VERSION 3.24)
project(network_test)
set(CMAKE_CXX_STANDARD 17)

add_executable(
        network_test
        Rand.cpp
        Layer.cpp
        ActivationFunction.cpp
        Penalty.cpp
        Network.cpp
        LearningRate.cpp
        tests.cpp
)
target_include_directories(network_test PUBLIC eigen EigenRand)