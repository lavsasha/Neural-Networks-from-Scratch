cmake_minimum_required(VERSION 3.22)
project(network_project)

set(CMAKE_CXX_STANDARD 17)

find_package(MNIST PATHS mnist/)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

add_library(NeuralNets_lib
        ActivationFunction.cpp
        Rand.cpp
        Layer.cpp
        Penalty.cpp
        Network.cpp
        LearningRate.cpp)
target_include_directories(NeuralNets_lib PUBLIC eigen EigenRand ${MNIST_INCLUDE_DIR})

add_executable(network_test main.cpp tests.cpp)
target_include_directories(
        network_test PUBLIC ${PROJECT_SOURCE_DIR})
add_executable(test_mnist test_mnist.cpp)
target_include_directories(
        test_mnist PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(network_test PUBLIC NeuralNets_lib)
target_link_libraries(test_mnist PUBLIC NeuralNets_lib)

target_compile_definitions(test_mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")