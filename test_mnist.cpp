#include "Network.h"
#include "mnist/include/mnist/mnist_reader.hpp"

using namespace NeuralNets;
using std::vector;

Matrix MnistDataX(const vector<vector<unsigned char>> &data) {
    Matrix data_x(data[0].size(), data.size());
    for (IndexType1 i = 0; i < data.size(); ++i) {
        for (IndexType1 j = 0; j < data[0].size(); ++j) {
            data_x(j, i) = IndexType2 (data[i][j]) / 255.;
        }
    }
    return data_x;
}

Matrix MnistDataY(const vector<unsigned char> &data) {
    Matrix data_y(10, data.size());
    for (IndexType1 i = 0; i < data.size(); ++i) {
        for (IndexType1 j = 0; j < 10; ++j) {
            data_y(j, i) = j == data[i] ? 1 : 0;
        }
    }
    return data_y;
}

int main() {
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> data =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    Network net({784, 256, 10}, {AF_id::ReLu, AF_id::Sigmoid});

    Matrix data_x = MnistDataX(data.training_images);
    Matrix data_y = MnistDataY(data.training_labels);
    LearningRate lr(50.0);
    net.Train(data_x, data_y, 30, Penalty::Initialize(PenaltyId::MSE),
              0.3, lr, 70);
}