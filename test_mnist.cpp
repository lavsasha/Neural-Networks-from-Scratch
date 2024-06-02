#include "Network.h"
#include "mnist/include/mnist/mnist_reader.hpp"

using namespace NeuralNets;
using std::vector;

namespace {
    Matrix MnistDataX(const vector<vector<unsigned char>> &data) {
        Matrix data_x(data[0].size(), data.size());
        for (IndexType i = 0; i < data.size(); ++i) {
            for (IndexType j = 0; j < data[0].size(); ++j) {
                data_x(j, i) = Scalar(data[i][j]) / 255.;
            }
        }
        return data_x;
    }

    Matrix MnistDataY(const vector<unsigned char> &data) {
        Matrix data_y(10, data.size());
        for (IndexType i = 0; i < data.size(); ++i) {
            for (IndexType j = 0; j < 10; ++j) {
                data_y(j, i) = j == data[i] ? 1 : 0;
            }
        }
        return data_y;
    }
}

int main() {
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> data =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    Network net({784, 256, 10}, {AF_id::ReLu, AF_id::Sigmoid});
    Matrix data_x = MnistDataX(data.training_images);
    Matrix data_y = MnistDataY(data.training_labels);
    Scalar init_rate = 50.0;
    LearningRate const_lr(init_rate);
    IndexType batch_size = 30;
    Scalar eps = 0.3;
    IndexType max_era = 70;
    net.Train(data_x, data_y, batch_size, Penalty::Initialize(PenaltyId::MSE),
              eps, const_lr, max_era);

    IndexType correct = 0;
    Matrix test_data_x = MnistDataX(data.test_images);
    Matrix test_data_y = MnistDataY(data.test_labels);
    for (IndexType i = 0; i < data.test_images.size(); ++i) {
        Scalar predicted_value = net.Evaluate(test_data_x.col(i)).array().maxCoeff();
        Scalar actual_value = test_data_y.col(i).array().maxCoeff();
        if (predicted_value == actual_value) {
            correct++;
        }
    }
    Scalar accuracy = Scalar(correct) / data.test_images.size() * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
