#include "Sigma.h"

Eigen::MatrixXd Sigma::EvaluateFunc(const Eigen::MatrixXd &x) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            res(i, j) = 1 / (1 + exp(-x(i, j)));
        }
    }
    return res;
}

Eigen::MatrixXd Sigma::EvaluateDerivative(const Eigen::MatrixXd &x) {
    Eigen::MatrixXd res(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            res(i, j) = exp(x(i, j)) / pow(exp(x(i, j)) + 1, 2);
        }
    }
    return res;
}