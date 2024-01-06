#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <Sigma.h>
#include <Eigen/Dense>

class Layer {
public:
    Eigen::MatrixXd A;
    Eigen::MatrixXd b;

    Layer(int64_t m, int64_t n);

    Eigen::MatrixXd CalcLayerValue(const Eigen::MatrixXd &x);

    Eigen::MatrixXd GradA(const Eigen::MatrixXd &x, const Eigen::MatrixXd &u);

    Eigen::MatrixXd Gradb(const Eigen::MatrixXd &x, const Eigen::MatrixXd &u);

    Eigen::MatrixXd NextGrad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u);
};