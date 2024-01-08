#pragma once

#include <cmath>
#include "Eigen/Dense"

class Sigma {
public:
    static Eigen::MatrixXd EvaluateFunc(const Eigen::MatrixXd &x);

    static Eigen::MatrixXd EvaluateDerivative(const Eigen::MatrixXd &x);
};