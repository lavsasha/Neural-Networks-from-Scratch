#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <functional>
#include <Eigen/Dense>

namespace NeuralNets {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Row = Eigen::RowVectorXd;
    using Index = Eigen::Index;

    inline Row GetRowOf1(Index cols) {
        return Row::Ones(cols);
    }

    inline Vector GetVecOf1(Index rows) {
        return Vector::Ones(rows);
    }
}
