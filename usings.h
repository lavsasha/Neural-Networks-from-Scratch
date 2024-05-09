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
    using IndexType1 = long long;
    using IndexType2 = double;
    using Index = Eigen::Index;
    using FunctionType = std::function<Matrix(const Matrix &)>;
    using PenaltyFuncType = std::function<IndexType2(const Vector &, const Vector &)>;
    using GradientFuncType = std::function<Matrix(const Matrix &, const Matrix &)>;
    using Step = std::function<IndexType2(Index)>;

    enum {
        measure = 100,
        max_number = 1000000000,
        sample_num = 5
    };

    static Row GetRowOf1(IndexType1 cols) {
        return Row::Ones(cols);
    }

    static Vector GetVecOf1(IndexType1 rows) {
        return Vector::Ones(rows);
    }

}
