#pragma once

#include <Eigen/Dense>
#include "Layer.h"

class Penalty {
public:
    static double FindDist(const Eigen::MatrixXd &z, const Eigen::MatrixXd &y);

    static std::pair<double, std::vector<Eigen::MatrixXd>> CalcPenalty(
            const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &data, std::vector<Layer> &network);

    static Eigen::MatrixXd FindInitialRow(const std::vector<std::pair<Eigen::MatrixXd,
            Eigen::MatrixXd>> &data, std::vector<Layer> &network);
};