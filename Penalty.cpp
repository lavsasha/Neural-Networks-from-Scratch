#include "Penalty.h"
#include "Network.h"

double Penalty::FindDist(const Eigen::MatrixXd &z, const Eigen::MatrixXd &y) {
    assert(z.rows() == y.rows());
    assert(z.cols() == y.cols());
    double res = 0;
    for (int i = 0; i < z.size(); ++i) {
        res += pow(z(i) - y(i), 2);
    }
    return res;
}

std::pair<double, std::vector<Eigen::MatrixXd>>
Penalty::CalcPenalty(const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &data,
                     std::vector<Layer> &network) {
    Eigen::MatrixXd x(data[0].first.rows(), data.size());
    for (int i = 0; i < data.size(); ++i) {
        x.col(i) = data[i].first;
    }
    double res = 0;
    Network net(network);
    std::vector<Eigen::MatrixXd> z = net.CalcNetworkValue(x);
    for (int i = 0; i < z.back().cols(); ++i) {
        res += FindDist(z.back().col(i), data[i].second);
    }
    res /= static_cast<double>(data.size());
    return {res, z};
}

Eigen::MatrixXd Penalty::FindInitialRow(const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &data,
                                        std::vector<Layer> &network) {
    Eigen::MatrixXd u(static_cast<int64_t>(data.size()), data[0].second.rows());
    Eigen::MatrixXd x(data[0].first.rows(), data.size());
    for (int i = 0; i < data.size(); ++i) {
        x.col(i) = data[i].first;
    }
    Network net(network);
    Eigen::MatrixXd z = net.CalcNetworkValue(x).back();
    assert(z.rows() == data[0].second.rows());
    for (int i = 0; i < data.size(); ++i) {
        Eigen::MatrixXd temp_row(1, z.rows());
        for (int j = 0; j < z.rows(); ++j) {
            temp_row(j) = 2 * (z(j, i) - data[i].second(j));
        }
        u.row(i) = temp_row;
    }
    return u;
}