#include "Network.h"

Network::Network(size_t num_of_layers, const std::vector<std::pair<int64_t, int64_t>> &params) {
    for (int i = 0; i < num_of_layers; ++i) {
        Layer new_layer(params[i].first, params[i].second);
        layers.push_back(new_layer);
    }
}

Network::Network(const std::vector<Layer> &net) {
    layers = net;
}

std::vector<Eigen::MatrixXd> Network::CalcNetworkValue(const Eigen::MatrixXd &x) {
    std::vector<Eigen::MatrixXd> val;
    Eigen::MatrixXd temp(x.rows(), x.cols());
    temp = x;
    val.push_back(temp);
    for (auto &l: layers) {
        temp = l.CalcLayerValue(temp);
        val.push_back(temp);
    }
    return val;
}

void Network::Train(const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &data, size_t batch_size,
                    double eps, size_t max_step, const int coef) {
    size_t step = 0;
    size_t num_of_batches = data.size() / batch_size;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> batch;
    while (true) {
        if ((step + 1) * batch_size <= num_of_batches * batch_size) {
            batch.insert(batch.end(), data.begin() + static_cast<int64_t>(step * batch_size),
                         data.begin() + static_cast<int64_t>((step + 1) * batch_size));
        } else {
            batch.insert(batch.end(), data.begin() + static_cast<int64_t>(batch_size * num_of_batches), data.end());
        }
        if (batch.empty()) {
            break;
        }
        size_t batch_step = 0;
        auto penalty = Penalty::CalcPenalty(batch, layers);
        auto u_rows = Penalty::FindInitialRow(batch, layers);
        while (penalty.first > eps) {
            for (int i = static_cast<int>(layers.size() - 1); i >= 0; --i) {
                auto delta_A = layers[i].GradA(penalty.second[i], u_rows);
                auto delta_b = layers[i].Gradb(penalty.second[i], u_rows);
                u_rows = layers[i].NextGrad(penalty.second[i], u_rows);
                layers[i].A -= coef * delta_A;
                layers[i].b -= coef * delta_b;
            }
            penalty = Penalty::CalcPenalty(batch, layers);
            u_rows = Penalty::FindInitialRow(batch, layers);
            ++batch_step;
            if (batch_step > max_step) {
                break;
            }
        }
        batch.clear();
        ++step;
        if (step > num_of_batches) {
            break;
        }
    }
}