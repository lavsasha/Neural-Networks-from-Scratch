#include <algorithm>
#include "Layer.h"
#include "Penalty.h"
class Network {
public:
    std::vector<Layer> layers;

    Network(size_t num_of_layers, const std::vector<std::pair<int64_t, int64_t>>& params);

    explicit Network(const std::vector<Layer>& net);

    std::vector<Eigen::MatrixXd> CalcNetworkValue(const Eigen::MatrixXd &x);

    void Train(const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &data, size_t batch_size,
                      double eps, size_t max_step, int coef);
};