#include "Network.h"

namespace NeuralNets {
    Network::Network(const std::initializer_list<Index> &layer_dim, std::initializer_list<AF_id> af_id) {
        assert(layer_dim.size() == af_id.size() + 1);
        layers_.reserve(layer_dim.size() - 1);
        auto dim_it = layer_dim.begin();
        for (auto af_it = af_id.begin(); af_it != af_id.end(); ++af_it, ++dim_it) {
            layers_.emplace_back(*dim_it, *(dim_it + 1), *af_it);
        }
    }

    Matrix Network::EvaluateNet(const Matrix &batch) const {
        Matrix cur_out = batch;
        for (const auto &layer: layers_) {
            cur_out = layer.Evaluate(cur_out);
        }
        return cur_out;
    }

    std::vector<Matrix> Network::FwdPropagation(const Matrix &batch) const {
        std::vector<Matrix> layer_values;
        layer_values.emplace_back(batch);
        bool is_empty = true;
        for (const auto &layer: layers_) {
            if (is_empty) {
                layer_values.emplace_back(layer.Evaluate(batch));
                is_empty = false;
            } else {
                layer_values.emplace_back(layer.Evaluate(layer_values.back()));
            }
        }
        return layer_values;
    }

    void Network::BackPropagation(const std::vector<Matrix> &layer_values, const Matrix &batch_y,
                                  const Penalty &penalty, IndexType2 step, IndexType1 num_batches) {
        Matrix cur_grad = penalty.FindInitialGradient(layer_values.back(), batch_y);
        for (IndexType1 i = layers_.size() - 1; i >= 0; --i) {
            Matrix delta_A = layers_[i].GradA(layer_values[i], cur_grad);
            Vector delta_b = layers_[i].Gradb(layer_values[i], cur_grad);
            cur_grad = layers_[i].NextGrad(layer_values[i], cur_grad);
            layers_[i].ChangeParams(delta_A, delta_b, step / num_batches);
        }
    }

    void Network::Era(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                      const Penalty &penalty, const LearningRate &learning_rate, Index era_num) {
        IndexType1 num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        for (IndexType1 i = 0; i < num_batches; ++i) {
            IndexType1 start_col = i * batch_size;
            IndexType1 end_col = (i + 1) * batch_size > data_x.cols() ? data_x.cols() : (i + 1) * batch_size;
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            BackPropagation(FwdPropagation(cur_batch_x), cur_batch_y, penalty,
                            learning_rate.GetRate(era_num), num_batches);
        }
    }

    IndexType2 Network::AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                                       IndexType1 batch_size, const Penalty &penalty) const {
        IndexType2 penalty_res = 0;
        IndexType1 num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        for (IndexType1 i = 0; i < num_batches; ++i) {
            IndexType1 start_col = i * batch_size;
            IndexType1 end_col = (i + 1) * batch_size > data_x.cols() ? data_x.cols() : (i + 1) * batch_size;
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            penalty_res += penalty.CalcPenalty(EvaluateNet(cur_batch_x), cur_batch_y);
        }
        return penalty_res / num_batches;
    }

    IndexType2 Network::SamplePenalty(const Matrix &data_x, const Matrix &data_y, const Penalty &penalty) const {
        Matrix sample_x = data_x.block(0, 0,
                                       data_x.rows(), data_x.cols() / sample_num + 1);
        Matrix sample_y = data_y.block(0, 0,
                                       data_y.rows(), data_y.cols() / sample_num + 1);
        return penalty.CalcPenalty(EvaluateNet(sample_x), sample_y);
    }

    bool Network::CheckTrainProcess(IndexType1 cur_era, IndexType2 cur_penalty, IndexType2& all_penalty, IndexType2 eps,
                                    LearningRate &learning_rate, const std::string &option) {
        if (option == "default") {
            std::cout << "On the " << cur_era << " era penalty is " << cur_penalty << '\n';
            if (cur_penalty < eps) {
                std::cout << "The required accuracy has been achieved!\n";
                return true;
            }
        } else {
            std::cout << "On the " << cur_era << " era sample_penalty is " << cur_penalty << '\n';
        }
        if (all_penalty > cur_penalty) {
            all_penalty = cur_penalty;
        } else {
            all_penalty = cur_penalty;
            std::cout << "Something's wrong, let's change the learning rate...\n";
            learning_rate.ChangeRate();
        }
        return false;
    }

    void Network::Train(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size, const Penalty &penalty,
                        IndexType2 eps, LearningRate &learning_rate, IndexType1 max_era) {
        IndexType1 check_frequency = max_era / measure > 1 ? max_era / measure : 1;
        IndexType2 sample_penalty = SamplePenalty(data_x, data_y, penalty);
        IndexType2 all_penalty = AveragePenalty(data_x, data_y, batch_size, penalty);
        std::cout << "Initial penalty: " << all_penalty << '\n';
        for (IndexType1 cur_era = 1; cur_era <= max_era; ++cur_era) {
            Era(data_x, data_y, batch_size, penalty, learning_rate, cur_era);
            if (cur_era % check_frequency == 0) {
                std::cout << "Era num: " << cur_era << '\n';
                IndexType2 cur_penalty = AveragePenalty(data_x, data_y, batch_size, penalty);
                if (CheckTrainProcess(cur_era, cur_penalty, all_penalty, eps, learning_rate)) {
                    break;
                }
                IndexType2 cur_smpl_penalty = SamplePenalty(data_x, data_y, penalty);
                CheckTrainProcess(cur_era, cur_smpl_penalty, sample_penalty, eps,
                                  learning_rate, "sample");
            }
        }
    }
}
