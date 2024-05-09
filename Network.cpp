#include "Network.h"

namespace NeuralNets {
    Network::Network(const std::initializer_list<Index> &layer_dim, std::initializer_list<AF_id> af_id) {
        assert(layer_dim.size() == af_id.size() + 1);
        layers_.reserve(layer_dim.size() - 1);
        auto dim_it = layer_dim.begin();
        for (auto af_it = af_id.begin(); af_it != af_id.end(); ++af_it, ++dim_it) {
            layers_.emplace_back(*(dim_it + 1), *dim_it, *af_it);
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

    void Network::BackPropagation(const std::vector<Matrix> &layer_values,
                                  const Matrix &batch_y, const Penalty &penalty, IndexType2 step) {
        Matrix cur_grad = penalty.FindInitialGradient(batch_y, layer_values.back());
        for (IndexType1 i = layers_.size() - 1; i >= 0; --i) {
            Matrix delta_A = layers_[i].GradA(layer_values[i], cur_grad);
            Vector delta_b = layers_[i].Gradb(layer_values[i], cur_grad);
            cur_grad = layers_[i].NextGrad(layer_values[i], cur_grad);
            layers_[i].ChangeParams(delta_A, delta_b, step);
        }
    }

    void Network::Era(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                      const Penalty &penalty, const LearningRate &learning_rate, Index era_num) {
        IndexType1 num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        for (IndexType1 i = 0; i < num_batches; ++i) {
            IndexType1 start_col = i * batch_size;
            IndexType1 end_col = std::min((i + 1) * batch_size, data_x.cols());
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            BackPropagation(FwdPropagation(cur_batch_x), cur_batch_y, penalty,
                            learning_rate.GetRate(era_num));
        }
    }
    IndexType2 Network::AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                                       IndexType1 batch_size, const Penalty &penalty) {
        IndexType2 penalty_res = 0;
        IndexType1 num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        for (IndexType1 i = 0; i < num_batches; ++i) {
            IndexType1 start_col = i * batch_size;
            IndexType1 end_col = std::min((i + 1) * batch_size, data_x.cols());
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            penalty_res += penalty.CalcPenalty(EvaluateNet(cur_batch_x),cur_batch_y);
        }
        return penalty_res / num_batches;
    }

    void Network::Train(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size, const Penalty &penalty,
                        IndexType2 eps, LearningRate &learning_rate, IndexType1 max_era) {
        IndexType1 check_frequency = max_era / measure;
        IndexType2 sample_penalty = max_number;
        IndexType2 all_penalty = max_number;
        for (IndexType1 cur_era = 0; cur_era < max_era; ++cur_era) {
            std::cout << "Era num: " << cur_era << '\n';
            Era(data_x, data_y, batch_size, penalty, learning_rate, cur_era);
            if (cur_era % check_frequency == 0) {
                IndexType2 cur_penalty = AveragePenalty(data_x, data_y, batch_size, penalty);
                std::cout << "On the " << cur_era << "th era penalty is " << cur_penalty << '\n';
                if (cur_penalty < eps) {
                    std::cout << "The required accuracy has been achieved!\n";
                    break;
                } else if (all_penalty > cur_penalty) {
                    all_penalty = cur_penalty;
                } else {
                    std::cout << "Something's wrong, let's change the learning rate...\n";
                    learning_rate.ChangeRate();
                }
                Matrix sample_x = data_x.block(0, 0,
                                                  data_x.rows(), data_x.cols() / sample_num);
                Matrix sample_y = data_y.block(0, 0,
                                               data_y.rows(), data_y.cols() / sample_num);
                IndexType2 cur_smpl_penalty = penalty.CalcPenalty(EvaluateNet(sample_x),sample_y);
                std::cout << "On the " << cur_era << "th era sample_penalty is " << cur_smpl_penalty << '\n';
                if (sample_penalty > cur_smpl_penalty) {
                    sample_penalty = cur_smpl_penalty;
                } else {
                    std::cout << "Something's wrong, let's change the learning rate...\n";
                    learning_rate.ChangeRate();
                }
            }
        }
    }
}
