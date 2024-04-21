#include "Network.h"

namespace NeuralNets {
    Network::Network(const std::initializer_list<Index> &layer_dim) {
        layers_.reserve(layer_dim.size());
        for (auto it = layer_dim.begin(); (it + 1) != layer_dim.end(); ++it) {
            layers_.emplace_back(*(it + 1), *it);
        }
    }

    std::vector<Matrix> Network::FwdPropagation(const Matrix &batch) {
        std::vector<Matrix> layer_values;
        bool is_empty = true;
        for (Layer &layer: layers_) {
            if (is_empty) {
                layer_values.push_back(layer.Evaluate(batch));
                is_empty = false;
            } else {
                layer_values.push_back(layer.Evaluate(layer_values.back()));
            }
        }
        return layer_values;
    }

    Matrix Network::EvaluateNet(const Matrix &batch) {
        return FwdPropagation(batch).back();
    }

    void Network::SinglePassage(const NeuralNets::Matrix &batch_x, const Matrix &batch_y, IndexType2 pace) {
        auto layer_values = FwdPropagation(batch_x);
        Matrix u_rows = Penalty::FindInitialGradient(batch_y, layer_values.back());
        for (IndexType1 i = layers_.size() - 1; i >= 0; --i) {
            Matrix delta_A = layers_[i].GradA(layer_values[i], u_rows);
            Matrix delta_b = layers_[i].Gradb(layer_values[i], u_rows);
            u_rows = layers_[i].NextGrad(layer_values[i], u_rows);
            layers_[i].A_ -= pace * delta_A;
            layers_[i].b_ -= pace * delta_b;
        }
    }

    void Network::Era(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                      IndexType1 num_batches, IndexType2 pace) {
        for (IndexType1 i = 0; i < num_batches; ++i) {
            IndexType1 start_col = i * batch_size;
            IndexType1 end_col = std::min((i + 1) * batch_size, data_x.cols());
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            SinglePassage(cur_batch_x, cur_batch_y, pace);
        }
    }

    IndexType2 Network::AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                                       IndexType1 batch_size, IndexType1 num_batches) {
        IndexType2 penalty = 0;
        for (IndexType1 i = 0; i < num_batches; ++i) {
            IndexType1 start_col = i * batch_size;
            IndexType1 end_col = std::min((i + 1) * batch_size, data_x.cols());
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            penalty += Penalty::CalcPenalty(cur_batch_y, EvaluateNet(cur_batch_x));
        }
        return penalty / num_batches;
    }

    void Network::Train(const Matrix &data_x, const Matrix &data_y, const Matrix &sample_x, const Matrix &sample_y,
                        IndexType1 batch_size, IndexType2 eps, IndexType1 max_era, IndexType2 pace) {
        IndexType1 check_frequency = max_era / measure;
        IndexType1 num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        IndexType2 sample_penalty = max_number;
        IndexType2 all_penalty = max_number;
        for (IndexType1 cur_era = 0; cur_era < max_era; ++cur_era) {
            Era(data_x, data_y, batch_size, num_batches, pace);
            if (cur_era % check_frequency == 0) {
                IndexType2 cur_penalty = AveragePenalty(data_x, data_y, batch_size, num_batches);
                if (cur_penalty < eps) {
                    break;
                } else if (all_penalty > cur_penalty) {
                    all_penalty = cur_penalty;
                } else {
                    //CHANGE TRAIN ALGORITHM
                }
                IndexType2 cur_smpl_penalty = Penalty::CalcPenalty(sample_y, EvaluateNet(sample_x));
                if (sample_penalty > cur_smpl_penalty) {
                    sample_penalty = cur_smpl_penalty;
                } else {
                    //CHANGE TRAIN ALGORITHM
                }
            }
        }
    }
}
