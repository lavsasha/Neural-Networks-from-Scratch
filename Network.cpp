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

    Matrix Network::Evaluate(const Matrix &batch) const {
        Matrix cur_out = batch;
        for (const auto &layer: layers_) {
            cur_out = layer.Evaluate(cur_out);
        }
        return cur_out;
    }

    std::vector<Matrix> Network::FwdPropagation(const Matrix &batch) const {
        std::vector<Matrix> layer_values;
        layer_values.push_back(batch);
        for (const auto &layer: layers_) {
            layer_values.push_back(layer.Evaluate(layer_values.back()));
        }
        return layer_values;
    }

    void Network::BackPropagation(const std::vector<Matrix> &layer_values, const Matrix &batch_y,
                                  const Penalty &penalty, Scalar step, IndexType num_batches) {
        Matrix cur_grad = penalty.FindInitialGradient(layer_values.back(), batch_y);
        for (IndexType i = layers_.size() - 1; i >= 0; --i) {
            Matrix delta_A = layers_[i].GradA(layer_values[i], cur_grad);
            Vector delta_b = layers_[i].Gradb(layer_values[i], cur_grad);
            cur_grad = layers_[i].NextGrad(layer_values[i], cur_grad);
            layers_[i].ChangeParams(delta_A, delta_b, step / num_batches);
        }
    }

    void Network::TrainEra(const Matrix &data_x, const Matrix &data_y, IndexType batch_size,
                           const Penalty &penalty, const LearningRate &learning_rate, Index era_num) {
        IndexType num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        for (IndexType i = 0; i < num_batches; ++i) {
            IndexType start_col = i * batch_size;
            IndexType end_col = (i + 1) * batch_size > data_x.cols() ? data_x.cols() : (i + 1) * batch_size;
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            BackPropagation(FwdPropagation(cur_batch_x), cur_batch_y, penalty,
                            learning_rate.GetRate(era_num), num_batches);
        }
    }

    Scalar Network::AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                                   IndexType batch_size, const Penalty &penalty) const {
        Scalar penalty_res = 0;
        IndexType num_batches = (data_x.cols() + batch_size - 1) / batch_size;
        for (IndexType i = 0; i < num_batches; ++i) {
            IndexType start_col = i * batch_size;
            IndexType end_col = (i + 1) * batch_size > data_x.cols() ? data_x.cols() : (i + 1) * batch_size;
            Matrix cur_batch_x = data_x.block(0, start_col,
                                              data_x.rows(), end_col - start_col);
            Matrix cur_batch_y = data_y.block(0, start_col,
                                              data_y.rows(), end_col - start_col);
            penalty_res += penalty.CalcPenalty(Evaluate(cur_batch_x), cur_batch_y);
        }
        return penalty_res / num_batches;
    }

    Scalar Network::SamplePenalty(const Matrix &data_x, const Matrix &data_y, const Penalty &penalty) const {
        Matrix sample_x = data_x.block(0, 0,
                                       data_x.rows(), data_x.cols() / sample_num + 1);
        Matrix sample_y = data_y.block(0, 0,
                                       data_y.rows(), data_y.cols() / sample_num + 1);
        return penalty.CalcPenalty(Evaluate(sample_x), sample_y);
    }

    bool Network::IsTargetPenaltyAchieved(IndexType cur_era, Scalar cur_penalty, Scalar &all_penalty, Scalar eps,
                                          LearningRate &learning_rate, const Option option) {
        if (option == Option::standard) {
            std::cout << "On the " << cur_era << " era penalty is " << cur_penalty << '\n';
            if (cur_penalty < eps) {
                std::cout << "The required penalty value has been achieved!\n";
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

    void Network::Train(const Matrix &data_x, const Matrix &data_y, IndexType batch_size, const Penalty &penalty,
                        Scalar eps, LearningRate &learning_rate, IndexType max_era) {
        IndexType check_frequency = max_era / measure > 1 ? max_era / measure : 1;
        Scalar sample_penalty = SamplePenalty(data_x, data_y, penalty);
        Scalar all_penalty = AveragePenalty(data_x, data_y, batch_size, penalty);
        std::cout << "Initial penalty: " << all_penalty << '\n';
        for (IndexType cur_era = 1; cur_era <= max_era; ++cur_era) {
            TrainEra(data_x, data_y, batch_size, penalty, learning_rate, cur_era);
            if (cur_era % check_frequency == 0) {
                std::cout << "Era num: " << cur_era << '\n';
                Scalar cur_penalty = AveragePenalty(data_x, data_y, batch_size, penalty);
                if (IsTargetPenaltyAchieved(cur_era, cur_penalty, all_penalty, eps, learning_rate)) {
                    break;
                }
                Scalar cur_smpl_penalty = SamplePenalty(data_x, data_y, penalty);
                IsTargetPenaltyAchieved(cur_era, cur_smpl_penalty, sample_penalty, eps,
                                        learning_rate, Option::sample);
            }
        }
    }
}
