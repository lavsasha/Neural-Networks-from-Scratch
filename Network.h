#pragma once

#include "Layer.h"
#include "LearningRate.h"
#include "Penalty.h"

namespace NeuralNets {
    constexpr int measure = 14;
    constexpr int sample_num = 5;
    using IndexType = long long;
    enum class Option {
        standard,
        sample
    };

    class Network {
        std::vector<Layer> layers_;

        std::vector<Matrix> FwdPropagation(const Matrix &batch) const;
        void BackPropagation(const std::vector<Matrix> &layer_values, const Matrix &batch_y,
                             const Penalty &penalty, Scalar step, IndexType num_batches);
        void TrainEra(const Matrix &data_x, const Matrix &data_y, IndexType batch_size,
                 const Penalty &penalty, const LearningRate &learning_rate, Index era_num);
        Scalar AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                              IndexType batch_size, const Penalty &penalty) const;
        Scalar SamplePenalty(const Matrix &data_x, const Matrix &data_y, const Penalty &penalty) const;
        static bool IsTargetPenaltyAchieved(IndexType cur_era, Scalar cur_penalty, Scalar& all_penalty,
                                            Scalar eps, LearningRate &learning_rate, Option option = Option::standard);

    public:
        Network(const std::initializer_list<Index> &layer_dim, std::initializer_list<AF_id> af_id);
        Matrix Evaluate(const Matrix &batch) const;
        void Train(const Matrix &data_x, const Matrix &data_y, IndexType batch_size, const Penalty &penalty,
                   Scalar eps, LearningRate &learning_rate, IndexType max_era);
    };
}
