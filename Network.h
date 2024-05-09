#pragma once

#include "Penalty.h"
#include "Layer.h"
#include "LearningRate.h"

namespace NeuralNets {
    class Network {
        std::vector<Layer> layers_;

        std::vector<Matrix> FwdPropagation(const Matrix &batch) const;
        void BackPropagation(const std::vector<Matrix> &layer_values, const Matrix &batch_y,
                             const Penalty &penalty, IndexType2 step);
        void Era(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                 const Penalty &penalty, const LearningRate &learning_rate, Index era_num);
        IndexType2 AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                                           IndexType1 batch_size, const Penalty &penalty);

    public:
        Network(const std::initializer_list<Index> &layer_dim, std::initializer_list<AF_id> af_id);
        Matrix EvaluateNet(const Matrix &batch) const;
        void Train(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size, const Penalty &penalty,
                   IndexType2 eps, LearningRate &learning_rate, IndexType1 max_era);
    };
}
