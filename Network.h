#pragma once

#include "Layer.h"
#include "LearningRate.h"
#include "Penalty.h"

namespace NeuralNets {
    class Network {
        std::vector<Layer> layers_;

        std::vector<Matrix> FwdPropagation(const Matrix &batch) const;
        void BackPropagation(const std::vector<Matrix> &layer_values, const Matrix &batch_y,
                             const Penalty &penalty, IndexType2 step, IndexType1 num_batches);
        void Era(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                 const Penalty &penalty, const LearningRate &learning_rate, Index era_num);
        IndexType2 AveragePenalty(const Matrix &data_x, const Matrix &data_y,
                                           IndexType1 batch_size, const Penalty &penalty) const;
        IndexType2 SamplePenalty(const Matrix &data_x, const Matrix &data_y, const Penalty &penalty) const;
        static IndexType3 CheckTrainProcess(IndexType1 cur_era, IndexType2 cur_penalty, IndexType2& all_penalty,
                                            IndexType2 eps, LearningRate &learning_rate,
                                            const std::string &option = "default");

    public:
        Network(const std::initializer_list<Index> &layer_dim, std::initializer_list<AF_id> af_id);
        Matrix EvaluateNet(const Matrix &batch) const;
        void Train(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size, const Penalty &penalty,
                   IndexType2 eps, LearningRate &learning_rate, IndexType1 max_era);
    };
}
