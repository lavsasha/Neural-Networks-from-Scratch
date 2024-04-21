#pragma once

#include "Penalty.h"

namespace NeuralNets {
    class Network {
        std::vector<Layer> layers_;

        std::vector<Matrix> FwdPropagation(const Matrix &batch);
        Matrix EvaluateNet(const Matrix &batch);
        void SinglePassage(const Matrix &batch_x, const Matrix &batch_y, IndexType2 pace);
        void Era(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                 IndexType1 num_batches, IndexType2 pace);
        IndexType2 AveragePenalty(const Matrix &data_x, const Matrix &data_y, IndexType1 batch_size,
                                         IndexType1 num_batches);

        friend class Layer;

    public:
        Network(const std::initializer_list<Index> &layer_dim);
        void Train(const Matrix &data_x, const Matrix &data_y, const Matrix &sample_x, const Matrix &sample_y,
                   IndexType1 batch_size, IndexType2 eps, IndexType1 max_era, IndexType2 step);
    };
}
