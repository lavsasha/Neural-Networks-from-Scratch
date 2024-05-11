#pragma once
#include "usings.h"

namespace NeuralNets {
    enum class LRSchedule {
        Constant, Linear, Exponential
    };

    class LearningRate {
        LRSchedule schedule_;
        IndexType2 init_rate_;
        IndexType2 reduct_coef_;

    public:
        LearningRate(IndexType2 init_rate);
        LearningRate(IndexType2 init_rate, IndexType2 reduct_coef);
        LearningRate(IndexType2 init_rate, LRSchedule schedule);

        IndexType2 GetRate(Index epoch) const;
        void ChangeRate();
    };
}
