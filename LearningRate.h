#pragma once
#include "EigenProxy.h"

namespace NeuralNets {
    using Scalar = double;

    enum class LRSchedule {
        Constant, Linear, Exponential
    };

    class LearningRate {
        LRSchedule schedule_;
        Scalar init_rate_;
        Scalar reduct_coef_;

    public:
        LearningRate(Scalar init_rate);
        LearningRate(Scalar init_rate, Scalar reduct_coef);
        LearningRate(Scalar init_rate, LRSchedule schedule);

        Scalar GetRate(Index epoch) const;
        void ChangeRate();
    };
}
