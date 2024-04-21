#pragma once

#include "Layer.h"

namespace NeuralNets {

    class Penalty {
        static IndexType2 Pow(IndexType2 x);
        static IndexType2 FindDist(const Vector &z, const Vector &y);

    public:

        static IndexType2 CalcPenalty(const Matrix& batch_y, const Matrix& network_out);
        static Matrix FindInitialGradient(const Matrix& batch_y, const Matrix& network_out);
    };
}
