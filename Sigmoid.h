#pragma once

#include "usings.h"

namespace NeuralNets {
    class Sigmoid {
        static IndexType2 CalcExp_0(IndexType2 x);
        static IndexType2 CalcExp_1(IndexType2 x);

    public:
        static Matrix EvaluateFunc(const Matrix &batch);
        static Matrix EvaluateDerivative(const Matrix &batch);

    };

}
