#pragma once

#include "usings.h"
#include "ActivationFunction.h"
#include "Rand.h"

namespace NeuralNets {
    class Layer {

        Matrix A_;
        Vector b_;
        ActivationFunction ActivationFunction_;

    public:
        Layer(IndexType1 rows, IndexType1 columns, AF_id func);
        Matrix Evaluate(const Matrix &batch) const;
        Matrix GradA(const Matrix &batch, const Matrix &grad);
        Vector Gradb(const Matrix &batch, const Matrix &grad);
        void ChangeParams(const Matrix& delta_A, const Vector& delta_b, IndexType2 step);
        Matrix NextGrad(const Matrix &batch, const Matrix &grad) const;
    };
}
