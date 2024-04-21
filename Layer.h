#pragma once

#include "Sigmoid.h"
#include "Rand.h"

namespace NeuralNets {
    class Layer {
        Matrix A_;
        Vector b_;

        static Row GetRowOf1(IndexType1 cols);
        static Vector GetVecOf1(IndexType1 rows);

        friend class Network;

    public:

        Layer(IndexType1 rows, IndexType1 columns);
        Matrix Evaluate(const Matrix &batch);
        Matrix GradA(const Matrix &batch, const Matrix &u);
        Matrix Gradb(const Matrix &batch, const Matrix &u);
        Matrix NextGrad(const Matrix &batch, const Matrix &u);
    };
}
