#pragma once

#include "usings.h"

namespace NeuralNets {

    enum class AF_id {
        Sigmoid, ReLu, Tanh
    };

    struct AFdatabase {

        template<AF_id>
        static IndexType2 Evaluate_0(IndexType2);

        template<AF_id>
        static IndexType2 Evaluate_1(IndexType2);

        template<AF_id>
        static Matrix Evaluate_0_mat(const Matrix &);

        template<AF_id>
        static Matrix Evaluate_1_mat(const Matrix &);


        template<> inline IndexType2 Evaluate_0<AF_id::Sigmoid>(IndexType2 x) {
            IndexType2 result = 1 / (1 + exp(-x));
            assert(std::isfinite(result));
            return result;
        }

        template<> inline IndexType2 Evaluate_1<AF_id::Sigmoid>(IndexType2 x) {
            assert(std::isfinite(x));
            IndexType2 result = 1 / (exp(x) + exp(-x) + 2);
            assert(std::isfinite(result));
            return result;
        }

        template<> inline IndexType2 Evaluate_0<AF_id::ReLu>(IndexType2 x) {
            return x * (x > 0);
        }

        template<> inline IndexType2 Evaluate_1<AF_id::ReLu>(IndexType2 x) {
            return x > 0 ? 1 : 0;
        }

        template<> inline IndexType2 Evaluate_0<AF_id::Tanh>(IndexType2 x) {
            IndexType2 result = 2 / (1 + exp(-2 * x)) - 1;
            assert(std::isfinite(result));
            return result;
        }

        template<> inline IndexType2 Evaluate_1<AF_id::Tanh>(IndexType2 x) {
            assert(std::isfinite(x));
            IndexType2 result = 4 / (exp(2 * x) + exp(-2 * x) + 2);
            assert(std::isfinite(result));
            return result;
        }

        template<> inline Matrix Evaluate_0_mat<AF_id::Sigmoid>(const Matrix &batch) {
            return batch.unaryExpr(
                    [](IndexType2 x) { return Evaluate_0<AF_id::Sigmoid>(x); });
        }

        template<> inline Matrix Evaluate_1_mat<AF_id::Sigmoid>(const Matrix &batch) {
            return batch.unaryExpr(
                    [](IndexType2 x) { return Evaluate_1<AF_id::Sigmoid>(x); });
        }

        template<> inline Matrix Evaluate_0_mat<AF_id::ReLu>(const Matrix &batch) {
            return batch.unaryExpr(
                    [](IndexType2 x) { return Evaluate_0<AF_id::ReLu>(x); });
        }

        template<> inline Matrix Evaluate_1_mat<AF_id::ReLu>(const Matrix &batch) {
            return batch.unaryExpr(
                    [](IndexType2 x) { return Evaluate_1<AF_id::ReLu>(x); });
        }

        template<> inline Matrix Evaluate_0_mat<AF_id::Tanh>(const Matrix &batch) {
            return batch.unaryExpr(
                    [](IndexType2 x) { return Evaluate_0<AF_id::Tanh>(x); });
        }

        template<> inline Matrix Evaluate_1_mat<AF_id::Tanh>(const Matrix &batch) {
            return batch.unaryExpr(
                    [](IndexType2 x) { return Evaluate_1<AF_id::Tanh>(x); });
        }
    };

    class ActivationFunction {

        FunctionType Evaluate_0_;
        FunctionType Evaluate_1_;

    public:
        ActivationFunction(FunctionType Evaluate_0, FunctionType Evaluate_1);

        template<AF_id func>
        static ActivationFunction Initialize() {
            return ActivationFunction(AFdatabase::Evaluate_0_mat<func>,
                                      AFdatabase::Evaluate_1_mat<func>);
        }

        static ActivationFunction Initialize(AF_id func);
        Matrix Apply(const Matrix &layer_val) const;
        Matrix Derivative(const Matrix &layer_val) const;

        bool IsEmpty();
    };
}
