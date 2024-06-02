#pragma once

#include "EigenProxy.h"

namespace NeuralNets {
    using Scalar = double;

    enum class AF_id {
        Sigmoid, ReLu, Tanh
    };

    template<AF_id>
    static Scalar Evaluate_0(Scalar);

    template<AF_id>
    static Scalar Evaluate_1(Scalar);

    template<AF_id>
    static Matrix Evaluate_0_mat(const Matrix &);

    template<AF_id>
    static Matrix Evaluate_1_mat(const Matrix &);

    template<>
    inline Scalar Evaluate_0<AF_id::Sigmoid>(Scalar x) {
        Scalar result = 1 / (1 + exp(-x));
        assert(std::isfinite(result));
        return result;
    }

    template<>
    inline Scalar Evaluate_1<AF_id::Sigmoid>(Scalar x) {
        assert(std::isfinite(x));
        Scalar result = 1 / (exp(x) + exp(-x) + 2);
        assert(std::isfinite(result));
        return result;
    }

    template<>
    inline Scalar Evaluate_0<AF_id::ReLu>(Scalar x) {
        return x * (x > 0);
    }

    template<>
    inline Scalar Evaluate_1<AF_id::ReLu>(Scalar x) {
        return x > 0;
    }

    template<>
    inline Scalar Evaluate_0<AF_id::Tanh>(Scalar x) {
        Scalar result = 2 / (1 + exp(-2 * x)) - 1;
        assert(std::isfinite(result));
        return result;
    }

    template<>
    inline Scalar Evaluate_1<AF_id::Tanh>(Scalar x) {
        assert(std::isfinite(x));
        Scalar result = 4 / (exp(2 * x) + exp(-2 * x) + 2);
        assert(std::isfinite(result));
        return result;
    }

    template<>
    inline Matrix Evaluate_0_mat<AF_id::Sigmoid>(const Matrix &batch) {
        return batch.unaryExpr(
                [](Scalar x) { return Evaluate_0<AF_id::Sigmoid>(x); });
    }

    template<>
    inline Matrix Evaluate_1_mat<AF_id::Sigmoid>(const Matrix &batch) {
        return batch.unaryExpr(
                [](Scalar x) { return Evaluate_1<AF_id::Sigmoid>(x); });
    }

    template<>
    inline Matrix Evaluate_0_mat<AF_id::ReLu>(const Matrix &batch) {
        return batch.unaryExpr(
                [](Scalar x) { return Evaluate_0<AF_id::ReLu>(x); });
    }

    template<>
    inline Matrix Evaluate_1_mat<AF_id::ReLu>(const Matrix &batch) {
        return batch.unaryExpr(
                [](Scalar x) { return Evaluate_1<AF_id::ReLu>(x); });
    }

    template<>
    inline Matrix Evaluate_0_mat<AF_id::Tanh>(const Matrix &batch) {
        return batch.unaryExpr(
                [](Scalar x) { return Evaluate_0<AF_id::Tanh>(x); });
    }

    template<>
    inline Matrix Evaluate_1_mat<AF_id::Tanh>(const Matrix &batch) {
        return batch.unaryExpr(
                [](Scalar x) { return Evaluate_1<AF_id::Tanh>(x); });
    }

    class ActivationFunction {
        using FunctionType = std::function<Matrix(const Matrix &)>;
        FunctionType Evaluate_0_;
        FunctionType Evaluate_1_;

    public:
        ActivationFunction(FunctionType Evaluate_0, FunctionType Evaluate_1);

        template<AF_id func> static ActivationFunction Initialize() {
            return ActivationFunction(Evaluate_0_mat<func>, Evaluate_1_mat<func>);
        }
        static ActivationFunction Initialize(AF_id func);

        Matrix Apply(const Matrix &linear_part) const;
        Matrix Derivative(const Matrix &linear_part) const;

        bool IsEmpty();
    };
}
