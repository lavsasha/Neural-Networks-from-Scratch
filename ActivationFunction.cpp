#include "ActivationFunction.h"

namespace NeuralNets {

    ActivationFunction::ActivationFunction(FunctionType Evaluate_0, FunctionType Evaluate_1)
            : Evaluate_0_(std::move(Evaluate_0)), Evaluate_1_(std::move(Evaluate_1)) {}

    ActivationFunction ActivationFunction::Initialize(AF_id func) {
        switch (func) {
            case AF_id::Sigmoid:
                return ActivationFunction::Initialize<AF_id::Sigmoid>();
            case AF_id::ReLu:
                return ActivationFunction::Initialize<AF_id::ReLu>();
            case AF_id::Tanh:
                return ActivationFunction::Initialize<AF_id::Tanh>();
            default:
                return ActivationFunction::Initialize<AF_id::Sigmoid>();
        }
    }

    Matrix ActivationFunction::Apply(const Matrix &layer_val) const {
        assert(Evaluate_0_ && "Empty Evaluate_0 method!");
        return Evaluate_0_(layer_val);
    }

    Matrix ActivationFunction::Derivative(const Matrix &layer_val) const {
        assert(Evaluate_1_ && "Empty Evaluate_1 method!");
        return Evaluate_1_(layer_val);
    }

    bool ActivationFunction::IsEmpty() {
        return Evaluate_0_ && Evaluate_1_;
    }
}
