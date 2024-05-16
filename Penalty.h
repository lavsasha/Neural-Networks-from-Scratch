#pragma once

#include "EigenProxy.h"

namespace NeuralNets {
    using Scalar = double;

    enum class PenaltyId {
        MSE, MAE, CrossEntropy
    };

    template<PenaltyId>
    static Scalar CalcPenalty(const Matrix &, const Matrix &);

    template<PenaltyId>
    static Matrix FindInitialGradient(const Matrix &, const Matrix &);

    template<>
    inline Scalar CalcPenalty<PenaltyId::MSE>(const Matrix &network_out, const Matrix &batch_y) {
        assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
        Matrix squared_diff = (network_out - batch_y).unaryExpr([](Scalar el) { return el * el; });
        return GetRowOf1(squared_diff.rows()) *
               (squared_diff * GetVecOf1(squared_diff.cols()) / squared_diff.cols());
    }

    template<>
    inline Matrix FindInitialGradient<PenaltyId::MSE>(const Matrix &network_out, const Matrix &batch_y) {
        assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
        return 2.0 * (network_out.transpose() - batch_y.transpose());
    }

    template<>
    inline Scalar CalcPenalty<PenaltyId::MAE>(const Matrix &network_out, const Matrix &batch_y) {
        assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
        Matrix abs_diff = (network_out - batch_y).unaryExpr([](Scalar el) { return abs(el); });
        return GetRowOf1(abs_diff.rows()) *
               (abs_diff * GetVecOf1(abs_diff.cols()) / abs_diff.cols());
    }

    template<>
    inline Matrix FindInitialGradient<PenaltyId::MAE>(const Matrix &network_out, const Matrix &batch_y) {
        assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
        return (network_out.transpose() - batch_y.transpose()).unaryExpr([](Scalar el) {
            return el > 0 ? 1.0 : -1.0;
        });
    }

    template<>
    inline Scalar CalcPenalty<PenaltyId::CrossEntropy>(const Matrix &network_out, const Matrix &batch_y) {
        assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
        assert(network_out.minCoeff() > 0 && "all elements of network_out must be positive");
        Matrix cross_diff = batch_y.cwiseProduct(network_out.unaryExpr([](Scalar el) {
            return log(el);
        }));
        return GetRowOf1(cross_diff.rows()) *
               (cross_diff * GetVecOf1(cross_diff.cols()) / cross_diff.cols());
    }

    template<>
    inline Matrix FindInitialGradient<PenaltyId::CrossEntropy>(const Matrix &network_out, const Matrix &batch_y) {
        assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
        assert(network_out.minCoeff() > 0 && "all elements of network_out must be positive");
        return -batch_y.transpose().cwiseProduct(
                network_out.transpose().unaryExpr([](Scalar el) { return 1.0 / el; }));
    }

    class Penalty {
    public:
        using PenaltyFuncType = std::function<Scalar(const Matrix &, const Matrix &)>;
        using GradientFuncType = std::function<Matrix(const Matrix &, const Matrix &)>;

        Penalty(PenaltyFuncType PenaltyFunc, GradientFuncType GradientFunc);

        template<PenaltyId penalty> static Penalty Initialize() {
            return Penalty(NeuralNets::CalcPenalty<penalty>, NeuralNets::FindInitialGradient<penalty>);
        };

        static Penalty Initialize(PenaltyId penalty);

        Scalar CalcPenalty(const Matrix &network_out, const Matrix &batch_y) const;

        Matrix FindInitialGradient(const Matrix &network_out, const Matrix &batch_y) const;

        bool IsEmpty();

    private:
        PenaltyFuncType PenaltyFunc_;
        GradientFuncType GradientFunc_;
    };
}
