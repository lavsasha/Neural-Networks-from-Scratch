#pragma once

#include "usings.h"

namespace NeuralNets {

    enum class PenaltyId {
        MSE, MAE, CrossEntropy
    };

    struct PenaltyDatabase {

        template<PenaltyId>
        static IndexType2 CalcPenalty(const Matrix &, const Matrix &);

        template<PenaltyId>
        static Matrix FindInitialGradient(const Matrix &, const Matrix &);

        template<>
        inline IndexType2 CalcPenalty<PenaltyId::MSE>(const Matrix &network_out, const Matrix &batch_y) {
            assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
            Matrix squared_diff = (network_out - batch_y).unaryExpr([](IndexType2 el) { return el * el; });
            return GetRowOf1(squared_diff.rows()) *
                   (squared_diff * GetVecOf1(squared_diff.cols()) / squared_diff.cols());
        }

        template<>
        inline Matrix FindInitialGradient<PenaltyId::MSE>(const Matrix &network_out, const Matrix &batch_y) {
            assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
            return 2.0 * (network_out.transpose() - batch_y.transpose());
        }

        template<>
        inline IndexType2 CalcPenalty<PenaltyId::MAE>(const Matrix &network_out, const Matrix &batch_y) {
            assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
            Matrix abs_diff = (network_out - batch_y).unaryExpr([](IndexType2 el) { return abs(el); });
            return GetRowOf1(abs_diff.rows()) *
                   (abs_diff * GetVecOf1(abs_diff.cols()) / abs_diff.cols());
        }

        template<>
        inline Matrix FindInitialGradient<PenaltyId::MAE>(const Matrix &network_out, const Matrix &batch_y) {
            assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
            return (network_out.transpose() - batch_y.transpose()).unaryExpr([](IndexType2 el) {
                return el > 0 ? 1.0 : -1.0;
            });
        }

        template<>
        inline IndexType2 CalcPenalty<PenaltyId::CrossEntropy>(const Matrix &network_out, const Matrix &batch_y) {
            assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
            Matrix cross_diff = batch_y.cwiseProduct(network_out.unaryExpr([](IndexType2 el) {
                return log(el);
            }));
            return GetRowOf1(cross_diff.rows()) *
                   (cross_diff * GetVecOf1(cross_diff.cols()) / cross_diff.cols());
        }

        template<>
        inline Matrix FindInitialGradient<PenaltyId::CrossEntropy>(const Matrix &network_out, const Matrix &batch_y) {
            assert(network_out.size() == batch_y.size() && "network_out and batch_y must have the same size");
            return -batch_y.transpose().cwiseProduct(
                    network_out.transpose().unaryExpr([](IndexType2 el) { return 1.0 / el; }));
        }
    };

    class Penalty {
        PenaltyFuncType PenaltyFunc_;
        GradientFuncType GradientFunc_;

    public:
        Penalty(PenaltyFuncType PenaltyFunc, GradientFuncType GradientFunc);

        template<PenaltyId penalty> static Penalty Initialize();
        static Penalty Initialize(PenaltyId penalty);

        IndexType2 CalcPenalty(const Matrix &network_out, const Matrix &batch_y) const;
        Matrix FindInitialGradient(const Matrix &network_out, const Matrix &batch_y) const;

        bool IsEmpty();
    };
}
