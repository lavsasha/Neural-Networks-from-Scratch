#include "Penalty.h"

namespace NeuralNets {
    Penalty::Penalty(PenaltyFuncType PenaltyFunc, GradientFuncType GradientFunc)
            : PenaltyFunc_(std::move(PenaltyFunc)),
              GradientFunc_(std::move(GradientFunc)) {
            assert(PenaltyFunc_ && "Empty Penalty function!");
            assert(GradientFunc_ && "Empty Gradient function!");
    }

    template<PenaltyId penalty> Penalty Penalty::Initialize() {
        return Penalty(PenaltyDatabase::CalcPenalty<penalty>, PenaltyDatabase::FindInitialGradient<penalty>);
    }

    Penalty Penalty::Initialize(PenaltyId penalty) {
        switch (penalty) {
            case PenaltyId::MSE: {
                return Initialize<PenaltyId::MSE>();
            }
            case PenaltyId::MAE: {
                return Initialize<PenaltyId::MAE>();
            }
            case PenaltyId::CrossEntropy: {
                return Initialize<PenaltyId::CrossEntropy>();
            }
            default:
                return Initialize<PenaltyId::MSE>();
        }
    }

    IndexType2 Penalty::CalcPenalty(const Matrix &network_out, const Matrix &batch_y) const {
        return PenaltyFunc_(network_out, batch_y);
    }

    Matrix Penalty::FindInitialGradient(const Matrix &network_out, const Matrix &batch_y) const {
        return GradientFunc_(network_out, batch_y);
    }

    bool Penalty::IsEmpty() { return PenaltyFunc_ && GradientFunc_; }

}
