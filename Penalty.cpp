#include "Penalty.h"

namespace NeuralNets {
    Penalty::Penalty(PenaltyFuncType PenaltyFunc, GradientFuncType GradientFunc)
            : PenaltyFunc_(std::move(PenaltyFunc)),
              GradientFunc_(std::move(GradientFunc)) {}

    template<PenaltyId penalty> Penalty Penalty::Initialize() {
        return Penalty(NeuralNets::CalcPenalty<penalty>, NeuralNets::FindInitialGradient<penalty>);
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
        assert(PenaltyFunc_ && "Empty PenaltyFunc method!");
        return PenaltyFunc_(network_out, batch_y);
    }

    Matrix Penalty::FindInitialGradient(const Matrix &network_out, const Matrix &batch_y) const {
        assert(GradientFunc_ && "Empty GradientFunc method!");
        return GradientFunc_(network_out, batch_y);
    }

    bool Penalty::IsEmpty() {
        return PenaltyFunc_ && GradientFunc_;
    }
}
