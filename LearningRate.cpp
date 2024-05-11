#include "LearningRate.h"

namespace NeuralNets {
    LearningRate::LearningRate(IndexType2 init_rate) : schedule_(LRSchedule::Constant), init_rate_(init_rate),
                                                       reduct_coef_(1) {}
    LearningRate::LearningRate(IndexType2 init_rate, LRSchedule schedule)
            : schedule_(schedule), init_rate_(init_rate), reduct_coef_(1) {
        assert(schedule == LRSchedule::Linear && "This constructor is only for linear schedule");}
    LearningRate::LearningRate(IndexType2 init_rate, IndexType2 reduct_coef)
            : schedule_(LRSchedule::Exponential), init_rate_(init_rate), reduct_coef_(reduct_coef) {}

    IndexType2 LearningRate::GetRate(Index epoch) const {
        switch (schedule_) {
            case LRSchedule::Constant:
                return init_rate_;
            case LRSchedule::Linear:
                return init_rate_ / epoch;
            case LRSchedule::Exponential:
                return init_rate_ * exp(-epoch * reduct_coef_);
            default:
                return init_rate_;
        }
    }

    void LearningRate::ChangeRate() {
        if (schedule_ == LRSchedule::Constant) {
            schedule_ = LRSchedule::Linear;
        } else if (schedule_ == LRSchedule::Linear) {
            schedule_ = LRSchedule::Exponential;
        } else {
            schedule_ = LRSchedule::Constant;
        }
    }
}
