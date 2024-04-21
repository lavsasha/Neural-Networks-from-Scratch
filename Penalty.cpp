#include "Penalty.h"

namespace NeuralNets {
    IndexType2 Penalty::Pow(IndexType2 x) {
        return std::pow(x, 2);
    }

    IndexType2 Penalty::FindDist(const Vector &z, const Vector &y) {
        assert(z.size() == y.size());
        Vector squared_diff = (z - y).unaryExpr(&Pow);
        return std::accumulate(squared_diff.data(), squared_diff.data() + squared_diff.size(), 0.0);
    }

    IndexType2 Penalty::CalcPenalty(const Matrix &batch_y, const Matrix &network_out) {
        assert(batch_y.rows() == network_out.rows());
        assert(batch_y.cols() == network_out.cols());
        Matrix squared_diff = (batch_y - network_out).unaryExpr(&Pow);
        return squared_diff.sum() / batch_y.cols();
    }

    Matrix Penalty::FindInitialGradient(const Matrix &batch_y, const Matrix &network_out) {
        assert(batch_y.rows() == network_out.rows());
        assert(batch_y.cols() == network_out.cols());
        return 2 * (batch_y.transpose() - network_out.transpose());
    }
}
