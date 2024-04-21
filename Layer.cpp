#include "Layer.h"

namespace NeuralNets {
    Row Layer::GetRowOf1(IndexType1 cols) {
        return Row::Ones(cols);
    }

    Vector Layer::GetVecOf1(IndexType1 rows) {
        return Vector::Ones(rows);
    }

    Layer::Layer(IndexType1 rows, IndexType1 columns) : A_(Rand::GetNormal(rows, columns)),
                                                        b_(Rand::GetNormal(rows, 1)) {}

    Matrix Layer::Evaluate(const Matrix &batch) {
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        return Sigmoid::EvaluateFunc(linear_part);
    }

    Matrix Layer::GradA(const Matrix &batch, const Matrix &u) {
        assert(batch.cols() == u.rows());
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        Matrix d_sigma = Sigmoid::EvaluateDerivative(linear_part);
        Matrix grad = d_sigma.cwiseProduct(u.transpose()) * batch.transpose();
        return grad / batch.cols();
    }

    Matrix Layer::Gradb(const Matrix &batch, const Matrix &u) {
        assert(batch.cols() == u.rows());
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        Matrix d_sigma = Sigmoid::EvaluateDerivative(linear_part);
        Matrix grad = d_sigma.cwiseProduct(u.transpose()) * GetVecOf1(batch.cols());
        return grad / batch.cols();
    }

    Matrix Layer::NextGrad(const Matrix &batch, const Matrix &u) {
        assert(batch.cols() == u.rows());
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        Matrix d_sigma = Sigmoid::EvaluateDerivative(linear_part);
        Matrix next_grad = u.cwiseProduct(d_sigma.transpose()) * A_;
        return next_grad;
    }
}
