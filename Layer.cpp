#include "Layer.h"

namespace NeuralNets {

    Layer::Layer(IndexType1 cols, IndexType1 rows, AF_id func)
            : A_(Rand::GetNormal(rows, cols)),
              b_(Rand::GetNormal(rows, 1)),
              ActivationFunction_(ActivationFunction::Initialize(func)) {}

    Matrix Layer::Evaluate(const Matrix &batch) const {
        assert(A_.cols() == batch.rows() && "Matrix A should have as many columns as there are rows in batch");
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        return ActivationFunction_.Apply(linear_part);
    }

    Matrix Layer::GradA(const Matrix &batch, const Matrix &grad) {
        assert(A_.cols() == batch.rows() && "Matrix A should have as many columns as there are rows in batch");
        assert(A_.rows() == grad.cols() && "Matrix A should have as many rows as there are columns in grad");
        assert(grad.rows() == batch.cols() && "batch should have as many columns as there are rows in grad");
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        Matrix d_sigma = ActivationFunction_.Derivative(linear_part);
        return d_sigma.cwiseProduct(grad.transpose()) * batch.transpose() / batch.cols();
    }

    Vector Layer::Gradb(const Matrix &batch, const Matrix &grad) {
        assert(A_.cols() == batch.rows() && "Matrix A should have as many columns as there are rows in batch");
        assert(A_.rows() == grad.cols() && "Matrix A should have as many rows as there are columns in grad");
        assert(grad.rows() == batch.cols() && "batch should have as many columns as there are rows in grad");
        Matrix linear_part = A_ * batch + b_ * GetRowOf1(batch.cols());
        Matrix d_sigma = ActivationFunction_.Derivative(linear_part);
        return d_sigma.cwiseProduct(grad.transpose()) * GetVecOf1(batch.cols()) / batch.cols();
    }

    void Layer::ChangeParams(const Matrix &delta_A, const Vector &delta_b, IndexType2 step) {
        assert(A_.size() == delta_A.size() && "Matrix A and its gradient should have the same size!");
        assert(b_.size() == delta_b.size() && "Vector b and its gradient should have the same size!");
        A_ -= step * delta_A;
        b_ -= step * delta_b;
    }

    Matrix Layer::NextGrad(const Matrix &batch, const Matrix &grad) const {
        assert(A_.cols() == batch.rows() && "Matrix A should have as many columns as there are rows in batch");
        assert(A_.rows() == grad.cols() && "Matrix A should have as many rows as there are columns in grad");
        assert(grad.rows() == batch.cols() && "batch should have as many columns as there are rows in grad");
        Matrix linear_part = (A_ * batch + b_ * GetRowOf1(batch.cols())).eval();
        Matrix d_sigma = ActivationFunction_.Derivative(linear_part);
        return grad.cwiseProduct(d_sigma.transpose()) * A_;
    }
}
