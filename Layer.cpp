#include "Layer.h"

Layer::Layer(int64_t m, int64_t n) {
    A.resize(m, n);
    b.resize(m, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = distribution(gen);
        }
        b(i) = distribution(gen);
    }
}

Eigen::MatrixXd Layer::CalcLayerValue(const Eigen::MatrixXd &x) {
    Eigen::MatrixXd b_matrix(b.rows(), x.cols());
    for (int i = 0; i < x.cols(); ++i) {
        b_matrix.col(i) = b;
    }
    Eigen::MatrixXd temp = A * x + b_matrix;
    return Sigma::EvaluateFunc(temp);
}

Eigen::MatrixXd Layer::GradA(const Eigen::MatrixXd &x, const Eigen::MatrixXd &u) {
    assert(x.cols() == u.rows());
    Eigen::MatrixXd grad(A.rows(), A.cols());
    Eigen::MatrixXd u_tr = u.transpose();
    Eigen::MatrixXd x_tr = x.transpose();
    Eigen::MatrixXd b_matrix(b.rows(), x.cols());
    for (int i = 0; i < x.cols(); ++i) {
        b_matrix.col(i) = b;
    }
    Eigen::MatrixXd temp = A * x + b_matrix;
    Eigen::MatrixXd d_sigma = Sigma::EvaluateDerivative(temp);
    for (int i = 0; i < d_sigma.cols(); ++i) {
        Eigen::MatrixXd col_diagonal = d_sigma.col(i).asDiagonal();
        grad += col_diagonal * u_tr.col(i) * x_tr.row(i);
    }
    return grad;
}

Eigen::MatrixXd Layer::Gradb(const Eigen::MatrixXd &x, const Eigen::MatrixXd &u) {
    assert(x.cols() == u.rows());
    Eigen::MatrixXd u_tr = u.transpose();
    Eigen::MatrixXd b_matrix(b.rows(), x.cols());
    for (int i = 0; i < x.cols(); ++i) {
        b_matrix.col(i) = b;
    }
    Eigen::MatrixXd temp = A * x + b_matrix;
    Eigen::MatrixXd d_sigma = Sigma::EvaluateDerivative(temp);
    Eigen::MatrixXd grad(b.rows(), 1);
    for (int i = 0; i < d_sigma.cols(); ++i) {
        Eigen::MatrixXd col_diagonal = d_sigma.col(i).asDiagonal();
        grad += col_diagonal * u_tr.col(i);
    }
    return grad;
}

Eigen::MatrixXd Layer::NextGrad(const Eigen::MatrixXd& x, const Eigen::MatrixXd& u) {
    assert(x.cols() == u.rows());
    Eigen::MatrixXd next_grad(u.rows(), A.cols());
    Eigen::MatrixXd b_matrix(b.rows(), x.cols());
    for (int i = 0; i < x.cols(); ++i) {
        b_matrix.col(i) = b;
    }
    Eigen::MatrixXd temp = A * x + b_matrix;
    Eigen::MatrixXd d_sigma = Sigma::EvaluateDerivative(temp);
    for (int i = 0; i < u.rows(); ++i) {
        Eigen::MatrixXd col_diagonal = d_sigma.col(i).asDiagonal();
        next_grad.row(i) = u.row(i) * col_diagonal * A;
    }
    return next_grad;
}