#include "Sigmoid.h"
namespace NeuralNets
{
    IndexType2 Sigmoid::CalcExp_0(IndexType2 x)
    {
        return 1 / (1 + exp(-x));
    }

    IndexType2 Sigmoid::CalcExp_1(IndexType2 x)
    {
        return 1 / (exp(x) + exp(-x) + 2);
    }

    Matrix Sigmoid::EvaluateFunc(const Matrix &batch)
    {
        return batch.unaryExpr(&CalcExp_0);
    }

    Matrix Sigmoid::EvaluateDerivative(const Matrix &batch)
    {
        return batch.unaryExpr(&CalcExp_1);
    }
}
