#include "Rand.h"

namespace NeuralNets
{
    Matrix Rand::GetNormal(IndexType1 rows, IndexType1 cols) {
        return Matrix::Random(rows, cols);
    }
}
