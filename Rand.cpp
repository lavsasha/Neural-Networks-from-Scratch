#include "Rand.h"

namespace NeuralNets
{
    Matrix Rand::GetNormal(IndexType1 rows, IndexType1 columns) {
        return Matrix::Random(rows, columns);
    }
}
