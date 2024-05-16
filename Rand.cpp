#include "Rand.h"

namespace NeuralNets
{
    Matrix Rand::GetNormal(Index rows, Index cols) {
        return Matrix::Random(rows, cols);
    }
}
