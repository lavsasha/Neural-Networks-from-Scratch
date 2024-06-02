#pragma once

#include "EigenProxy.h"

namespace NeuralNets
{
    class Rand {
    public:
        static Matrix GetNormal(Index rows, Index cols);
    };
}
