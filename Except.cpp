#include "Except.h"
#include <exception>
#include <iostream>

namespace Except {
    void React() {
        try {
            throw;
        } catch (std::exception &e) {
            std::cerr << e.what() << '\n';
        } catch (...) {
        }
    }
}
