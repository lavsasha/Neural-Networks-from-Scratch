#include "Except.h"
#include "tests.h"

int main() {
    try {
        test::RunAllTests();
    } catch (...) {
        Except::React();
    }
    return 0;
}
