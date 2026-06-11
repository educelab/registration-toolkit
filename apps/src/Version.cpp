#include <cstdlib>
#include <iostream>

#include "rt/Version.hpp"

using namespace rt;

auto main() -> int
{
    std::cout << ProjectInfo::NameAndVersion();
    std::cout << " (" << ProjectInfo::RepositoryShortHash() << ")\n";
    return EXIT_SUCCESS;
}
