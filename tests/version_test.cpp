#include <catch2/catch_test_macros.hpp>

#include "ash/version.hpp"

TEST_CASE("version is reported", "[version]") {
    REQUIRE_FALSE(ash::version().empty());
    REQUIRE(ash::version() == ash::kVersion);
}
