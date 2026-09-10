add_executable(ash_tests
    version_test.cpp)

target_link_libraries(ash_tests
    PRIVATE
        ash_core
        Catch2::Catch2WithMain)

ash_set_warnings(ash_tests)

include(Catch)
catch_discover_tests(ash_tests)
