add_executable(ash_tests
    agent_loop_test.cpp
    boundary_test.cpp
    eval_test.cpp
    fixture_replay_test.cpp
    provider_test.cpp
    record_test.cpp
    task_test.cpp
    thread_pool_test.cpp
    tool_test.cpp
    version_test.cpp)

target_link_libraries(ash_tests
    PRIVATE
        ash_core
        ash_eval
        Catch2::Catch2WithMain)

# The boundary test reads the public headers from the source tree.
target_compile_definitions(ash_tests PRIVATE ASH_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

ash_set_warnings(ash_tests)
ash_set_sanitizers(ash_tests)

include(Catch)
catch_discover_tests(ash_tests)
