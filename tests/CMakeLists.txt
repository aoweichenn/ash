add_executable(ash_tests
    agent_loop_test.cpp
    boundary_test.cpp
    cli_logic_test.cpp
    concurrency_test.cpp
    eval_test.cpp
    fixture_replay_test.cpp
    limited_provider_test.cpp
    line_reader_test.cpp
    permission_test.cpp
    provider_test.cpp
    record_test.cpp
    shell_tool_test.cpp
    sse_test.cpp
    stream_test.cpp
    task_test.cpp
    terminal_test.cpp
    thread_pool_test.cpp
    tool_test.cpp
    version_test.cpp)

target_link_libraries(ash_tests
    PRIVATE
        ash_core
        ash_eval
        Catch2::Catch2WithMain)

# The CLI is not a library, so the parts of it that are decisions rather than
# printing are tested by including the headers directly. Only the pure ones:
# console.hpp reaches for <iostream> and has no business in a test.
target_include_directories(ash_tests PRIVATE "${PROJECT_SOURCE_DIR}/apps/cli")

# The boundary test reads the public headers from the source tree.
target_compile_definitions(ash_tests PRIVATE ASH_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

ash_set_warnings(ash_tests)
ash_set_sanitizers(ash_tests)

include(Catch)
# A concurrency regression here shows up as a deadlock, and ctest would
# otherwise sit on it for its 25-minute default. Every test in the suite
# finishes in well under a second, so a minute means something is wedged.
catch_discover_tests(ash_tests PROPERTIES TIMEOUT 60)
