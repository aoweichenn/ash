add_library(ash_core
    version.cpp
    cost.cpp
    core/agent_loop.cpp
    io/http_client.cpp
    io/thread_pool.cpp
    model/anthropic.cpp
    model/openai_compat.cpp
    model/provider.cpp
    model/sse.cpp
    model/stream.cpp
    record/decorators.cpp
    record/journal.cpp
    record/replay.cpp
    record/replay_run.cpp
    tool/builtin.cpp
    tool/permission.cpp
    tool/registry.cpp
    tool/shell.cpp)

target_include_directories(ash_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_definitions(ash_core
    PUBLIC ASH_VERSION_STRING="${PROJECT_VERSION}")

target_link_libraries(ash_core
    PUBLIC
        nlohmann_json::nlohmann_json
        Threads::Threads
        CURL::libcurl
        OpenSSL::SSL
        OpenSSL::Crypto
    PRIVATE
        spdlog::spdlog)

ash_set_warnings(ash_core)
ash_set_sanitizers(ash_core)
