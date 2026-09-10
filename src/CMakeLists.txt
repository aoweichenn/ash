add_library(ash_core
    version.cpp
    io/thread_pool.cpp)

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
