add_library(ash_eval
    report.cpp
    runner.cpp
    suite.cpp)

target_include_directories(ash_eval
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/eval>
        $<INSTALL_INTERFACE:eval>)

target_link_libraries(ash_eval
    PUBLIC
        ash_core
    PRIVATE
        nlohmann_json::nlohmann_json)

ash_set_warnings(ash_eval)
ash_set_sanitizers(ash_eval)
