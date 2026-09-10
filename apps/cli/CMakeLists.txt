add_executable(ash main.cpp)

target_link_libraries(ash PRIVATE ash_core ash_eval)

ash_set_warnings(ash)
ash_set_sanitizers(ash)
