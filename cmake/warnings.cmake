function(ash_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wnon-virtual-dtor
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wnull-dereference)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # GCC 15's -Wnull-dereference fires on inlined libstdc++ internals and
        # reports the struct definition rather than a real dereference -- a
        # false positive. Clang's analysis is sound, so it keeps the warning;
        # genuine null dereferences are caught by the ASan/UBSan build instead.
        target_compile_options(${target} PRIVATE -Wno-null-dereference)
    endif()

    if(ASH_WERROR)
        if(MSVC)
            target_compile_options(${target} PRIVATE /WX)
        else()
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()
