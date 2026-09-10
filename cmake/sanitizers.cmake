option(ASH_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# This is the net that catches what the compiler's static analysis only guesses
# at: real null dereferences, use-after-free, data races' close relatives, and
# undefined behaviour. Run the test suite under it before a release.
function(ash_set_sanitizers target)
    if(NOT ASH_SANITIZE)
        return()
    endif()

    if(MSVC)
        message(WARNING "ASH_SANITIZE is ignored for MSVC")
        return()
    endif()

    target_compile_options(${target} PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        # Undefined behaviour is a failure, not a warning to scroll past.
        -fno-sanitize-recover=undefined)

    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
endfunction()
