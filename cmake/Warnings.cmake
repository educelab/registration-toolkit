# GCC & Clang
set(project_warnings
    -Wall
    -Wextra
    -pedantic
)

# Extra warnings we want to enable
list(APPEND project_warnings
    -Wattributes
    -Wbad-function-cast
    -Wc++14-compat
    -Wcast-align
    -Wcast-qual
    -Wchar-subscripts
    -Wcomment
    -Wconversion
    -Wdelete-incomplete
    -Wdelete-non-virtual-dtor
    -Wenum-compare
    -Wimplicit
    -Wmain
    -Wmissing-field-initializers
    -Wmissing-noreturn
    -Wmissing-prototypes
    -Wold-style-cast
    -Woverloaded-virtual
    -Wpointer-arith
    -Wtautological-compare
    -Wundef
    -Wuninitialized
    -Wunreachable-code
    -Wunused
    -Wunused-macros
    -Wvla
)

# Clang specific?
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
list(APPEND project_warnings
    -Wambiguous-ellipsis
    -Warray-bounds-pointer-arithmetic
    -Wassign-enum
    -Wbind-to-temporary-copy
    -Wbitfield-constant-conversion
    -Wbitwise-op-parentheses
    -Wbool-conversion
    -Wc++11-long-long
    -Wclass-varargs
    -Wconditional-uninitialized
    -Wcovered-switch-default
    -Wdeprecated-implementations
    -Wdirect-ivar-access
    -Wdivision-by-zero
    -Wdocumentation
    -Wduplicate-enum
    -Wduplicate-method-arg
    -Wduplicate-method-match
    -Wenum-conversion
    -Wexplicit-ownership-type
    -Wextra-semi
    -Wextra-tokens
    -Wheader-guard
    -Wheader-hygiene
    -Widiomatic-parentheses
    -Wimplicit-fallthrough
    -Winfinite-recursion
    -Wlocal-type-template-args
    -Wloop-analysis
    -Wmethod-signatures
    -Wmismatched-tags
    -Wmove
    -Wnew-returns-null
    -Wpessimizing-move
    -Wredundant-move
    -Wself-assign
    -Wself-move
    -Wsemicolon-before-method-body
    -Wsometimes-uninitialized
    -Wstring-conversion
    -Wstrlcpy-strlcat-size
    -Wstrncat-size
    -Wthread-safety-analysis
    -Wunneeded-internal-declaration
    -Wunreachable-code-break
    -Wunreachable-code-return
    -Wused-but-marked-unused
    -Wvexing-parse
)
endif()

# Disable noisy or unncessary warnings
list(APPEND project_warnings
    -Wno-narrowing
    -Wno-c++1z-extensions
    -Wno-c++98-compat
    -Wno-missing-braces
    -Wno-switch-enum
    -Wno-sign-conversion
    -Wno-shorten-64-to-32
    -Wno-unknown-attributes
    -Wno-unused-function
)
add_compile_options(${project_warnings})
