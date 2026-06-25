include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)

FetchContent_MakeAvailable(googletest)

function(add_om_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE OpenMat cuda cudart GTest::gtest_main)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_om_test(test_arithmetic      test_arithmetic.cpp)
add_om_test(test_fused_ops       test_fused_ops.cpp)
add_om_test(test_device_transfer test_device_transfer.cpp)
add_om_test(test_factory         test_factory.cpp)
add_om_test(test_reductions      test_reductions.cpp)
add_om_test(test_benchmarks      test_benchmarks.cpp)
add_om_test(test_reshape         test_reshape.cpp)
add_om_test(test_transpose       test_transpose.cpp)
add_om_test(test_streams         test_streams.cpp)
