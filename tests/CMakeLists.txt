include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(tensor_ops_test tensor_ops_test.cpp)
target_link_libraries(tensor_ops_test PRIVATE OpenMat cuda cudart GTest::gtest_main)

add_test(NAME tensor_ops_test COMMAND tensor_ops_test)
