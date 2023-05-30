project(testcoreapi)
# set(CMAKE_VERBOSE_MAKEFILE ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_link_options(-D_GLIBCXX_USE_CXX11_ABI=0)

option(SNAPML_LIB_DIR "SNAPML_LIB_DIR" "")
option(SNAPML_INCLUDE_DIR "SNAPML_INCLUDE_DIR" "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
include_directories(PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(test-rf "src/capi-rf-example.cpp" "src/Dataset.cpp")
add_executable(test-boost "src/capi-booster-example.cpp" "src/Dataset.cpp")
add_executable(test-dt "src/capi-dt-example.cpp" "src/Dataset.cpp")

target_include_directories(test-rf PUBLIC "${SNAPML_INCLUDE_DIR}")
target_include_directories(test-boost PUBLIC "${SNAPML_INCLUDE_DIR}")
target_include_directories(test-dt PUBLIC "${SNAPML_INCLUDE_DIR}")

target_link_libraries(test-rf "${SNAPML_LIB_DIR}/libsnapmlcore.so")
target_link_libraries(test-boost "${SNAPML_LIB_DIR}/libsnapmlcore.so")
target_link_libraries(test-dt "${SNAPML_LIB_DIR}/libsnapmlcore.so")

# target_link_libraries(test-rf "${SNAPML_LIB_DIR}/libsnapmlcore_avx2.so")
# target_link_libraries(test-boost "${SNAPML_LIB_DIR}/libsnapmlcore_avx2.so")
# target_link_libraries(test-dt "${SNAPML_LIB_DIR}/libsnapmlcore_avx2.so")

