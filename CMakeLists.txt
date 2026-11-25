cmake_minimum_required(VERSION 3.16)
project(knn17 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -march=native: the micro-kernel relies on the compiler mapping the 16-wide
# vector type to AVX-512 (one register) or AVX2 (two registers).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
option(KNN_SANITIZE "Build with address/undefined sanitizers" OFF)
if(KNN_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(OpenMP REQUIRED)

add_library(knn STATIC src/reference.cpp src/tiled.cpp src/data.cpp src/utils.cpp)
target_include_directories(knn PUBLIC include)
target_link_libraries(knn PUBLIC OpenMP::OpenMP_CXX)

add_executable(knn_test tests/test_knn.cpp)
target_link_libraries(knn_test knn)

add_executable(knn_bench bench/knn_bench.cpp)
target_link_libraries(knn_bench knn)

add_executable(bandwidth bench/bandwidth.cpp)
target_link_libraries(bandwidth knn)

enable_testing()
add_test(NAME knn_test COMMAND knn_test)
