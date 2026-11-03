cmake_minimum_required(VERSION 3.24)
project(strata_boundary_probe LANGUAGES CXX CUDA)

get_filename_component(strata_root "../../.." ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(pending_write
    pending_write.cpp
    pending_write_cuda.cu
    "${strata_root}/src/kernels/cuda/native_gr_postops.cu")
target_include_directories(pending_write PRIVATE "${strata_root}/include")
target_compile_features(pending_write PRIVATE cxx_std_17 cuda_std_17)
# Match the native GR kernel's arithmetic in the main CUDA build.
set_source_files_properties("${strata_root}/src/kernels/cuda/native_gr_postops.cu"
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")
