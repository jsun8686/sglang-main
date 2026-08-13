# =============================================================================
# CMakeLists.txt — HiSparse NPU kernels (AscendC) for SGLang.
#
# Build (on NPU host with CANN toolchain):
#   source /usr/local/Ascend/cann-8.5.0/set_env.sh
#   mkdir -p build && cd build
#   cmake .. -DSOC_VERSION=ascend910_9391
#   make -j$(nproc)
#   # hisparse_lru.so is produced in the source directory
#
# Or use setuptools:
#   SOC_VERSION=ascend910_9391 python setup.py build_ext --inplace
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(hisparse_lru C CXX)

# ── CANN path ──
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME})
    elseif(EXISTS "/usr/local/Ascend/cann-8.5.0")
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-8.5.0")
    elseif(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    else()
        message(FATAL_ERROR "Cannot find CANN installation. Set ASCEND_CANN_PACKAGE_PATH.")
    endif()
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

# ── SoC version ──
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910_9391")
endif()
message(STATUS "SOC_VERSION: ${SOC_VERSION}")

# ── RUN_MODE ──
if(NOT DEFINED RUN_MODE)
    set(RUN_MODE "npu")
endif()

# ── Find AscendC cmake modules ──
set(_ascendc_cmake_candidates
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake
    ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake
    ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake
)
set(ASCENDC_CMAKE_DIR "")
foreach(_path ${_ascendc_cmake_candidates})
    if(EXISTS ${_path}/ascendc.cmake)
        set(ASCENDC_CMAKE_DIR ${_path})
        break()
    endif()
endforeach()

if(NOT ASCENDC_CMAKE_DIR)
    message(FATAL_ERROR "ascendc_kernel_cmake not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
message(STATUS "ASCENDC_CMAKE_DIR: ${ASCENDC_CMAKE_DIR}")
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ── Python / torch / torch_npu ──
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 include: ${Python3_INCLUDE_DIRS}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'include'))"
    OUTPUT_VARIABLE TORCH_NPU_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_NPU_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch lib: ${TORCH_LIB_DIR}")
message(STATUS "Torch NPU include: ${TORCH_NPU_INCLUDE}")
message(STATUS "Torch NPU lib: ${TORCH_NPU_LIB_DIR}")

# ── ACL paths ──
set(ACL_INCLUDE_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
)
set(ACL_LIB_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. AscendC kernel library
# ═══════════════════════════════════════════════════════════════════════════
ascendc_library(hisparse_lru_kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/hisparse_lru_kernel.cpp
)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Host shared library (launch + pybind11)
# ═══════════════════════════════════════════════════════════════════════════
add_library(hisparse_lru SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/hisparse_lru_launch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hisparse_lru_host.cpp
)

target_link_libraries(hisparse_lru PRIVATE hisparse_lru_kernels)

target_include_directories(hisparse_lru PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE}
    ${ACL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_directories(hisparse_lru PRIVATE
    ${TORCH_LIB_DIR}
    ${TORCH_NPU_LIB_DIR}
    ${ACL_LIB_DIRS}
)

target_link_libraries(hisparse_lru PRIVATE
    torch
    torch_cpu
    torch_python
    torch_npu
    ascendcl
)

target_compile_options(hisparse_lru PRIVATE
    -std=c++17
    -O2
    -fPIC
)

target_compile_definitions(hisparse_lru PRIVATE
    TORCH_EXTENSION_NAME=hisparse_lru
)

set_target_properties(hisparse_lru PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "Output: ${CMAKE_CURRENT_SOURCE_DIR}/hisparse_lru.so")
