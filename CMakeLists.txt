cmake_minimum_required(VERSION 3.18)
project(tyger.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TYGER_VULKAN        "tyger: build the Vulkan backend"   ON)
option(TYGER_BUILD_TOOLS   "tyger: build cli, quantize, server" ON)
option(TYGER_NATIVE        "tyger: optimize for the build machine's cpu" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ggml is linked statically into libtyger
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${TYGER_VULKAN} CACHE BOOL "" FORCE)
set(GGML_NATIVE ${TYGER_NATIVE} CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

find_package(Threads REQUIRED)

add_library(tyger_obj OBJECT
    src/tyger.cpp
    src/model.cpp
    src/mel.cpp
    src/encoder.cpp
    src/turn_taker.cpp
    src/rnnt.cpp
    src/rules.cpp
    src/engine.cpp
)
target_include_directories(tyger_obj PUBLIC include src)
target_compile_definitions(tyger_obj PUBLIC TYGER_BUILD)
target_link_libraries(tyger_obj PUBLIC ggml Threads::Threads)
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(tyger_obj PUBLIC OpenMP::OpenMP_CXX)
endif()
if (TYGER_NATIVE)
    if (MSVC)
        target_compile_options(tyger_obj PRIVATE /arch:AVX2)
    else()
        target_compile_options(tyger_obj PRIVATE -march=native)
    endif()
endif()
if (MSVC)
    target_compile_options(tyger_obj PRIVATE /W3 /utf-8 /fp:precise)
else()
    target_compile_options(tyger_obj PRIVATE -Wall -Wextra -Wno-unused-parameter -fvisibility=hidden)
endif()

add_library(tyger SHARED $<TARGET_OBJECTS:tyger_obj>)
target_include_directories(tyger PUBLIC include)
target_link_libraries(tyger PRIVATE ggml Threads::Threads)
if (OpenMP_CXX_FOUND)
    target_link_libraries(tyger PRIVATE OpenMP::OpenMP_CXX)
endif()

option(TYGER_BUILD_TESTS "tyger: build tests" ON)
if (TYGER_BUILD_TESTS)
    add_executable(test-stream tests/test_stream.cpp tools/common/wav.cpp)
    target_include_directories(test-stream PRIVATE tools/common)
    target_link_libraries(test-stream PRIVATE tyger_obj)
endif()

if (TYGER_BUILD_TOOLS)
    add_executable(tyger-cli tools/cli/main.cpp tools/common/wav.cpp)
    target_include_directories(tyger-cli PRIVATE tools/common)
    target_link_libraries(tyger-cli PRIVATE tyger)

    add_executable(tyger-quantize tools/quantize/main.cpp)
    target_link_libraries(tyger-quantize PRIVATE ggml)

    add_executable(tyger-server tools/server/main.cpp tools/server/ws.cpp)
    target_include_directories(tyger-server PRIVATE tools/server)
    target_link_libraries(tyger-server PRIVATE tyger Threads::Threads)
    if (WIN32)
        target_link_libraries(tyger-server PRIVATE ws2_32)
    endif()

    foreach(t tyger-cli tyger-quantize tyger-server)
        if (MSVC)
            target_compile_options(${t} PRIVATE /utf-8)
        endif()
    endforeach()
endif()
