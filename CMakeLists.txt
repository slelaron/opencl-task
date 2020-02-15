cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(opencl-task)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OpenMP not found!")
endif()

convertIntoHeader(src/cl/multiply.cl src/cl/multiply_cl.h multiply)
add_executable(opencl-task src/main.cpp src/cl/multiply_cl.h)
target_link_libraries(opencl-task libclew libgpu libutils)
