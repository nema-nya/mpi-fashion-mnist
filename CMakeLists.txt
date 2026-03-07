cmake_minimum_required(VERSION 3.25)

project(rvs LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

file(GLOB_RECURSE srcs "src/*.c")
set(srcs_main ${srcs})
list(APPEND srcs_main "main.c")
add_executable(${PROJECT_NAME} ${srcs_main})

set(srcs_mpi_main ${srcs})
list(APPEND srcs_mpi_main "mpi_main.c")
add_executable(${PROJECT_NAME}_mpi ${srcs_mpi_main})

target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME}_mpi PUBLIC src)

find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MATH_LIBRARY})
    target_link_libraries(${PROJECT_NAME}_mpi PRIVATE ${MATH_LIBRARY})
endif()

find_package(MPI REQUIRED)

if(MPI_FOUND)
    target_include_directories(${PROJECT_NAME}_mpi PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME}_mpi PRIVATE ${MPI_LIBRARIES})
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

target_compile_options(${PROJECT_NAME} PUBLIC 
    $<$<CONFIG:Debug>:-O0 -g3>
    $<$<CONFIG:Release>:-O2>
)