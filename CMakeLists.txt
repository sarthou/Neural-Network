cmake_minimum_required (VERSION 3.1 FATAL_ERROR)


###############################################################################
# Define SimpleNeuralNetwork project
###############################################################################

project(SNN)


###############################################################################
# Set some usefull variables
###############################################################################

# Set directories
set(SNN_SRCDIR "src/")
set(SNN_INCLUDEDIR "include/")
set(EXAMPLE_SRCDIR "example/")

# Set C++ compiler options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic ${CMAKE_CXX_FLAGS}")
# Find all source files of snn library
file(GLOB_RECURSE SNN_SRC "${SNN_SRCDIR}/snn/*.cpp")


###############################################################################
# Add recipe for snn static library (creating libsnn.a)
###############################################################################

include_directories(${SNN_INCLUDEDIR})
add_library(snn STATIC ${SNN_SRC})


###############################################################################
# Add recipe for some example executables
###############################################################################

# Four bits counting example
add_executable(example_four_bits ${EXAMPLE_SRCDIR}/four_bits/four_bits.cpp)
target_link_libraries(example_four_bits snn)

# Bitmap outline example
add_executable(example_outline ${EXAMPLE_SRCDIR}/outline/outline.cpp ${EXAMPLE_SRCDIR}/outline/bitmap/bmp.cpp)
target_link_libraries(example_outline snn)

