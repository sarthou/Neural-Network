cmake_minimum_required (VERSION 2.6)

project(SNN)

set(SNN_SRCDIR "src/")
set(SNN_INCLUDEDIR "include/")
set(EXAMPLE_SRCDIR "example/")
set(REQUIRED_CXX_STANDARD_VERSION 11)

file(GLOB_RECURSE SNN_SRC "${SNN_SRCDIR}/snn/*.cpp")

include_directories(${SNN_INCLUDEDIR})

add_library(snn STATIC ${SNN_SRC})
set_target_properties(snn PROPERTIES CXX_STANDARD ${REQUIRED_CXX_STANDARD_VERSION})

add_executable(example_four_bits ${EXAMPLE_SRCDIR}/four_bits/four_bits.cpp)
target_link_libraries(example_four_bits snn)
set_target_properties(example_four_bits PROPERTIES CXX_STANDARD ${REQUIRED_CXX_STANDARD_VERSION})

