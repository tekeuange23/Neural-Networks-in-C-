cmake_minimum_required(VERSION 2.6)

project(MLP)

#  Locate GTest
find_package(GTest REQUIRED)      
include_directories(${GTEST_INCLUDE_DIRS})                    # this is so we can #include <gtest/gtest.h>
 
#  Link runTests with 
#  what we want to test and 
## the GTest and pthread library
add_executable(runTests tests/Test.cpp)                       # add this executable
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread)    # link google test to this executable