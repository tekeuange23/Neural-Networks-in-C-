# setup ---> https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/
cmake_minimum_required(VERSION 2.6)

project(MLP)

#  Locate GTest  (this is so we can #include <gtest/gtest.h>)
if (DEFINED ENV{GITHUB_ACTIONS})
  if (WIN32)
    link_directories("D:/gtest/lib")
    include_directories("D:/gtest/include")

    set(CMAKE_CXX_FLAGS_RELEASE "/MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd")

  elseif (APPLE)
    link_directories("/usr/local/lib")
    include_directories("/usr/local/include")

  else # (UBUNTU)
    find_package(GTest REQUIRED)      
    include_directories(${GTEST_INCLUDE_DIRS})                    # this is so we can #include <gtest/gtest.h>
  
  endif ()
endif ()
 
#  Link runTests with what we want to test and  the GTest and pthread library
add_executable(runTests Test.cpp)                             # add this executable
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread)    # link google test to this executable