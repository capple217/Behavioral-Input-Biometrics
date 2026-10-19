cmake_minimum_required(VERSION 3.20)
project(BehavioralInputBiometrics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)
option(BIB_TSAN "Build tests with ThreadSanitizer" OFF)

find_package(Threads REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Consumer-side shim loaded by Python via ctypes (libbibring.dylib / .so)
add_library(bibring SHARED src/bibring.cpp)
if(NOT APPLE)
    target_link_libraries(bibring PRIVATE rt)
endif()

# Portable producer for tests and session replay
add_executable(fake_producer src/fake_producer.cpp)
target_link_libraries(fake_producer PRIVATE Threads::Threads)
if(NOT APPLE)
    target_link_libraries(fake_producer PRIVATE rt)
endif()

# The real sensor: macOS only
if(APPLE)
    add_executable(input_sensor src/input_sensor.cpp)
    target_link_libraries(input_sensor PRIVATE Threads::Threads "-framework ApplicationServices" "-framework CoreFoundation")
endif()

# Queue tests (+ optional comparison against Boost.Lockfree, header-only)
add_executable(spsc_queue_test tests/spsc_queue_test.cpp)
target_link_libraries(spsc_queue_test PRIVATE Threads::Threads)
find_path(BOOST_LOCKFREE_INCLUDE boost/lockfree/spsc_queue.hpp HINTS /opt/homebrew/include /usr/local/include)
if(BOOST_LOCKFREE_INCLUDE)
    target_include_directories(spsc_queue_test PRIVATE ${BOOST_LOCKFREE_INCLUDE})
    target_compile_definitions(spsc_queue_test PRIVATE BIB_HAVE_BOOST)
endif()
if(BIB_TSAN)
    target_compile_options(spsc_queue_test PRIVATE -fsanitize=thread -g)
    target_link_options(spsc_queue_test PRIVATE -fsanitize=thread)
endif()

enable_testing()
add_test(NAME spsc_queue COMMAND spsc_queue_test 2000000)
