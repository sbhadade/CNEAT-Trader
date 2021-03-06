cmake_minimum_required(VERSION 3.10)
project(CNEAT_Trader)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

include_directories(../include ../src ${CURSES_INCLUDE_DIR})


find_package(Threads)
find_package(Curses REQUIRED)

add_executable(CNEAT_Trader
        ../include/convertcsv.hpp
        ../include/ErrorLog.hpp
        ../src/Main.cpp
        ../src/OHLCVManager.hpp
        ../src/TraderPool.hpp
        ../src/TraderPool.cpp
        ../src/EvalFunctions.cpp
        ../src/EvalFunctions.h
        ../src/ThreadSync.hpp
        ../src/ThreadSync.cpp
        ../src/cneat.cpp
        ../src/cneat.h
        ../src/cann.cpp
        ../src/cann.h)

target_link_libraries(CNEAT_Trader ${CMAKE_THREAD_LIBS_INIT} ${CURSES_LIBRARIES})
