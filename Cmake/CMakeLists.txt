cmake_minimum_required(VERSION 3.10)
project(CNEAT_Trader)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

include_directories(../include ../src)


find_package(Threads)

add_executable(CNEAT_Trader
        ../include/CANN.hpp
        ../include/CNEAT.hpp
        ../include/convertcsv.hpp
        ../include/ErrorLog.hpp
        ../src/Main.cpp
        ../src/OHLCVManager.hpp
        ../src/TraderPool.hpp
        ../src/EvalFunctions.cpp
        ../src/EvalFunctions.hpp
        ../src/ThreadSync.hpp ../src/cneat.cpp ../src/cneat.h)

target_link_libraries(CNEAT_Trader ${CMAKE_THREAD_LIBS_INIT}
        )
