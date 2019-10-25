//
//  ThreadSync.cpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

// C / C++
#include <stdio.h>

// External

// Project
#include "./ThreadSync.hpp"


/**************************************************************************************
 * Constructor / Destructor
 * ------------------------
 * Called on new and delete.
 **************************************************************************************/

ThreadSync::ThreadSync() noexcept : b_KillThreads(false) {
    i_Waiting = 0;
}

ThreadSync::~ThreadSync() noexcept {}

/**************************************************************************************
 * Update
 * ------
 * Update snyc.
 **************************************************************************************/

void ThreadSync::WaitForSignal() noexcept {
    // Increment waiting counter
    s_CounterMutex.lock();
    ++i_Waiting;
    s_CounterMutex.unlock();

    // Wait for signal
    std::unique_lock<std::mutex> s_Lock(s_ConditionMutex);
    s_Condition.wait(s_Lock);
}

void ThreadSync::NotifyWaiting() noexcept {
    // Lock and signal
    std::lock_guard<std::mutex> s_Guard(s_ConditionMutex);
    s_Condition.notify_all();

    // Reset waiting counter
    s_CounterMutex.lock();
    i_Waiting = 0;
    s_CounterMutex.unlock();
}

/**************************************************************************************
 * Getters
 * -------
 * ThreadSync getters.
 **************************************************************************************/

int ThreadSync::GetWaiting() noexcept {
    std::lock_guard<std::mutex> s_CounterGuard(s_CounterMutex);
    return i_Waiting;
}

bool ThreadSync::GetKillThreads() noexcept {
    return b_KillThreads;
}

/**************************************************************************************
 * Setters
 * -------
 * ThreadSync setters.
 **************************************************************************************/

void ThreadSync::SetKillThreads(bool b_Kill) noexcept {
    b_KillThreads = b_Kill;
}
