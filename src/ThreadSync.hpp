//
//  ThreadSync.hpp
//  CNT
//
//  Created by Liam Briegel on 25.10.19.
//  Copyright © 2019 Liam Briegel. All rights reserved.
//

#ifndef ThreadSync_hpp
#define ThreadSync_hpp


// C / C++
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>

// External

// Project


class ThreadSync {
public:

    /**************************************************************************************
     * Constructor / Destructor
     **************************************************************************************/

    /**
     *  Default constructor.
     */

    ThreadSync() noexcept;

    /**
     *  Default destructor.
     */

    ~ThreadSync() noexcept;

    /**************************************************************************************
     * Update
     **************************************************************************************/

    /**
     *  Catches a thread an waits for a notification.
     */

    void WaitForSignal() noexcept;

    /**
     *  Notify all waiting threads to start running.
     */

    void NotifyWaiting() noexcept;

    /**************************************************************************************
     * Getters
     **************************************************************************************/

    /**
     *  Get the amount of threads waiting.
     */

    int GetWaiting() noexcept;

    /**
     *  Check if all spawned threads should be killed.
     */

    bool GetKillThreads() noexcept;

    /**************************************************************************************
     * Setters
     **************************************************************************************/

    /**
     *  Set if all spawned threads should be killed.
     */

    void SetKillThreads(bool b_Kill) noexcept;

private:

    /**************************************************************************************
     * Data
     **************************************************************************************/

    // Thread
    std::mutex s_CounterMutex;
    std::mutex s_ConditionMutex;
    std::condition_variable s_Condition;
    int i_Waiting;
    std::atomic<bool> b_KillThreads;

protected:

};


#endif /* ThreadSync_hpp */
