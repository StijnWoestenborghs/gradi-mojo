#pragma once
#include <chrono>
#include <iostream>

/*
Timing class:
Create the Timer object inside a scope:
    Basically, when the timer gets created start the timer
    When the timer gets destroyed (desctructor) stop the timer --> end of scope
*/
class Timer
{
public:
    Timer()
    {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        Stop();
    }

    void Stop()
    {
        auto endTimepoint = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();
        auto duration = end - start;
        double ms = duration * 0.001;

        std::cout << duration << "µs (" << ms << "ms)\n";

    }
private:
    std::chrono::time_point< std::chrono::high_resolution_clock> m_StartTimepoint;
};