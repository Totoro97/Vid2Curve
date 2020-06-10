//
// Created by aska on 2019/9/11.
//

#pragma once
#include <chrono>

class StopWatch {
public:
  StopWatch();
  ~StopWatch() = default;
  double TimeDuration();
  std::chrono::steady_clock::time_point t_point_;
};
