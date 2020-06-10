//
// Created by aska on 2019/9/11.
//

#include "StopWatch.h"

StopWatch::StopWatch() {
  t_point_ = std::chrono::steady_clock::now();
}

double StopWatch::TimeDuration() {
  std::chrono::steady_clock::time_point new_point = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(new_point - t_point_);
  t_point_ = new_point;
  return time_span.count();
}