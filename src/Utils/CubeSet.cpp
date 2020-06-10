//
// Created by aska on 2019/7/20.
//

#include "CubeSet.h"

#include <algorithm>
#include <iostream>

CubeSet::CubeSet(const Eigen::Vector3d &center, double r, double fineness)
  : center_(center), r_(r), fineness_(fineness) {
}

void CubeSet::UpdateCurrentSet() {
  for (const auto& pr : st_) {
    if (!mp_.count(pr.first)) {
      mp_.emplace(pr.first, pr.second);
    }
    else {
      mp_[pr.first] += pr.second;
    }
  }
  st_.clear();
}

void CubeSet::CalcAllCubes(const Eigen::Vector3d& o, const Eigen::Vector3d& v, std::vector<long long>* cubes) {
  cubes->clear();
  double in_times[3] = {
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity() };
  double out_times[3] = {
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity() };

  const double eps = 1e-9;
  for (int t = 0; t < 3; t++) {
    if (std::abs(v(t)) < eps) {
      if (o(t) >= center_(t) - r_ && o(t) <= center_(t) + r_) {
        in_times[t] = -std::numeric_limits<double>::infinity();
        out_times[t] = std::numeric_limits<double>::infinity();
      }
    }
    else {
      in_times[t]  = (center_(t) - r_ - o(t)) / v(t);
      out_times[t] = (center_(t) + r_ - o(t)) / v(t);
      if (v(t) < 0.0) {
        std::swap(in_times[t], out_times[t]);
      }
    }
  }

  double in_time = -std::numeric_limits<double>::infinity();
  double out_time = std::numeric_limits<double>::infinity();
  for (int t = 0; t < 3; t++) {
    in_time =  std::max(in_time, in_times[t]);
    out_time = std::min(out_time, out_times[t]);
  }

  Eigen::Vector3d corner = center_ - Eigen::Vector3d(r_, r_, r_);
  double step_len = fineness_ / v.norm();
  long long x = -(1LL << 20);
  long long y = -(1LL << 20);
  long long z = -(1LL << 20);
  for (double current_t = std::max(0.0, in_time); current_t <= out_time; current_t += step_len) {
    Eigen::Vector3d current_pt = o + v * current_t;
    long long new_x = std::floor((current_pt(0) - corner(0) + 1e-5) / fineness_);
    long long new_y = std::floor((current_pt(1) - corner(1) + 1e-5) / fineness_);
    long long new_z = std::floor((current_pt(2) - corner(2) + 1e-5) / fineness_);
    if (new_x == x && new_y == y && new_z == z) {
      continue;
    }
    x = new_x;
    y = new_y;
    z = new_z;
    if (x < 0 || y < 0 || z < 0) {
      LOG(FATAL) << "[CubeSet]: Error: Negative index.";
    }
    cubes->emplace_back((x << 40) + (y << 20) + z);
  }
}

void CubeSet::AddRay(const Eigen::Vector3d &o, const Eigen::Vector3d &v, double weight) {
  rays_.emplace_back(o, v);
  std::vector<long long> cubes;
  CalcAllCubes(o, v, &cubes);
  for (long long cube : cubes) {
    if (!st_.count(cube)) {
      st_.emplace(cube, weight);
    } else {
      st_[cube] = std::max(st_[cube], weight);
    }
  }
}

void CubeSet::FindDenseCubes(double cnt_threshold, std::vector<std::pair<int, Eigen::Vector3d>> &cubes) {
  cubes.clear();
  Eigen::Vector3d corner = center_ - Eigen::Vector3d(r_, r_, r_);
  long long mask = (1LL << 20) - 1;
  const int kCounterInf = 1e9;
  for (const auto& ray : rays_) {
    std::vector<long long> cube_indexes;
    CalcAllCubes(ray.first, ray.second, &cube_indexes);
    long long cube_to_add = -1;
    double counter = -1;
    for (long long cube_index : cube_indexes) {
      double current_counter = mp_[cube_index];
      if (current_counter > counter) {
        counter = current_counter;
        cube_to_add = cube_index;
      }
    }
    if (counter + 1.0 < kCounterInf && counter > cnt_threshold) {
      mp_[cube_to_add] = kCounterInf;
      Eigen::Vector3d pt((cube_to_add >> 40) & mask, (cube_to_add >> 20) & mask, cube_to_add & mask);
      pt = (pt + Eigen::Vector3d(0.5, 0.5, 0.5)) * fineness_ + corner;
      cubes.emplace_back(counter, pt);
    }
  }
}