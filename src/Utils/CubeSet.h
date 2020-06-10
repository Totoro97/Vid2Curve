//
// Created by aska on 2019/7/20.
//

#ifndef ARAP_CUBESET_H
#define ARAP_CUBESET_H

#include "Common.h"

#include <unordered_map>
#include <unordered_set>

struct CubeNode {
public:
  CubeNode() = default;
  int counter_ = 0;
  std::vector<int> source_indexes_;
};

class CubeSet {
public:
  CubeSet(const Eigen::Vector3d &center, double r, double fineness);
  ~CubeSet() = default;
  void UpdateCurrentSet();
  void CalcAllCubes(const Eigen::Vector3d& o, const Eigen::Vector3d& v, std::vector<long long>* cubes);
  void AddRay(const Eigen::Vector3d& o, const Eigen::Vector3d& v, double weight = 1.0);
  void FindDenseCubes(double cnt_threshold, std::vector<std::pair<int, Eigen::Vector3d>>& cubes);
  // data
  Eigen::Vector3d center_;
  double r_;
  double fineness_;
  int global_timestamp_;
  std::unordered_map<long long, double> st_;
  std::unordered_map<long long, double> mp_;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays_;
};


#endif //ARAP_CUBESET_H