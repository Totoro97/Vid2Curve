//
// Created by aska on 2019/4/12.
//

#pragma once
#include <Eigen/Eigen>
#include <string>
#include "Loader.h"

using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

class SimpleRenderer {
public:
  SimpleRenderer(int height, int weigth, double focal_length);
  void CacheData();
  void CalcRays(int idx, std::vector<Vector3d> &rays, double add_error = 0.0);
  void OutputAllImages(std::string dir_path, double add_noise = 0.0);
  void OutputAllMats(std::string dir_path);
  // data
  std::vector<Vector3d> points_;
  std::vector<Matrix3d> traces_;
  PointsLoader *points_loader_;
  TracesLoader *traces_loader_;
  int height_, width_;
  int n_points_, n_traces_;
  double focal_length_;
};