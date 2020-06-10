//
// Created by aska on 2019/11/9.
//

#pragma once
#include "Common.h"
#include "QuadTree.h"
#include <memory>

class ICP {
public:
  ICP(std::vector<Eigen::Vector2d>& points,
      const std::vector<Eigen::Vector3d>& normals,
      const std::vector<double>& weights);

  void FitRT(const std::vector<Eigen::Vector3d>& world_points,
             Eigen::Matrix3d* R,
             Eigen::Vector3d* T,
             int max_iter_num = 20);
  double FitRTSingleStep(const std::vector<Eigen::Vector3d>& world_points, Eigen::Matrix3d* R, Eigen::Vector3d* T);
  std::pair<Eigen::Vector3d, Eigen::Vector3d> NearestRayNormal(const Eigen::Vector2d& point);
  std::pair<Eigen::Vector3d, Eigen::Vector3d> RayNormalByIdx(int idx);
  std::unique_ptr<QuadTree> quad_tree_;
  std::vector<Eigen::Vector3d> normals_;
  std::vector<double> weights_;
  std::vector<Eigen::Vector2d> points_;
  int n_points_;
};