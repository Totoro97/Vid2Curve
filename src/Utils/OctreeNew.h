//
// Created by aska on 2020/1/1.
//

#pragma once
#include "Common.h"
#include <numeric>
#include <vector>

struct OctreeNode {
  OctreeNode() = default;
  ~OctreeNode() {
    for (int i = 0; i < 8; i++) {
      if (sons_[i] != nullptr) {
        delete(sons_[i]);
      }
    }
  }
  OctreeNode* sons_[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
  Eigen::Vector3d o_;
  int u_idx_ = -1;
  double r_ = 0.0;
};

class OctreeNew {
public:
  OctreeNew(const std::vector<Eigen::Vector3d>& points);
  ~OctreeNew();
  OctreeNode* BuildTreeNode(const Eigen::Vector3d& corner, double r, int l_bound, int r_bound);
  int NearestIdx(const Eigen::Vector3d& point);
  int NearestIdx(OctreeNode* u, const Eigen::Vector3d& point, double dis);
  void SearchingR(const Eigen::Vector3d& o, double r, std::vector<int>* neighbors);
  void SearchingR(OctreeNode* u, const Eigen::Vector3d& o, double r, std::vector<int>* neighbors);
  // data.
  OctreeNode* root_ = nullptr;
  double fineness_ = 0.0;
  std::vector<Eigen::Vector3d> points_;
  std::vector<int> p_;
  Eigen::Vector3d corner_;
  int n_points_;
};
