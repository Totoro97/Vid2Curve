//
// Created by aska on 2019/10/19.
//
#pragma once
#include "Common.h"

#include <vector>

struct QuadTreeNode {
  QuadTreeNode() = default;
  ~QuadTreeNode() {
    for (int i = 0; i < 4; i++) {
      if (sons_[i] != nullptr) {
        delete(sons_[i]);
      }
    }
  }
  QuadTreeNode* sons_[4] = { nullptr, nullptr, nullptr, nullptr };
  Eigen::Vector2d o_;
  int u_idx_ = -1;
  double r_ = 0.0;
};

class QuadTree {
public:
  QuadTree(const std::vector<Eigen::Vector2d>& points);
  ~QuadTree();
  QuadTreeNode* BuildTreeNode(const Eigen::Vector2d& corner, double r, int l_bound, int r_bound);
  Eigen::Vector2d NearestPoint(const Eigen::Vector2d& point);
  int NearestIdx(const Eigen::Vector2d& point);
  int NearestIdx(QuadTreeNode* u, const Eigen::Vector2d& point, double dis);
  void SearchingR(const Eigen::Vector2d& o, double r, std::vector<int>* neighbors);
  void SearchingR(QuadTreeNode* u, const Eigen::Vector2d& o, double r, std::vector<int>* neighbors);
  // data.
  QuadTreeNode* root_ = nullptr;
  double fineness_ = 0.0;
  std::vector<Eigen::Vector2d> points_;
  std::vector<int> p_;
  Eigen::Vector2d corner_;
  int n_points_;
};