//
// Created by aska on 2019/12/13.
//
#pragma once
#include "Common.h"

struct SegmentQuadTreeNode {
  SegmentQuadTreeNode(const Eigen::Vector2d& o, double r) : o_(o), r_(r) {};
  Eigen::Vector2d o_;
  double r_ = -1.0;
  SegmentQuadTreeNode* sons_[4] = { nullptr, nullptr, nullptr, nullptr };
  std::vector<int> indexes_;
};

class SegmentQuadTree {
public:
  SegmentQuadTree(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& segments,
                  const std::vector<Eigen::Vector2d>& tangs);
  void InsertSegment(SegmentQuadTreeNode* u, int seg_idx);
  void FindIntersections(const Eigen::Vector2d& o,
                         const Eigen::Vector2d& v,
                         double t_min,
                         double t_max,
                         std::vector<std::pair<double, Eigen::Vector2d>>* intersections);
  void FindIntersections(SegmentQuadTreeNode* u,
                         const Eigen::Vector2d& o,
                         const Eigen::Vector2d& v,
                         double t_min,
                         double t_max,
                         std::vector<std::pair<double, Eigen::Vector2d>>* intersections);
  double FindIntersectionsTwoSegment(const Eigen::Vector2d& o,
                                     const Eigen::Vector2d& v,
                                     double t_min,
                                     double t_max,
                                     const std::pair<Eigen::Vector2d, Eigen::Vector2d>& segment);
  bool IsSegmentInsideBox(const Eigen::Vector2d& o,
                          double r,
                          const std::pair<Eigen::Vector2d, Eigen::Vector2d>& segment);
  bool IsSegmentIntersectBox(const Eigen::Vector2d& o,
                             double r,
                             const std::pair<Eigen::Vector2d, Eigen::Vector2d>& segment);
  bool IsSegmentIntersectBox(const Eigen::Vector2d& o,
                             double r,
                             const Eigen::Vector2d& seg_o,
                             const Eigen::Vector2d& seg_v,
                             double t_min,
                             double t_max);
  std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> segments_;
  std::vector<Eigen::Vector2d> tangs_;
  std::vector<std::unique_ptr<SegmentQuadTreeNode>> node_pool_;
  SegmentQuadTreeNode* root_;
};