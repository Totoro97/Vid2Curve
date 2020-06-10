//
// Created by aska on 2019/10/22.
//
#pragma once

#include "Common.h"
#include "OctreeNew.h"
#include <vector>

enum GraphType {
  UNKNOWN, TREE, TREE_WITH_CIRCLE, TREE_WITH_BRANCH_LINK, RAW_GRAPH
};

class Graph {
public:
  Graph() = default;
  virtual void GetPaths(std::vector<std::vector<int>>* paths) = 0;
  // virtual void GetSmoothPaths(std::vector<std::vector<int>>* pahts);
  virtual void GetLinkedPaths(const std::vector<Eigen::Vector3d>& points,
                              std::vector<std::vector<int>>* paths,
                              double cos_thres = -0.7,
                              int max_degree = 10,
                              bool jump_junctions = true);
  virtual void GetLinkedPathsWithSameDirection(const std::vector<Eigen::Vector3d>& points,
                                               std::vector<std::vector<int>>* paths,
                                               double cos_thres = -0.7,
                                               int max_degree = 10,
                                               bool jump_junctions = true);
  // virtual void GetSmoothLikedPaths();

  int Degree(int u) {
    return edges_[u].size();
  }

  const std::vector<std::pair<int, double>>& OutPoints(int u) const {
    return edges_[u];
  }

  const std::vector<std::vector<std::pair<int, double>>>& Edges() const {
    return edges_;
  }

  std::vector<std::vector<std::pair<int, double>>> edges_;
  int n_points_;
};

class SpanningTree : public Graph {
public:
  SpanningTree(const std::vector<Eigen::Vector3d>& points,
               OctreeNew* octree,
               double r,
               double hope_dist,
               int max_degree = 4,
               const std::vector<double>* const points_radius = nullptr);
  void GetPaths(std::vector<std::vector<int>>* paths) final;
  double r_;
  double hope_dist_;
  int max_degree_;
  const std::vector<Eigen::Vector3d>* points_;
};

class IronTown : public Graph {
public:
  IronTown(const std::vector<Eigen::Vector3d>& points,
           OctreeNew* octree,
           double r,
           double hope_dist,
           const std::vector<double>* const points_radius = nullptr);
  IronTown(const std::vector<Eigen::Vector3d>& points,
           const std::vector<std::vector<std::pair<int, double>>>& edges);
  void GetPaths(std::vector<std::vector<int>>* paths) final;
  double r_;
  double hope_dist_;
  const std::vector<Eigen::Vector3d>* points_;
};

class SelfDefinedGraph : public Graph {
public:
  SelfDefinedGraph(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<std::vector<std::pair<int, double>>>& edges,
                   const std::vector<std::vector<int>>& paths);
  void GetPaths(std::vector<std::vector<int>>* paths) final;

  const std::vector<Eigen::Vector3d>* points_;
  std::vector<std::vector<int>> paths_;
};
