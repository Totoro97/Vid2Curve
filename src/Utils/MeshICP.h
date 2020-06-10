//
// Created by aska on 2020/1/1.
//

#ifndef ARAP_MESHICP_H
#define ARAP_MESHICP_H

#include "Common.h"
#include "OctreeNew.h"
#include <tuple>
#include <vector>

struct MeshEvaluationResult {
  double average_distance;
  double max_distance;
  double global_scale;
};

class MeshICP {
public:
  MeshICP(const std::vector<Eigen::Vector3d>& my_points,
          const std::vector<std::tuple<int, int, int>>& my_faces,
          const std::vector<Eigen::Vector3d>& gt_points,
          const std::vector<std::tuple<int, int, int>>& gt_faces);
  MeshEvaluationResult Evaluate(std::vector<double>* points_distances = nullptr);
  void FitRT();
  double FitRTSingleStep();
  double FitScaleSingleStep();
  std::vector<Eigen::Vector3d> my_points_, gt_points_;
  std::vector<std::tuple<int, int, int>> my_faces_, gt_faces_;
  std::vector<std::vector<std::tuple<int, int, int>>> gt_faces_at_points_;
  std::vector<std::vector<Eigen::Vector3d>> gt_normals_;
  int sum_normals_;
  std::unique_ptr<OctreeNew> octree_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d T_;
  double scale_ = 1.0;
};

#endif //ARAP_MESHICP_H