//
// Created by aska on 2019/4/21.
//
#pragma once

#include <Eigen/Eigen>
// #include <line3D.h>
// #include <icpPointToPlane.h>
#include "CurveExtractor.h"
#include "../Utils/Camera.h"
#include "../Utils/GlobalDataPool.h"
#include "../Utils/Graph.h"
#include "../Utils/ICP.h"
#include "../Utils/OctreeNew.h"
#include "../Utils/SegmentQuadTree.h"

#include <unordered_map>
#include <unordered_set>

enum TrackType { NAIVE, MATCHING, MATCHING_OF };

class View {
public:
  View(CurveExtractor *extractor, int time_stamp, double focal_length);
  void UpdateWorldPoints(std::vector<Eigen::Vector3d> *points);
  void UpdateMatching();
  int  Track(const std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>>& pose_candidates,
             const std::vector<Eigen::Vector3d>& points,
             const std::vector<int>& points_history,
             const std::vector<Eigen::Vector3d>& tangs,
             const std::vector<double>& tang_scores,
             IronTown* curve_network,
             SpanningTree* spanning_tree,
             OctreeNew* octree,
             Eigen::Vector2d initial_flow = Eigen::Vector2d::Zero(),
             TrackType track_type = TrackType::NAIVE);
  int TrackSingleStep(const std::vector<Eigen::Vector3d>& points,
                      const std::vector<int>& tracking_indexes,
                      const std::vector<int>& current_matching_indexes);
  void UpdateRTRobust(const std::vector<Eigen::Vector3d>& points,
                      const std::vector<int>& points_history);
  void UpdateRT(const std::vector<Eigen::Vector3d>& points,
                const std::vector<int>& points_history,
                bool show_demo = false,
                int max_iter_num = 100);
  void UpdateRTCeres(const std::vector<Eigen::Vector3d>& points,
                     const std::vector<int>& points_history,
                     int max_iter_num = 100);
  void UpdateRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& T);
  bool BadTrackingResult(const std::vector<Eigen::Vector3d>& points);

  void FindPointsInFOV(const std::vector<Eigen::Vector3d>& points,
                       double ratio,
                       std::vector<int>* indexes);

  int GetNearestPointIdx(const Eigen::Vector3d& world_point);
  std::pair<Eigen::Vector3d, Eigen::Vector3d> GetRayAndNormal(const Eigen::Vector3d &point);
  std::pair<Eigen::Vector3d, Eigen::Vector3d> GetWorldRayAndNormal(const Eigen::Vector3d &point);
  Eigen::Vector2d GetTang(const Eigen::Vector3d &point);

  void BuildGridIndex();

  bool MatchingCached();
  void ClearMatchingCache();
  void ClearSelfOcclusionState();
  void CacheMatchings(const std::vector<Eigen::Vector3d>& world_points,
                      const std::vector<Eigen::Vector3d>& tangs,
                      const std::vector<double>& tang_scores,
                      const std::vector<std::vector<std::pair<int, double>>>& edges,
                      double hope_dist,
                      bool use_initial_alignment,
                      Eigen::Vector2d initial_flow = Eigen::Vector2d::Zero(),
                      bool show_debug_info = false);
  void CopyWorldPointMatching(int u, int new_n_points);

  std::pair<Eigen::Vector3d, Eigen::Vector3d> GetMatching(int world_point_idx);
  std::pair<Eigen::Vector3d, Eigen::Vector3d> GetWorldRayByIdx(int idx);
  Eigen::Vector2d GetMatchingPixels(int world_point_idx);

  double GetMatchingRadius(int world_point_idx);

  void OutVisualizationInfo(const std::string& dir_path = "./views_information/");

  void OutDebugImage(const std::string& img_name,
                     const std::vector<Eigen::Vector3d>& points,
                     bool show_or_write = false,
                     int wait_time = 10);
  void OutDebugImage(const std::string& mark,
                     const std::vector<Eigen::Vector3d>& points,
                     GlobalDataPool* global_data_pool);
  cv::Mat OutDebugImageCore(const std::vector<Eigen::Vector3d>& points);

  // void AddMissingImg(int idx, L3DPP::Line3D *line3d, const std::vector<Eigen::Vector3d> &points);

  double AverageNearestProjectionError(const std::vector<Eigen::Vector3d>& points);
  double CalcBidirectionProjectionError(const std::vector<Eigen::Vector3d> &points);
  double CalcIOU(const std::vector<Eigen::Vector3d> &points);

  void UpdateMissingQuadTree();
  void GetMissingWorldRays(const std::vector<Eigen::Vector3d>& points,
                           std::vector<std::pair<Eigen::Vector3d, double>>* world_rays);
  void GetMissingPaths(std::vector<std::vector<int>>* missing_paths,
                       int too_short_path_threshold = 5);
  void GetDepthIntersectionsByWorldRay(const Eigen::Vector3d& o,
                                       const Eigen::Vector3d& v,
                                       double d_min,
                                       double d_max,
                                       std::vector<std::pair<double, double>>* depth_intersections);
  void RefreshCoveredMap();
  bool SinglePointHitMap(const Eigen::Vector3d& world_point);
  double CalcCoveredPixelsRatio(const std::vector<Eigen::Vector3d>& world_points, const std::vector<int>& path_indexes);
  bool WorldPointOutImageRange(const Eigen::Vector3d& point, double ratio = 0.01);

  // Data
  std::vector<Eigen::Vector3d>* world_points_;
  // Index as world points.
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matching_ray_normals_;
  std::vector<Eigen::Vector2d> matching_pixels_;
  std::vector<int> matching_indexes_;
  std::vector<double> matching_radius_;
  std::vector<int> is_world_point_self_occluded_;
  // Index as view points.
  std::vector<double> matched_std_dev_;
  std::vector<int> nearest_matched_idx_;
  std::vector<int> matched_;
  std::vector<int> is_self_occluded_;

  int n_points_;
  int time_stamp_;
  double focal_length_;
  double average_depth_ = 1.0;
  std::unique_ptr<double[]> M_;
  // std::unique_ptr<IcpPointToPlane> icp_;
  std::unique_ptr<ICP> icp_;
  std::unique_ptr<QuadTree> quad_tree_;
  std::unique_ptr<QuadTree> missing_quad_tree_;
  std::unique_ptr<SegmentQuadTree> segment_quad_tree_;
  CurveExtractor *extractor_;
  Camera camera_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d T_;
  
  // Data structure for grid index.
  std::unordered_map<unsigned, std::vector<int>> grids_;
  unsigned grid_w_;
  unsigned grid_h_;

  // Data structure for single pixel matching cache.
  std::unordered_map<unsigned, std::vector<int>> single_pixel_cache_;

  // Data structure for covered map.
  // std::unordered_set<int> covered_map_;
  std::vector<int> covered_;

  // Some configures.
  bool emphasize_missing_paths_ = true;

  // States.
  bool is_necessary_ = false;

  Eigen::Vector2d view_center_;
};
