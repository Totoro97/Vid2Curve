//
// Created by aska on 2019/4/21.
//
#pragma once
#include "Optimizer.h"
#include "CurveExtractor.h"
#include "View.h"
#include "../Utils/GlobalDataPool.h"
#include "../Utils/Graph.h"
#include "../Utils/OctreeNew.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <Eigen/Eigen>
#include <vector>

enum SolverType { LSQR, CERES, CERES_BEZIER };
enum ViewWeightDistributionType { STD_DEV_THRESHOLD, STD_DEV, UNIFORM };
enum ModelState { INITIALIZING, ITERATING };

struct ModelData {
  std::vector<CurveExtractor*> curve_extractors;
  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> camera_poses;
  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector3d> tangs;
  std::vector<double> tang_scores;
};

class Model {
public:
  Model(ModelData* model_sate, const PropertyTree &ptree, GlobalDataPool* global_data_pool);
  ~Model();
  void SetParameters(const PropertyTree &ptree);
  double Score();
  void EstimateHopeDist();
  void FeedExtractor(CurveExtractor *extractor);
  void FeedExtractor(CurveExtractor *extractor, const Eigen::Matrix3d &R, const Eigen::Vector3d &T);
  void Update();

  double UpdatePoints(bool use_linked_paths = false, Graph* graph = nullptr, Graph* tree = nullptr);
  double UpdatePointsGlobal();
  double UpdatePointsSplitAndSolve(bool use_linked_paths = false,
                                   Graph* graph = nullptr,
                                   Graph* tree = nullptr);
  double UpdatePointsSingleComponent(const std::vector<int>& indexes,
                                     const std::vector<double>& weights,
                                     std::vector<Eigen::Vector3d>* solutions);
  void GetControlIndexes(const std::vector<int>& indexes, std::vector<int>* control_indexes);
  void PathsSmoothing();
  void Update3DRadius(double extend_2d_radius = 0.0);
  void RadiusSmoothing(Graph* graph = nullptr);
  void RecoverUncertainRadius(Graph* graph);

  void GetSinglePointViewWeight(int pt_idx, std::vector<std::pair<double, View*>>* weighted_views);
  double GetSinglePoint3dRadius(int pt_idx);
  void MoveModelPointsToCenter();
  void UpdateViews();
  double MissingRatio();
  void UpdateTangs(Graph* graph = nullptr);
  void DeleteOutliers();

  // Functions for adding missing/lost points;
  void AddLostPoints();
  void AddLostPointsByDenseVoxel();
  void AddLostPointsBySearch();
  void GetDepthCandidatesByWorldRay(const Eigen::Vector3d& o,
                                    const Eigen::Vector3d& v,
                                    double d_min,
                                    double d_max,
                                    std::vector<double>* depth_candidates);
  void AddLostPointsByDepthSampling();
  double FindBestDepthByWorldRay(const Eigen::Vector3d& o,
                                 const Eigen::Vector3d& v,
                                 double initial_depth,
                                 double d_min,
                                 double d_max);

  void RefinePoints();
  void FinalProcess(const std::string& out_file_name = "curves");
  void OutputFinalModel(Graph* graph, const std::string& out_file_name = "curves");
  SelfDefinedGraph* MergeJunction();
  IronTown* RadiusBasedMergeJunction();
  void UpdateSpanningTree();
  void UpdateDataStructure(bool update_3d_radius = true);
  void UpdateOctree();
  void AdjustPoints(Graph* graph);
  bool IsTrackingGood();
  bool IsCameraMovementSufficient();
  bool IsCameraMotionSimilar(Model* ano_model);
  bool IsSinglePointFeasible(const Eigen::Vector3d& point,
                             double single_view_max_error = 2.0,
                             double ave_max_error = 1.5);
  bool IsSinglePointVisible(const Eigen::Vector3d& point);
  // Before using this fuction, pls make sure hitmap of each view has been refreshed.
  bool IsSinglePointNeeded(const Eigen::Vector3d& point);
  void ShowModelPoints();
  void ShowCameras();
  void OutputProcedureMesh();

  std::vector<Eigen::Vector3d> points_;
  std::vector<Eigen::Vector3d> tangs_;
  std::vector<int> points_history_;
  std::vector<double> tang_scores_;
  std::vector<double> points_radius_3d_;
  std::vector<int> points_feasible_;
  std::vector<std::unique_ptr<View>> view_pools_;
  std::vector<View*> views_;  // active views;
  int global_view_ticker_ = -1;
  int global_iter_num_ = 0;
  int n_points_;
  double focal_length_;
  double init_depth_;
  double hope_dist_;

  // Data structures.
  OctreeNew* octree_ = nullptr;
  GlobalDataPool* global_data_pool_ = nullptr;
  IronTown* curve_network_ = nullptr;
  SpanningTree* spanning_tree_ = nullptr;

  // For last tracking.
  Eigen::Matrix3d last_R_;
  Eigen::Vector3d last_T_;
  // state flags.
  bool is_last_view_keyframe_ = false;
  bool is_tracking_good_ = false;

  // parameters.
  double gather_weight_ = 0.0;
  double hope_dist_weight_ = 1.0;
  double final_smoothing_weight_ = 1.0;
  double shrink_error_weight_ = 1.0;
  double radius_dilation_ = 1.0;
  double radius_smoothing_weight_ = 1.0;
  double topology_searching_radius_ = 5.0;
  std::string update_points_method_ = "SPLIT_AND_SOLVE";
  std::string smooth_method_ = "LAPLACIAN";
  std::string out_curve_format_ = "SWEEP";
  std::string final_process_method_ = "RADIUS_BASED_MERGE_JUNCTION";
  std::string add_lost_points_method_ = "DP";
  bool output_procedure_mesh_ = false;
  bool is_model_feasible_ = true;
  TrackType track_type_ = TrackType::MATCHING_OF;
  SolverType solver_type_ = SolverType::CERES;
  ModelState model_state_ = ModelState::INITIALIZING;
  ViewWeightDistributionType view_weight_distribution_type_ = ViewWeightDistributionType::STD_DEV_THRESHOLD;
};

// For ceres solver.
