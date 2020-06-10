//
// Created by aska on 2019/4/21.
//
// This is new
#include <chrono>
#include <numeric>
#include <string>
#include <thread>
#include <set>

#include "Model.h"
#include "CurveMatcher.h"
#include "../Utils/BezierCurve.h"
#include "../Utils/Common.h"
#include "../Utils/CubeSet.h"
#include "../Utils/SweepSurface.h"
#include "../Utils/Utils.h"

namespace {
  const int kMaxKeyFrameN = 30;
  const int kSlidingWindowN = 5;
}

Model::Model(ModelData* model_state, const PropertyTree& ptree, GlobalDataPool* global_data_pool) {
  SetParameters(ptree);
  global_data_pool_ = global_data_pool;
  points_ = model_state->points;
  n_points_ = points_.size();
  points_history_ = std::vector<int>(n_points_, 0);
  UpdateOctree();
  tangs_ = model_state->tangs;
  tang_scores_ = model_state->tang_scores;
  UpdateTangs();

  int extractors_size = model_state->curve_extractors.size();
  CHECK_EQ(extractors_size, model_state->camera_poses.size());
  for (int i = 0; i < extractors_size; i++) {
    FeedExtractor(model_state->curve_extractors[i],
                  model_state->camera_poses[i].first,
                  model_state->camera_poses[i].second);
  }
  MoveModelPointsToCenter();
  ShowModelPoints();
}

Model::~Model() {
  delete curve_network_;
  delete spanning_tree_;
  delete octree_;
}

void Model::SetParameters(const PropertyTree &ptree) {
  focal_length_ = ptree.get<double>("Global.FocalLength");
  init_depth_ = ptree.get<double>("Global.InitDepth");
  hope_dist_weight_ = ptree.get<double>("Global.IteratingSmoothingWeight");
  final_smoothing_weight_ = ptree.get<double>("Global.FinalSmoothingWeight");
  radius_smoothing_weight_ = ptree.get<double>("Global.RadiusSmoothingWeight");
  shrink_error_weight_ = ptree.get<double>("Global.ShrinkErrorWeight");
  radius_dilation_ = ptree.get<double>("Global.RadiusDilation");
  smooth_method_ = ptree.get<std::string>("Model.SmoothMethod");
  final_process_method_ = ptree.get<std::string>("Model.FinalProcessMethod");
  add_lost_points_method_ = ptree.get<std::string>("Model.AddLostPointsMethod");
  output_procedure_mesh_ = ptree.get<bool>("Model.OutputProcedureMesh");
  out_curve_format_ = ptree.get<std::string>("Model.OutCurveFormat");
  hope_dist_ = 2.0 / focal_length_;

  std::string view_weight_type = ptree.get<std::string>("Model.ViewWeightDistributionType");
  if (view_weight_type == "STD_DEV_THRESHOLD") {
    view_weight_distribution_type_ = ViewWeightDistributionType::STD_DEV_THRESHOLD;
  }
  else if (view_weight_type == "STD_DEV") {
    view_weight_distribution_type_ = ViewWeightDistributionType::STD_DEV;
  }
  else if (view_weight_type == "UNIFORM") {
    view_weight_distribution_type_ = ViewWeightDistributionType::UNIFORM;
  }
  else {
    LOG(FATAL) << "No such view weight distribution type.";
  }

  std::string solver_type = ptree.get<std::string>("Model.SolverType");
  if (solver_type == "LSQR") {
    solver_type_ = SolverType::LSQR;
  }
  else if (solver_type == "CERES") {
    solver_type_ = SolverType::CERES;
  }
  else if (solver_type == "CERES_BEZIER") {
    LOG(INFO) << "This is the optimized implementation for less execution time.";
    LOG(FATAL) << "This part is still under developing.";
    solver_type_ = SolverType::CERES_BEZIER;
  }
  else {
    LOG(FATAL) << "No such solver type.";
  }
  topology_searching_radius_ = 5.0;
}

double Model::Score() {
  UpdateDataStructure();
  int view_idx = 0;
  int n_views = views_.size();
#pragma omp parallel for private(view_idx) default(none) shared(n_views, points_, tangs_, tang_scores_, hope_dist_, views_, spanning_tree_)
  for (view_idx = 0; view_idx < n_views; view_idx++) {
    const auto& view = views_[view_idx];
    view->CacheMatchings(points_, tangs_, tang_scores_, spanning_tree_->Edges(), hope_dist_, false);
  }

  double sum_dis = 0.0;
  double max_dis = 3.0;
  for (int u = 0; u < n_points_; u++) {
    std::vector<std::pair<double, View*>> valid_views;
    GetSinglePointViewWeight(u, &valid_views);
    double current_dis = 0;
    for (const auto& pr : valid_views) {
      View* view = pr.second;
      if (view->matching_indexes_[u] >= 0) {
        current_dis += pr.first * (view->GetMatchingPixels(u) - view->camera_.World2Image(points_[u])).norm();
      }
    }
    sum_dis += current_dis;
    max_dis = std::max(max_dis, current_dis);
  }
  double valid_dis = sum_dis / n_points_;
  double missing_ratio = MissingRatio();
  if (missing_ratio > 0.8) {
    return -1e9;
  }
  double estimated_dis = valid_dis * (1.0 - missing_ratio) + max_dis * 1.1 * missing_ratio;
  return -estimated_dis;
}

void Model::FeedExtractor(CurveExtractor *extractor) {
  StopWatch stop_watch;
  global_view_ticker_++;
  auto new_view = new View(extractor, global_view_ticker_, focal_length_);
  LOG(INFO) << "Build new view time: " << stop_watch.TimeDuration();
  UpdateDataStructure();
  LOG(INFO) << "Update data structure time: " << stop_watch.TimeDuration();
  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pose_candidates;
  pose_candidates.emplace_back(last_R_, last_T_);
  CHECK_GT(views_.size(), 0);
  if (views_.size() >= 2) {
    auto view = views_.back();
    auto past_view = *(std::prev(views_.end()));
    pose_candidates.emplace_back(view->R_ * view->R_ * past_view->R_.inverse(), view->T_ * 2.0 - past_view->T_);
  }

  int track_quality = -1;
  for (auto initial_flow :
      std::vector<Eigen::Vector2d>{ Eigen::Vector2d::Zero(), new_view->view_center_ - views_.back()->view_center_ }) {
    track_quality = new_view->Track(pose_candidates,
                                    points_,
                                    points_history_,
                                    tangs_,
                                    tang_scores_,
                                    curve_network_,
                                    spanning_tree_,
                                    octree_,
                                    initial_flow,
                                    track_type_);
    if (track_quality >= 0) {
      break;
    }
  }
  is_tracking_good_ = (track_quality >= 0);
  LOG(INFO) << "Tracking time: " << stop_watch.TimeDuration();
  // views_.back()->OutDebugImage("last", points_, global_data_pool_);

  // LOG(INFO) << "Core time: " << stop_watch.TimeDuration();

  // new_view->UpdateRTCeres(points_, points_history_);

  // new_view->UpdateRTRobust(points_, points_history_);
  // new_view->OutDebugImage("debug", points_, true, 10);
  // new_view->OutDebugImage("last", points_, global_data_pool_);
  last_R_ = new_view->R_;
  last_T_ = new_view->T_;
  view_pools_.emplace_back(new_view);
  views_.emplace_back(new_view);
  LOG(INFO) << "Feed new extractor time: " << stop_watch.TimeDuration();
}

void Model::FeedExtractor(CurveExtractor *extractor, const Eigen::Matrix3d &R, const Eigen::Vector3d &T) {
  StopWatch stop_watch;
  global_view_ticker_++;
  auto new_view = new View(extractor, global_view_ticker_, focal_length_);
  new_view->UpdateRT(R, T);

  // TODO: Use global structure.
  /*UpdateDataStructure();
  new_view->CacheMatchings(
      points_, tangs_, tang_scores_, spanning_tree_->Edges(), hope_dist_, true, false);

  // new_view->UpdateRTCeres(points_, points_history_);
  new_view->UpdateRTRobust(points_, points_history_);*/
  last_R_ = new_view->R_;
  last_T_ = new_view->T_;
  // LOG(INFO) << "OutDebugImage";
  // new_view->OutDebugImage("debug", points_, true, 10);
  // new_view->OutDebugImage("last", points_, global_data_pool_);
  view_pools_.emplace_back(new_view);
  views_.emplace_back(new_view);
  // LOG(INFO) << "Feed new extractor time: " << stop_watch.TimeDuration();
}

// global iteration.
void Model::Update() {
  StopWatch stop_watch_per_update;
  // Update Model.
  const int max_iter_num = views_.size() < 5 ? 1 : 5;
  UpdateOctree();
  if (views_.size() > ::kSlidingWindowN * 2) {
    AddLostPoints();
  }
  for (const auto& view : views_) {
    view->ClearSelfOcclusionState();
  }
  UpdateDataStructure();
  CHECK(curve_network_ != nullptr);
  CHECK(spanning_tree_ != nullptr);

  double variance = 1e9;
  double last_variance = 1e10;
  double max_variance = 0.0;

  for (global_iter_num_ = 0;
       variance > hope_dist_ * 0.25 && variance > max_variance * 0.5 && global_iter_num_ < max_iter_num; global_iter_num_++) {
    StopWatch stop_watch;
    last_variance = variance;
    variance = UpdatePoints(false, (Graph*) curve_network_, (Graph*) spanning_tree_);
    // variance = UpdatePoints(false);
    // global_data_pool_->UpdatePlottingData({variance / hope_dist_, AveBidirectionProjectionError()});
    LOG(INFO) << "---> UpdatePoints time: " << stop_watch.TimeDuration();
    UpdateTangs();
    LOG(INFO) << "---> UpdateTangs time: " << stop_watch.TimeDuration();
    UpdateViews();
    LOG(INFO) << "---> UpdateViews time: " << stop_watch.TimeDuration();

    max_variance = std::max(max_variance, variance);
  }
  global_iter_num_ = 0;

  StopWatch stop_watch;
  AdjustPoints((Graph*) curve_network_);
  LOG(INFO) << "---> AdjustPoints time: " << stop_watch.TimeDuration();

  // Clear matching cache;
  for (const auto& view : views_) {
    view->ClearSelfOcclusionState();
  }

  for (const auto& view : views_) {
    view->matching_ray_normals_.clear();
  }
  DeleteOutliers();

  MoveModelPointsToCenter();
  ShowModelPoints();
  if (output_procedure_mesh_) {
    OutputProcedureMesh();
  }
  LOG(INFO) << "---> Duration per update: " << stop_watch_per_update.TimeDuration();
}

void Model::ShowModelPoints() {
#ifdef USE_GUI
  if (global_data_pool_ == nullptr) {
    return;
  }
  // Output points for visualization & debugging.
  bool save_to_file = false;
  if (save_to_file) {
    Utils::SavePointsAsPly("points.ply", points_);
  }
  else {
    global_data_pool_->UpdateModelPoints((void*) this, points_);
  }

  ShowCameras();
#endif
}

void Model::ShowCameras() {
#ifdef USE_GUI
  if (global_data_pool_ == nullptr) {
    return;
  }
  std::vector<Eigen::Matrix3d> K_invs;
  std::vector<Eigen::Matrix4d> Wfs;
  std::vector<std::pair<int, int>> width_and_heights;
  for (const auto& view : views_) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
    K(0, 0) = K(1, 1) = view->focal_length_;
    K(0, 2) = view->extractor_->width_ * 0.5;
    K(1, 2) = view->extractor_->height_ * 0.5;
    K(2, 2) = 1.0;

    Eigen::Matrix4d Wf = Eigen::Matrix4d::Zero();
    Wf.block(0, 0, 3, 3) = view->R_;
    Wf.block(0, 3, 3, 1) = view->T_;
    Wf(3, 3) = 1.0;

    K_invs.emplace_back(K.inverse());
    Wfs.emplace_back(Wf.inverse());
    width_and_heights.emplace_back(view->extractor_->width_, view->extractor_->height_);
  }
  global_data_pool_->UpdateCameras((void*) this, K_invs, Wfs, width_and_heights);
#endif
}

void Model::OutputProcedureMesh() {
  Update3DRadius();
  auto graph = std::make_unique<IronTown>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_);
  OutputFinalModel(graph.get(), std::string("ProcedureMesh/") + std::to_string(views_.back()->time_stamp_));

  // Output all view information
  std::vector<View*> tmp_views_;
  auto view_pool_iter = view_pools_.begin();
  for (const auto& active_view : views_) {
    CHECK(view_pool_iter != view_pools_.end());
    while (view_pool_iter->get() != active_view) {
      CHECK(!tmp_views_.empty());
      View* current_view = view_pool_iter->get();
      std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pose_candidates;
      // static.
      pose_candidates.emplace_back(tmp_views_.back()->R_, tmp_views_.back()->T_);
      current_view->
          Track(pose_candidates, points_, points_history_, tangs_, tang_scores_, curve_network_, spanning_tree_, octree_);
      // current_view->UpdateRT(tmp_views_.back()->R_, tmp_views_.back()->T_);
      // current_view->UpdateRTRobust(points_);
      tmp_views_.emplace_back(current_view);
      view_pool_iter++;
      CHECK(view_pool_iter != view_pools_.end());
    }
    tmp_views_.emplace_back(view_pool_iter->get());
    view_pool_iter++;
  }
  CHECK(view_pool_iter == view_pools_.end());
  for (const auto& view : tmp_views_) {
    view->OutVisualizationInfo("ProcedureMesh/ViewsInformation/" +
        std::to_string(views_.back()->time_stamp_) + "/");
  }
}

bool Model::IsTrackingGood() {
  return is_tracking_good_;
}

double Model::MissingRatio() {
  int n_overall_missing_points = 0;
  int n_overall_points = 0;
  // Pls make sure matchings are already cached.
  for (const auto& view : views_) {
    std::vector<std::vector<int>> missing_paths;
    view->GetMissingPaths(&missing_paths);
    n_overall_points += view->n_points_;
    for (const auto& path : missing_paths) {
      n_overall_missing_points += path.size();
    }
  }
  return n_overall_missing_points / double(n_overall_points);
}

bool Model::IsCameraMovementSufficient() {
  if (views_.size() < 10) {
    return false;
  }
  std::vector<Eigen::Vector3d> poses;
  for (const auto& view : views_) {
    poses.emplace_back(-view->R_.transpose() * view->T_);
  }
  double trans_len = 0.0;
  for (int i = 0; i + 1 < poses.size(); i++) {
    trans_len += (poses[i] - poses[i + 1]).norm();
  }
  return trans_len > 0.3;
}

bool Model::IsCameraMotionSimilar(Model* ano_model) {
  CHECK_EQ(views_.size(), ano_model->views_.size());
  if (views_.size() < 5) {
    return false;
  }
  double my_path_len = 0.0;
  double error_sum = 0.0;
  for (int i = 0; i + 1 < views_.size(); i++) {
    Eigen::Vector3d my_prev_at_next  = views_[i + 1]->camera_.World2Cam(-views_[i]->R_.inverse() * views_[i]->T_);
    Eigen::Vector3d ano_prev_at_next =
        ano_model->views_[i + 1]->camera_.World2Cam(-ano_model->views_[i]->R_.inverse() * ano_model->views_[i]->T_);
    my_path_len += my_prev_at_next.norm();
    error_sum += (my_prev_at_next - ano_prev_at_next).norm();
  }
  const double kMaxDifferenceRatio = 0.2;
  return error_sum < kMaxDifferenceRatio * my_path_len;
}

void Model::DeleteOutliers() {
  if (views_.size() < 5) {
    return;
  }
  CHECK_EQ(points_.size(), points_history_.size());
  std::vector<int> feasible(n_points_, 0);
  int pt_idx = 0;
#pragma omp parallel for private(pt_idx) default(none) shared(n_points_, points_, feasible)
  for (pt_idx = 0; pt_idx < n_points_; pt_idx++) {
    if (IsSinglePointFeasible(points_[pt_idx])) {
      feasible[pt_idx] = 1;
    }
  }
  std::vector<Eigen::Vector3d> new_points;
  std::vector<int> new_points_history;
  for (int pt_idx = 0; pt_idx < n_points_; pt_idx++) {
    if (feasible[pt_idx]) {
      new_points.emplace_back(points_[pt_idx]);
      new_points_history.emplace_back(points_history_[pt_idx]);
    }
  }
  points_ = std::move(new_points);
  points_history_ = std::move(new_points_history);
  n_points_ = points_.size();
  UpdateOctree();
  UpdateTangs();
}

void Model::AdjustPoints(Graph* graph) {
  CHECK(graph != nullptr);
  CHECK_EQ(points_.size(), points_history_.size());
  StopWatch stop_watch;
  std::vector<std::vector<int>> paths;
  graph->GetPaths(&paths);
  std::vector<bool> need_del(n_points_, false);
  bool remove_small_components = false;
  if (remove_small_components) {
    for (const auto& path : paths) {
      if (path.size() <= 4 && graph->Degree(path.front()) > 1 && graph->Degree(path.back()) > 1) {
        for (int i = 1; i + 1 < path.size(); i++) {
          need_del[path[i]] = true;
        }
      }
    }
  }

  bool remove_severe_occlution_points = false;
  if (views_.size() >= 5 && remove_severe_occlution_points) {
    for (int u = 0; u < n_points_; u++) {
      int valid_view_cnt = 0;
      for (auto& view : views_) {
        if (view->matching_indexes_[u] == -1) {
          continue;
        }
        double std_dev = view->matched_std_dev_[view->matching_indexes_[u]];
        if (std_dev < 5.0 * hope_dist_) {
          valid_view_cnt += 1;
        }
      }
      if (valid_view_cnt < 3) {
        need_del[u] = true;
      }
    }
  }
  
  bool remove_weird_link = false;
  if (remove_weird_link) {
    std::vector<std::vector<std::pair<int, Eigen::Vector3d>>> out_paths(n_points_);
    for (int path_idx = 0; path_idx < paths.size(); path_idx++) {
      const auto& path = paths[path_idx];
      if (path.size() < 2) {
        continue;
      }
      int n = std::min(int(path.size()) / 2, 4);
      out_paths[path.front()].emplace_back(
          path_idx, (points_[path[n]] - points_[path.front()]).normalized());
      out_paths[path.back()].emplace_back(
          path_idx, (points_[path[path.size() - 1 - n]] - points_[path.back()]).normalized());
    }

    for (int path_idx = 0; path_idx < paths.size(); path_idx++) {
      const auto& path = paths[path_idx];
      if (path.size() > 4 || path.size() < 2) {
        continue;
      }
      bool weird = true;
      int n = std::min(int(path.size()) / 2, 4);
      Eigen::Vector3d fr_vec = (points_[path[n]] - points_[path.front()]).normalized();
      for (const auto& pr : out_paths[path.front()]) {
        if (pr.first != path_idx) {
          weird &= (std::abs(pr.second.dot(fr_vec)) < 0.5);
        }
      }
      Eigen::Vector3d bk_vec = (points_[path[path.size() - 1 - n]] - points_[path.back()]).normalized();
      for (const auto& pr : out_paths[path.back()]) {
        if (pr.first != path_idx) {
          weird &= (std::abs(pr.second.dot(fr_vec)) < 0.5);
        }
      }
      if (weird) {
        for (int i = 1; i + 1 < path.size(); i++) {
          need_del[path[i]] = true;
        }
      }
    }
  }

  LOG(INFO) << "Before removing overlapping duration: " << stop_watch.TimeDuration();
  // Remove overlapped small curves.
  graph->GetLinkedPaths(points_, &paths, -0.7 , 8);
  if (views_.size() >= 5) {
    for (auto& view : views_) {
      view->is_necessary_ = false;
    }
    std::vector<double> projected_length(paths.size(), 0);
    for (int i = 0; i < paths.size(); i++) {
      const auto& current_path = paths[i];
      CHECK(!current_path.empty());
      for (const auto& view : views_) {
        Eigen::Vector2d past_pt = view->camera_.World2Image(points_[current_path[0]]);
        for (int j = 1; j < current_path.size(); j++) {
          Eigen::Vector2d new_pt = view->camera_.World2Image(points_[current_path[j]]);
          projected_length[i] += (new_pt - past_pt).norm();
          past_pt = new_pt;
        }
      }
    }
    std::vector<int> p(paths.size(), 0);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(),
              p.end(),
              [&projected_length](int a, int b){ return projected_length[a] > projected_length[b]; });
    for (auto& view : views_) {
      view->RefreshCoveredMap();
    }
    const double kMinValidCoveredRatio = 0.3;
    std::vector<int> is_view_covered(views_.size(), -1);
    for (int path_idx_i = 0; path_idx_i < p.size(); path_idx_i++) {
      int path_idx = p[path_idx_i];
      const auto& current_path = paths[path_idx];
      int valid_covering_cnt = views_.size();
      int base_view_cnt = views_.size();
      for (int view_idx = 0; view_idx < views_.size(); view_idx++) {
        auto& view = views_[view_idx];
        double covered_ratio = view->CalcCoveredPixelsRatio(points_, current_path);
        if (covered_ratio < -1e-3) {
          base_view_cnt--;
        }
        if (covered_ratio < kMinValidCoveredRatio) {
          valid_covering_cnt--;
        } else {
          is_view_covered[view_idx] = path_idx;
        }
      }
      const int kMinValidCoveringCntThreshold = 3;
      if ((valid_covering_cnt < kMinValidCoveringCntThreshold) &&
          (current_path.size() > 5 ||
          graph->Degree(current_path.front()) <= 1 || graph->Degree(current_path.back()) <= 1)) {
        for (int i = 1; i + 1 < current_path.size(); i++) {
          need_del[current_path[i]] = true;
        }
        for (int u : { current_path.front(), current_path.back() }) {
          if (graph->Degree(u) <= 1) {
            need_del[u] = true;
          }
        }
      }
      else if (valid_covering_cnt == kMinValidCoveringCntThreshold &&
          graph->Degree(current_path.front()) > 1 && graph->Degree(current_path.back()) > 1) {
        for (int view_idx = 0; view_idx < views_.size(); view_idx++) {
          if (is_view_covered[view_idx] == path_idx) {
            views_[view_idx]->is_necessary_ = false;
          }
        }
      }
    }
  }

  graph->GetPaths(&paths);
  LOG(INFO) << "After removing overlapping duration: " << stop_watch.TimeDuration();
  // Remove too dense curves ("dense" here means they gather in a point in too many views.)
  bool remove_too_dense_curves = true;
  if (remove_too_dense_curves) {
    for (const auto& current_path : paths) {
      int n_too_dense_views = 0;
      for (auto& view : views_) {
        double path_len = 0.0;
        double hope_size = 0.0;
        // Eigen::Vector2d last_projection = view->camera_.World2Image(points_[current_path.front()]);
        Eigen::Vector3d last_point = points_[current_path.front()];
        for (int v : current_path) {
          // Eigen::Vector2d current_projection = view->camera_.World2Image(points_[v]);
          Eigen::Vector3d current_point = points_[v];
          // path_len += (current_projection - last_projection).norm();
          // hope_size += (current_point - last_point).norm() / hope_dist_;
          Eigen::Vector3d current_point_cam = view->camera_.World2Cam(current_point);
          Eigen::Vector3d last_point_cam = view->camera_.World2Cam(last_point);
          path_len += std::abs((last_point_cam - current_point_cam).dot(current_point_cam.normalized()));
          hope_size += (current_point - last_point).norm();
          // last_projection = current_projection;
          last_point = current_point;
        }
        const double kTooDenseRatioThreashold = 0.9;
        if (path_len > kTooDenseRatioThreashold * hope_size) {
          n_too_dense_views++;
        }
      }
      if (n_too_dense_views > 0.9 * views_.size() && views_.size() > 0) {
        for (int v : current_path) {
          need_del[v] = true;
        }
      }
    }
  }

  std::vector<Eigen::Vector3d> new_points;
  std::vector<int> new_points_history;

  // Curve grows.
  UpdateTangs(graph);
  if (views_.size() > 5) {
    const double kDirectedSearchingR = 20.0 * hope_dist_;
    const double kCosThres = 0.25;
    for (const auto& path : paths) {
      if (path.size() <= 10) { // Too short
        continue;
      }
      // CHECK(path.front() != path.back());
      std::vector<std::pair<int, int>>
          tmp_list = { { path[0], path[1] }, { path[path.size() - 1], path[path.size() - 2] } };
      for (const auto& pr : tmp_list) {
        int u = pr.first;
        int v = pr.second;
        if (graph->Degree(u) > 1 || need_del[u] || need_del[v]) {
          continue;
        }
        Eigen::Vector3d bias = tangs_[u];
        CHECK(tang_scores_[u] > 1e-3);
        std::vector<int> neighbors;
        octree_->SearchingR(points_[u], kDirectedSearchingR, &neighbors);
        bool found = false;
        for (int wooden : neighbors) {
          if ((points_[u] - points_[wooden]).norm() < hope_dist_) {
            continue;
          }
          found |=
              ((points_[wooden] - points_[u]).normalized().dot(bias) > kCosThres);
          if (found) {
            break;
          }
        }
        if (!found) {
          continue;
        }
        for (int i = 1; i <= 5; i++) {
          Eigen::Vector3d grow_pt = points_[u] + bias * hope_dist_ * i;
          if (IsSinglePointFeasible(grow_pt, 1.0, 1.0)) {
            new_points.push_back(grow_pt);
            new_points_history.push_back(-1);
          } else {
            break;
          }
        }
      }
    }
  }
  LOG(INFO) << "After growing points duration: " << stop_watch.TimeDuration();

  bool use_uniform_resample = true;
  if (use_uniform_resample) {
    const double sampling_len = hope_dist_;
    for (const auto &path : paths) {
      for (int i = 1; i + 1 < path.size(); i++) {
        int a = path[i - 1];
        int b = path[i];
        int c = path[i + 1];
        double len = (points_[a] - points_[b]).norm() + (points_[b] - points_[c]).norm();
        if (!need_del[a] && !need_del[c] && len < sampling_len) {
          need_del[b] = true;
        }
      }
    }

    LOG(INFO) << "After removing dense points duration: " << stop_watch.TimeDuration();

    for (const auto &path : paths) {
      for (int i = 0; i + 1 < path.size(); i++) {
        int a = path[i];
        int b = path[i + 1];
        double len = (points_[a] - points_[b]).norm();
        if (need_del[a] || need_del[b] || len < 2.0 * sampling_len) {
          continue;
        }
        int n_segments = std::ceil(len / sampling_len);
        Eigen::Vector3d step_vector = (points_[b] - points_[a]) / (double) n_segments;
        bool is_path_feasible = (points_history_[a] > 5 && points_history_[b] > 5);
        for (int t = 1; t < n_segments; t++) {
          if (!IsSinglePointFeasible(points_[a] + step_vector * t, 1.0, 1.0)) {
            is_path_feasible = false;
            break;
          }
        }
        if (is_path_feasible) {
          int new_history = std::min(points_history_[a], points_history_[b]);
          for (int t = 1; t < n_segments; t++) {
            new_points.emplace_back(points_[a] + step_vector * t);
            new_points_history.emplace_back(new_history);
          }
        }
      }
    }
    LOG(INFO) << "After adding new points duration: " << stop_watch.TimeDuration();
  }
  else {
    // Image based resampling.
    for (const auto &path : paths) {
      if (path.size() < 2) {
        continue;
      }
      std::vector<double> sampling_ratio(path.size() - 1, -1.0);
      for (int i = 0; i + 1 < path.size(); i++) {
        double current_sampling_ratio = (points_[path[i]] - points_[path[i + 1]]).norm() * focal_length_ / 8.0;
        for (const auto& view : views_) {
          Eigen::Vector2d bias =
              view->camera_.World2Image(points_[path[i]]) - view->camera_.World2Image(points_[path[i + 1]]);
          current_sampling_ratio = std::max(current_sampling_ratio, std::max(std::abs(bias(0)), std::abs(bias(1))));
        }
        sampling_ratio[i] = current_sampling_ratio;
      }
      for (int i = 0; i + 2 < path.size(); i++) {
        double len = sampling_ratio[i] + sampling_ratio[i + 1];
        if (len < 1.0 && !need_del[path[i]] && !need_del[path[i + 2]]) {
          need_del[path[i + 1]] = true;
        }
      }

      for (int i = 0; i + 1 < path.size(); i++) {
        int a = path[i];
        int b = path[i + 1];
        if (sampling_ratio[i] < 2.0 || need_del[a] || need_del[b]) {
          continue;
        }
        int n_segments = int(sampling_ratio[i]);
        Eigen::Vector3d step_vector = (points_[b] - points_[a]) / (double) n_segments;
        bool is_path_feasible = (points_history_[a] > 5 && points_history_[b] > 5);
        for (int t = 1; t < n_segments; t++) {
          if (!IsSinglePointFeasible(points_[a] + step_vector * t, 1.0, 1.0)) {
            is_path_feasible = false;
            break;
          }
        }
        if (is_path_feasible) {
          int new_history = std::min(points_history_[a], points_history_[b]);
          for (int t = 1; t < n_segments; t++) {
            new_points.emplace_back(points_[a] + step_vector * t);
            new_points_history.emplace_back(new_history);
          }
        }
      }
    }
  }

  for (int u = 0; u < n_points_; u++) {
    if (!need_del[u]) {
      new_points.emplace_back(points_[u]);
      new_points_history.emplace_back(points_history_[u]);
    }
  }
  points_ = std::move(new_points);
  points_history_ = std::move(new_points_history);
  // Update history.
  for (auto& history : points_history_) {
    history ++;
  }
  n_points_ = points_.size();
  CHECK_EQ(n_points_, points_history_.size());
  UpdateOctree();
  LOG(INFO) << "After building octree duration: " << stop_watch.TimeDuration();
  UpdateTangs();
}

bool Model::IsSinglePointFeasible(const Eigen::Vector3d& point, double single_view_max_error, double ave_max_error) {
  if (!IsSinglePointVisible(point)) {
    return false;
  }

  double error = 0.0;
  int false_cnt = 0;
  int in_view_cnt = 0;

  for (const auto& view : views_) {
    if (view->WorldPointOutImageRange(point)) {
      error += ave_max_error + 1e-8;
      continue;
    }
    in_view_cnt ++;
    // TODO: Less time cost.
    int nearest_idx = view->GetNearestPointIdx(point);
    Eigen::Vector2d pt = view->camera_.World2Image(point);
    Eigen::Vector2d pt_match = view->extractor_->points_[nearest_idx];

    double radius = std::max(view->extractor_->estimated_rs_[nearest_idx], 2.5);
    double single_error = (pt - pt_match).norm();
    if (single_error > radius * single_view_max_error) {
      false_cnt++;
    }
    error += single_error / radius;
  }
  error /= views_.size();

  return error < ave_max_error && false_cnt * 6 < in_view_cnt;
}

bool Model::IsSinglePointVisible(const Eigen::Vector3d& point) {
  for (auto& view : views_) {
    if (view->WorldPointOutImageRange(point)) {
      continue;
    }
    double depth = view->camera_.World2Cam(point)(2);
    if (depth < 2e-2) {
      return false;
    }
  }
  return true;
}

void Model::AddLostPoints() {
  StopWatch stop_watch;

  if (add_lost_points_method_ == "SEARCH") {
    AddLostPointsBySearch();
  }
  else if (add_lost_points_method_ == "DENSE_VOXEL") {
    LOG(FATAL) << "Deprecated.";
    AddLostPointsByDenseVoxel();
  }
  else if (add_lost_points_method_ == "DEPTH_SAMPLING") {
    LOG(FATAL) << "Deprecated.";
    AddLostPointsByDepthSampling();
  }
  else {
    LOG(FATAL) << "No such adding lost points method.";
  }
  LOG(INFO) << "---> UpdateMissingPoints time: " << stop_watch.TimeDuration();
}

void Model::AddLostPointsByDenseVoxel() {
  LOG(FATAL) << "Deprecated.";
  StopWatch stop_watch;
  const double inf = 1e9;
  Eigen::Vector3d ma(-inf, -inf, -inf);
  Eigen::Vector3d mi(inf, inf, inf);
  for (const auto &point : points_) {
    for (int t = 0; t < 3; t++) {
      ma(t) = std::max(ma(t), point(t));
      mi(t) = std::min(mi(t), point(t));
    }
  }
  auto center = 0.5 * (ma + mi);
  double fineness = hope_dist_ * 4;
  double hope_r = std::max(ma(0) - mi(0), std::max(ma(1) - mi(1), ma(2) - mi(2))) * 0.6;
  double r = fineness;
  while (r < hope_r) {
    r *= 2.0;
  }
  auto cube_set = std::make_unique<CubeSet>(center, r, fineness);
  for (const auto &view : views_) {
    std::vector<std::pair<Eigen::Vector3d, double>> world_rays;
    std::vector<std::vector<int>> missing_paths;
    view->GetMissingPaths(&missing_paths);
    for (const auto& path : missing_paths) {
      Eigen::Vector3d o;
      Eigen::Vector3d v;
      for (const int u : path) {
        std::tie(o, v) = view->GetWorldRayByIdx(u);
        cube_set->AddRay(o, v / v.norm(), 1.0);
      }
    }
    cube_set->UpdateCurrentSet();
  }
  std::vector<std::pair<int, Eigen::Vector3d>> cubes;
  cube_set->FindDenseCubes(std::max(views_.size() * 0.4, topology_searching_radius_), cubes);
  int n_new_points = 0;
  for (auto cube : cubes) {
    if (IsSinglePointFeasible(cube.second)) {
      n_new_points++;
      points_.push_back(cube.second);
    }
  }
  LOG(INFO) << "New points size: " << n_new_points;
  LOG(INFO) << "Add lost points time duration: " << stop_watch.TimeDuration();
  n_points_ = points_.size();
  UpdateOctree();
}

void Model::AddLostPointsBySearch() {
  // Currently, we only consider the last view.
  if (views_.size() < kSlidingWindowN) {
    return;
  }
  View* last_view = views_[views_.size() - kSlidingWindowN];
  
  std::vector<std::vector<int>> missing_paths;
  Eigen::Vector3d boundary_min( 1e9,  1e9,  1e9);
  Eigen::Vector3d boundary_max(-1e9, -1e9, -1e9);
  for (const auto& pt : points_) {
    for (int t = 0; t < 3; t++) {
      boundary_min(t) = std::min(boundary_min(t), pt(t));
      boundary_max(t) = std::max(boundary_max(t), pt(t));
    }
  }
  Eigen::Vector3d center = (boundary_min + boundary_max) * 0.5;
  boundary_min = center + (boundary_min - center) * 1.2;
  boundary_max = center + (boundary_max - center) * 1.2;
  UpdateSpanningTree();
  last_view->CacheMatchings(points_, tangs_, tang_scores_, spanning_tree_->Edges(), hope_dist_, false);
  // Pls make sure matchings have been cached before calling this function.
  last_view->GetMissingPaths(&missing_paths);
  for (const auto& path : missing_paths) {
    // LOG(INFO) << "missing path!";
    std::vector<std::vector<double>> depth_candidates;
    // LOG(INFO) << "path.size(): " << path.size();
    for (int idx : path) {
      Eigen::Vector3d o;
      Eigen::Vector3d v;
      std::tie(o, v) = last_view->GetWorldRayByIdx(idx);
      // Get boundary.
      double d_min = 0.0;
      double d_max = 1e9;
      for (double t = 0; t < 3; t++) {
        if (std::fabs(v(t)) < 1e-5) {
          if (o(t) < boundary_min(t) || o(t) > boundary_max(t)) {
            d_min = 1e9;
            d_max = -1e9;
          }
        } else {
          double l = (boundary_min(t) - o(t)) / v(t);
          double r = (boundary_max(t) - o(t)) / v(t);
          if (l > r) {
            std::swap(l, r);
          }
          d_min = std::max(d_min, l);
          d_max = std::min(d_max, r);
        }
      }
      depth_candidates.emplace_back();
      GetDepthCandidatesByWorldRay(o, v, d_min, d_max, &(depth_candidates.back()));
      for (double depth : depth_candidates.back()) {
        if (IsSinglePointFeasible(o + v * depth)) {
          points_.emplace_back(o + v * depth);
          points_history_.emplace_back(0);
        }
      }
    }
  }
  n_points_ = points_.size();
  CHECK_EQ(n_points_, points_history_.size());
  UpdateOctree();
  UpdateTangs();
}

void Model::GetDepthCandidatesByWorldRay(const Eigen::Vector3d& o,
                                         const Eigen::Vector3d& v,
                                         double d_min,
                                         double d_max,
                                         std::vector<double>* depth_candidates) {
  if (d_min > d_max) {
    return;
  }
  depth_candidates->clear();
  std::vector<std::pair<double, double>> depth_intersections;
  CHECK_GE(views_.size(), kSlidingWindowN);
  for (int view_idx = std::max(0, (int) views_.size() - 2 * kSlidingWindowN); view_idx < views_.size(); view_idx++) {
    std::vector<std::pair<double, double>> current_depth_intersections;
    views_[view_idx]->GetDepthIntersectionsByWorldRay(o, v, d_min, d_max, &current_depth_intersections);
    depth_intersections.insert(depth_intersections.end(),
                               current_depth_intersections.begin(),
                               current_depth_intersections.end());
  }
  std::vector<double> in_depths;
  std::vector<double> out_depths;
  for (const auto& pr : depth_intersections) {
    in_depths.emplace_back(pr.first);
    out_depths.emplace_back(pr.second);
  }
  std::sort(in_depths.begin(), in_depths.end());
  std::sort(out_depths.begin(), out_depths.end());
  auto in_iter = in_depths.begin();
  auto out_iter = out_depths.begin();
  double current_score = 0.0;
  double max_score = -1.0;
  double best_depth = (d_min + d_max) * 0.5;
  double past_depth = d_min;
  while (in_iter != in_depths.end() || out_iter != out_depths.end()) {
    if (in_iter != in_depths.end() && (out_iter == out_depths.end() || *in_iter < *out_iter)) {
      current_score += 1.0;
      past_depth = *in_iter;
      in_iter++;
    } else {
      if (current_score > max_score) {
        max_score = current_score;
        best_depth = (past_depth + *out_iter) * 0.5;
      }
      current_score -= 1.0;
      past_depth = *out_iter;
      out_iter++;
    }
  }
  // LOG(INFO) << "max_score: " << max_score << " best_depth: " << best_depth;
  if (max_score > 3.0) {
    depth_candidates->emplace_back(best_depth);
  }
}

void Model::AddLostPointsByDepthSampling() {
  // Currently, we only consider the last view.
  if (views_.size() < kSlidingWindowN) {
    return;
  }
  View* last_view = views_[views_.size() - kSlidingWindowN];
  std::vector<std::vector<int>> missing_paths;
  // Pls make sure matchings have been cached before calling this function.
  last_view->GetMissingPaths(&missing_paths);
  Eigen::Vector3d boundary_min( 1e9,  1e9,  1e9);
  Eigen::Vector3d boundary_max(-1e9, -1e9, -1e9);
  for (const auto& pt : points_) {
    for (int t = 0; t < 3; t++) {
      boundary_min(t) = std::min(boundary_min(t), pt(t));
      boundary_max(t) = std::max(boundary_max(t), pt(t));
    }
  }
  Eigen::Vector3d center = (boundary_min + boundary_max) * 0.5;
  boundary_min = center + (boundary_min - center) * 1.2;
  boundary_max = center + (boundary_max - center) * 1.2;
  std::vector<Eigen::Vector3d> new_points;
  for (const auto& path : missing_paths) {
    for (int idx : path) {
      Eigen::Vector3d o;
      Eigen::Vector3d v;
      std::tie(o, v) = last_view->GetWorldRayByIdx(idx);

      // Get boundary.
      double d_min = 0.0;
      double d_max = 1e9;
      for (double t = 0; t < 3; t++) {
        if (std::fabs(v(t)) < 1e-5) {
          if (o(t) < boundary_min(t) || o(t) > boundary_max(t)) {
            d_min = 1e9;
            d_max = -1e9;
          }
        } else {
          double l = (boundary_min(t) - o(t)) / v(t);
          double r = (boundary_max(t) - o(t)) / v(t);
          if (l > r) {
            std::swap(l, r);
          }
          d_min = std::max(d_min, l);
          d_max = std::min(d_max, r);
        }
      }

      // LOG(FATAL) << d_min << " " << d_max;
      // Sampling and solving.
      const int kSampleN = 20;
      Eigen::Vector3d past_point(1e9, 1e9, 1e9);
      std::vector<Eigen::Vector3d> current_points;
      for (int t = 0; t <= kSampleN; t++) {
        // double new_d = FindBestDepthByWorldRay(o, v, (d_max - d_min) / kSampleN * t + d_min, d_min, d_max);
        double new_d = (d_max - d_min) / kSampleN * t + d_min;
        Eigen::Vector3d new_point = o + v * new_d;
        // LOG(INFO) << "d_min: " << d_min << " d_max: " << d_max << " d: " << new_d;
        if (new_d > d_min && new_d < d_max &&
           (past_point - new_point).norm() > hope_dist_ && IsSinglePointFeasible(new_point)) {
          current_points.emplace_back(new_point);
          past_point = new_point;
        }
      }
      if (!current_points.empty()) {
        new_points.emplace_back(current_points[current_points.size() / 2.0]);
      }
    }
  }
  points_.insert(points_.end(), new_points.begin(), new_points.end());
  n_points_ = points_.size();
  UpdateOctree();
  UpdateTangs();
}

double Model::FindBestDepthByWorldRay(const Eigen::Vector3d& o,
                                      const Eigen::Vector3d& v,
                                      double initial_depth,
                                      double d_min,
                                      double d_max) {
  double d = initial_depth;
  double variance = 1.0;
  double residual = 0.0;
  int iter_num = 0;
  while (iter_num++ < 10 && variance > 1e-2) {
    std::vector<std::tuple<double, Eigen::Vector3d, Eigen::Vector3d>> target_planes;
    for (int view_idx = views_.size() - kSlidingWindowN + 1; view_idx < views_.size(); view_idx++) {
      auto& view = views_[view_idx];
      if (view->WorldPointOutImageRange(o + v * d)) {
        continue;
      }
      int nearest_idx = view->GetNearestPointIdx(o + v * d);
      Eigen::Vector3d target_o;
      Eigen::Vector3d target_v;
      std::tie(target_o, target_v) = view->GetWorldRayByIdx(nearest_idx);
      Eigen::Vector3d x;
      Eigen::Vector3d y;
      std::tie(x, y) = Math::GetOrthVectors(target_v);
      double w = target_v.cross(v).norm();
      target_planes.emplace_back(w, target_o, x);
      target_planes.emplace_back(w, target_o, y);
    }

    if (target_planes.size() < 3) {
      LOG(INFO) << "Not enough plane.";
      return -1.0;
    }
    // CHECK_GE(target_planes.size(), 3);
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    for (const auto& plane : target_planes) {
      Eigen::Vector3d target_o = o - std::get<1>(plane);
      if (target_o.norm() < 1e-3) {
        continue;
      }
      Eigen::Vector3d target_n = std::get<2>(plane);
      double weight = std::get<0>(plane);
      // minimize |(target_o + v * d).dot(target_n)|^2
      for (int t = 0; t < 3; t++) {
        a += weight * v(t) * v(t) * target_n(t) * target_n(t);
        b += weight * 2.0 * v(t) * target_o(t) * target_n(t) * target_n(t);
        c += weight * target_o(t) * target_o(t) * target_n(t) * target_n(t);
      }
    }
    double past_d = d;
    d = -0.5 * b / a;
    variance = std::abs(d - past_d);
    residual = a * d * d + b * d + c;
  }
  return d;
}

double Model::UpdatePoints(bool use_linked_paths, Graph* graph, Graph* tree) {
  if (update_points_method_ == "GLOBAL") {
    return UpdatePointsGlobal();
  }
  else if (update_points_method_ == "SPLIT_AND_SOLVE") {
    return UpdatePointsSplitAndSolve(use_linked_paths, graph, tree);
  }
  else {
    LOG(FATAL) << "No such method.";
    return -1.0;
  }
}

double Model::UpdatePointsSplitAndSolve(bool use_linked_paths, Graph* graph, Graph* tree) {
  StopWatch stop_watch;
  CHECK(graph != nullptr);
  CHECK(tree != nullptr);
  // Update local tangs.

  // Cache matchings.
  int view_idx;
  int n_views = views_.size();
#pragma omp parallel for private(view_idx) default(none) shared(n_views, points_, tangs_, tang_scores_, hope_dist_, views_, tree)
  for (view_idx = 0; view_idx < n_views; view_idx++) {
    const auto& view = views_[view_idx];
    view->CacheMatchings(points_, tangs_, tang_scores_, tree->Edges(), hope_dist_, false);
  }
  LOG(INFO) << "Core duration: " << stop_watch.TimeDuration();
  views_.back()->OutDebugImage("last", points_, global_data_pool_);
  // Get weights;
  std::vector<double> point_weights(n_points_, 1.0);
  for(int u = 0; u < n_points_; u++) {
    if (graph->Degree(u) == 1) {
      point_weights[u] = 1e3;
    }
  }

  points_feasible_.resize(n_points_);
  int pt_idx = 0;
#pragma omp parallel for private(pt_idx) default(none) shared(n_points_, points_, points_feasible_)
  for (pt_idx = 0; pt_idx < n_points_; pt_idx++) {
    if (IsSinglePointFeasible(points_[pt_idx])) {
      points_feasible_[pt_idx] = 1;
    }
  }
  // Solve.
  std::vector<std::vector<int>> paths;
  if (use_linked_paths) {
    graph->GetLinkedPaths(points_, &paths, -0.7, 10, false);
  }
  else {
    graph->GetPaths(&paths);
  }

  std::vector<std::vector<Eigen::Vector3d>> path_solutions;
  for (const auto& path : paths) {
    path_solutions.emplace_back();
    path_solutions.back().resize(path.size());
  }
  int n_paths = paths.size();
  int path_idx = 0;
#pragma omp parallel for private(path_idx) default(none) shared(n_paths, paths, path_solutions, point_weights)
  for (path_idx = 0; path_idx < n_paths; path_idx++) {
    UpdatePointsSingleComponent(paths[path_idx], point_weights, &path_solutions[path_idx]);
  }

  std::vector<Eigen::Vector3d> new_points(n_points_, Eigen::Vector3d::Zero());
  std::vector<int> points_base(n_points_, 0);

  for (int i = 0; i < n_paths; i++) {
    for (int j = 0; j < paths[i].size(); j++) {
      new_points[paths[i][j]] += path_solutions[i][j];
      points_base[paths[i][j]] += 1;
    }
  }

  std::vector<Eigen::Vector3d> past_points = points_;
  points_.clear();

  double residual = 0.0;
  std::vector<double> residuals(n_points_);
  for (int i = 0; i < n_points_; i++) {
    // CHECK(points_base[i] > 0);
    if (!points_base[i]) {
      points_.emplace_back(past_points[i]);
      continue;
    }
    Eigen::Vector3d new_point = new_points[i] / points_base[i];
    points_.emplace_back(new_point);
    // }
  }
  UpdateTangs(graph);
  for (int i = 0; i < n_points_; i++) {
    double res = (points_[i] - past_points[i]).cross(tangs_[i]).norm();
    residual += res;
    residuals[i] = res;
  }

  LOG(INFO) << "Solving duration: " << stop_watch.TimeDuration();
  n_points_ = points_.size();
  UpdateOctree();
  std::sort(residuals.begin(), residuals.end());
  return residuals[int(residuals.size() * 0.95)];
  //
  // return residual / n_points_;
}

double Model::UpdatePointsSingleComponent(const std::vector<int>& indexes,
                                          const std::vector<double>& /*weights*/,
                                          std::vector<Eigen::Vector3d>* solutions) {
  const double kTangWeight = 0.5;
  CHECK(!indexes.empty());
  CHECK_EQ(indexes.size(), solutions->size());
  // If the path is good enough, don't optimize.
  /*
  std::vector<double> path_errors;
  for (int u : indexes) {
    std::vector<std::pair<double, View*>> valid_views;
    GetSinglePointViewWeight(u, &valid_views);
    double errors = 0.0;
    for (const auto& pr : valid_views) {
      View* view = pr.second;
      if (view->GetMatchingRadius(u) > 0.0) {
        // if (view->matched_std_dev_[view->matching_indexes_[u]] > 10.0 * hope_dist_) {
        //   errors.emplace_back(0.0);
        // } else {
        errors += pr.first * (view->GetMatchingPixels(u) - view->camera_.World2Image(points_[u])).norm();
        // }
      }
    }
    path_errors.emplace_back(errors);
  }
  CHECK(!path_errors.empty());
  std::sort(path_errors.begin(), path_errors.end());
  if (path_errors[path_errors.size() * 90 / 100] < 1.5) {
    for (int i = 0; i < indexes.size(); i++) {
      (*solutions)[i] = points_[indexes[i]];
    }
    return 0.0;
  }
*/
  // If the path has infeasible points, don't optimize.
  bool path_feasible = true;
  for (int u : indexes) {
    if (!points_feasible_[u]) {
      path_feasible = false;
      break;
    }
  }
  if (!path_feasible) {
    for (int i = 0; i < indexes.size(); i++) {
      (*solutions)[i] = points_[indexes[i]];
    }
    return 0.0;
  }

  // Prepareation for LSQR solver.
  std::vector<unsigned> coord_r;
  std::vector<unsigned> coord_c;
  std::vector<double> values;
  std::vector<double> B;
  auto AddValue = [&coord_r, &coord_c, &values](int r, int c, double val) {
    coord_r.push_back(r);
    coord_c.push_back(c);
    CHECK(!std::isnan(val));
    values.push_back(val);
  };

  int idx = -1;

  std::vector<int> sampled_index_for_solving;
  GetControlIndexes(indexes, &sampled_index_for_solving);

  std::vector<double> solution(indexes.size() * 3);
  for (int i = 0; i < indexes.size(); i++) {
    for (int t = 0; t < 3; t++) {
      solution[i * 3 + t] = points_[indexes[i]](t);
    }
  }
  bool use_lsqr = false;
  bool use_ceres_solver = false;
  bool use_ceres_solver_bezier = true;
  if (indexes.size() < 10 || solver_type_ == SolverType::CERES) {
    use_ceres_solver = true;
    use_ceres_solver_bezier = false;
  }
  if (indexes.size() < 10 || solver_type_ == SolverType::LSQR) {
    use_lsqr = true;
    use_ceres_solver = false;
    use_ceres_solver_bezier = false;
  }
  if (use_lsqr) {
    // LOG(FATAL) << "Deprecated.";
    // View constraint.
    for (int i : sampled_index_for_solving) {
      int u = indexes[i];
      std::vector<std::pair<double, View *>> valid_views;
      GetSinglePointViewWeight(u, &valid_views);
      if (valid_views.empty()) {
        for (int j = 0; j < indexes.size(); j++) {
          (*solutions)[j] = points_[indexes[j]];
        }
        return 0.0;
      }
      CHECK(!valid_views.empty());
      for (int view_idx = 0; view_idx < valid_views.size(); view_idx++) {
        const auto &view = valid_views[view_idx].second;
        // const double current_weight = std::sqrt(valid_views[view_idx].first) * weights[u] *
        std::max(0.5, std::min(2.0, view->average_depth_ / view->camera_.World2Cam(points_[u])(2)));
        const double current_weight = std::sqrt(valid_views[view_idx].first);
        CHECK(!std::isnan(current_weight));
        CHECK(view->MatchingCached());
        int p_idx = i * 3;
        // auto pr = view->GetRayAndNormal(points_[i]);
        auto pr = view->GetMatching(u);
        Eigen::Vector3d current_ray = pr.first;
        Eigen::Vector3d current_normal = pr.second;
        current_normal /= current_normal.norm();
        // Distance to plane constraint.
        // min: (R_ * P + T_ - current_ray).dot(current_normal)
        // min: (R_ * P).dot(current_normal) - (-T_ + current_ray).dot(current_normal)
        idx++;
        for (int t = 0; t < 3; t++) {
          double val =
            view->R_(0, t) * current_normal(0) + view->R_(1, t) * current_normal(1) +
            view->R_(2, t) * current_normal(2);
          if (std::abs(val) < 1e-8) {
            continue;
          }
          AddValue(idx, p_idx + t, val * current_weight);
        }
        B.push_back((-view->T_ + current_ray).dot(current_normal) * current_weight);

        // Core points constraint.
        // min: (R_ * P + T_ - current_ray).dot(current_tang)
        // min: (R_ * P).dot(current_tang) - (-T_ + current_ray).dot(current_tang)
        // TODO: Hard code here.
        Eigen::Vector3d current_tang = current_ray.cross(current_normal);
        current_tang /= current_tang.norm();

        // double tang_weight = std::max(1.0 - tang_scores_[u], 0.5);
        // const double tang_weight = 0.5;
        const double tang_weight = (i == 0 || i + 1 == indexes.size()) ? 1.0 : kTangWeight;
        // const double tang_weight = 1.0;
        idx++;
        for (int t = 0; t < 3; t++) {
          AddValue(idx, p_idx + t,
                   current_weight * tang_weight *
                   (view->R_(0, t) * current_tang(0) + view->R_(1, t) * current_tang(1) +
                    view->R_(2, t) * current_tang(2)));
        }
        B.push_back((-view->T_ + current_ray).dot(current_tang) * current_weight * tang_weight);
      }
    }
    // Smooth constraint.
    if (hope_dist_weight_ > 1e-9) {
      if (smooth_method_ == "TANGENT") {
        for (int i = 1; i + 2 < indexes.size(); i++) {
          int u = indexes[i];
          int v = indexes[i + 1];
          Eigen::Vector3d bias = points_[u] - points_[v];
          Eigen::Vector3d hope_bias = (bias.dot(tangs_[v]) > 0.0 ? hope_dist_ : -hope_dist_) * tangs_[v];
          for (int t = 0; t < 3; t++) {
            idx++;
            AddValue(idx, i * 3 + t, hope_dist_weight_);
            AddValue(idx, (i + 1) * 3 + t, -hope_dist_weight_);
            B.push_back(hope_dist_weight_ * hope_bias(t));
          }
        }
      } else if (smooth_method_ == "LAPLACIAN") {
        for (int i = 1; i + 1 < indexes.size(); i++) {
          double ratio = std::abs(
            (points_[indexes[i]] - points_[indexes[i - 1]]).dot(points_[indexes[i + 1]] - points_[indexes[i - 1]]) /
            (points_[indexes[i]] - points_[indexes[i + 1]]).dot(points_[indexes[i + 1]] - points_[indexes[i - 1]]));
          ratio = std::max(0.1, std::min(10.0, ratio));
          double base = 2.0 / (1.0 + ratio);
          // (p[i] - p[i - 1]) - ratio(p[i + 1] - p[i]) == 0
          for (int t = 0; t < 3; t++) {
            idx++;
            AddValue(idx, i * 3 + t, -(1.0 + ratio) * hope_dist_weight_ * base);
            AddValue(idx, (i - 1) * 3 + t, 1.0 * hope_dist_weight_ * base);
            AddValue(idx, (i + 1) * 3 + t, ratio * hope_dist_weight_ * base);
            // AddValue(idx, i * 3 + t, -2.0 * hope_dist_weight_);
            // AddValue(idx, (i - 1) * 3 + t, hope_dist_weight_);
            // AddValue(idx, (i + 1) * 3 + t, hope_dist_weight_);
            B.push_back(0.0);
          }
        }
        if (indexes.front() == indexes.back() && indexes.size() > 3) { // Handle circular path
          int a = 1;
          int b = 0;
          int c = indexes.size() - 2;
          double ratio = std::abs(
            (points_[indexes[b]] - points_[indexes[a]]).dot(points_[indexes[c]] - points_[indexes[a]]) /
            (points_[indexes[b]] - points_[indexes[c]]).dot(points_[indexes[c]] - points_[indexes[a]]));
          ratio = std::max(0.1, std::min(10.0, ratio));
          // (p[i] - p[i - 1]) - ratio(p[i + 1] - p[i]) == 0
          for (int t = 0; t < 3; t++) {
            idx++;
            AddValue(idx, b * 3 + t, -(1.0 + ratio) * hope_dist_weight_);
            AddValue(idx, a * 3 + t, 1.0 * hope_dist_weight_);
            AddValue(idx, c * 3 + t, ratio * hope_dist_weight_);
            B.push_back(0.0);
          }
          for (int t = 0; t < 3; t++) {
            idx++;
            const double weight = 1e3;
            AddValue(idx, 0 * 3 + t, weight);
            AddValue(idx, (indexes.size() - 1) * 3 + t, -weight);
            B.push_back(0.0);
          }
        }
      } else {
        LOG(FATAL) << "No such smoothing method.";
      }
    }

    // Gather.
    const int kTooShortPath = 3;
    if (gather_weight_ > 1e-9 && indexes.size() <= kTooShortPath && indexes.size() >= 2) {
      Eigen::Vector3d mean_pt(0.0, 0.0, 0.0);
      for (int u : indexes) {
        mean_pt += points_[u];
      }
      mean_pt /= indexes.size();
      for (int i = 0; i < indexes.size(); i++) {
        int u = indexes[i];
        Eigen::Vector3d hope_pt = mean_pt * 0.5 + points_[u] * 0.5;
        for (int t = 0; t < 3; t++) {
          idx++;
          AddValue(idx, i * 3 + t, gather_weight_);
          B.push_back(gather_weight_ * hope_pt(t));
        }
      }
    }
    // LOG(INFO) << "before solving.";
    // LOG(INFO) << "n_varaibles: " << indexes.size() * 3 << "n_functions: " << B.size();
    if (indexes.size() * 3 <= B.size()) {
      Utils::SolveLinearSqrLSQR((int) B.size(), indexes.size() * 3, coord_r, coord_c, values, B, solution);
    }
    // LOG(INFO) << "after solving.";
  }
  else if (use_ceres_solver) { // use ceres solver.
    ceres::Problem problem;
    // View constraint.
    for (int i : sampled_index_for_solving) {
      int u = indexes[i];
      std::vector<std::pair<double, View*>> valid_views;
      GetSinglePointViewWeight(u, &valid_views);
      if (valid_views.empty()) {
        for (int j = 0; j < indexes.size(); j++) {
          (*solutions)[j] = points_[indexes[j]];
        }
        return 0.0;
      }
      CHECK(!valid_views.empty());
      // std::random_shuffle(valid_views.begin(), valid_views.end());
      // valid_views.resize(std::min((int) valid_views.size(), 5));
      double w_sum = 0.0;
      for (const auto& pr : valid_views) {
        w_sum += pr.first;
      }
      for (auto& pr : valid_views) {
        pr.first /= w_sum;
      }

      for (int view_idx = 0; view_idx < valid_views.size(); view_idx++) {
        const auto &view = valid_views[view_idx].second;
        const double current_weight = std::sqrt(valid_views[view_idx].first);
        CHECK(!std::isnan(current_weight));
        CHECK(view->MatchingCached());
        int view_matching_idx = view->matching_indexes_[u];
        Eigen::Vector2d matching_o = view->extractor_->points_[view_matching_idx];
        Eigen::Vector2d matching_v = view->extractor_->tangs_[view_matching_idx];
        double tang_score = view->extractor_->tang_scores_[view_matching_idx];
        const double tang_weight = (i == 0 || i + 1 == indexes.size() || tang_score < 0.90) ? 1.0 : kTangWeight;
        // const double tang_weight = kTangWeight;
        ceres::CostFunction* cost_function =
          ModelToViewError::Create(view->R_,
                                   view->T_,
                                   matching_o(0),
                                   matching_o(1),
                                   matching_v(0),
                                   matching_v(1),
                                   current_weight,
                                   tang_weight,
                                   view->focal_length_,
                                   view->extractor_->width_,
                                   view->extractor_->height_);
        problem.AddResidualBlock(cost_function, nullptr, solution.data() + (i * 3));
      }
    }

    // Smoothness.
    for (int i = 1; i + 1 < indexes.size(); i++) {
      // double ratio = std::abs(
      //   (points_[indexes[i]] - points_[indexes[i - 1]]).dot(points_[indexes[i + 1]] - points_[indexes[i - 1]]) /
      //   (points_[indexes[i]] - points_[indexes[i + 1]]).dot(points_[indexes[i + 1]] - points_[indexes[i - 1]]));
      // ratio = std::max(0.1, std::min(10.0, ratio));
      // double base = 2.0 / (1.0 + ratio);
      // (p[i] - p[i - 1]) - ratio(p[i + 1] - p[i]) == 0
      for (int t = 0; t < 3; t++) {
        idx++;
        double len_a = (points_[indexes[i]] - points_[indexes[i - 1]]).norm();
        double len_b = (points_[indexes[i]] - points_[indexes[i + 1]]).norm();
        double len_mean = (len_a + len_b) * 0.5;
        ceres::CostFunction* cost_function =
            SmoothnessError::Create(hope_dist_weight_ * hope_dist_ / len_mean, hope_dist_weight_ * hope_dist_ / len_mean);
        problem.AddResidualBlock(cost_function,
                                 nullptr,
                                 solution.data() + (i - 1) * 3,
                                 solution.data() + i * 3,
                                 solution.data() + (i + 1) * 3);
      }
    }
    // Shrink.
    if (shrink_error_weight_ > 1e-9 && indexes.size() > 2) {
      std::vector<double> shrink_scale_weights(indexes.size());
      double weight_sum = 0.0;
      for (int i = 0; i + 1 < indexes.size(); i++) {
        // double w = 1.0 / (points_[indexes[i]] - points_[indexes[i + 1]]).norm();
        double w = 1.0;
        shrink_scale_weights[i] = w;
        weight_sum += w;
      }
      double co = (indexes.size() - 1) / weight_sum;
      for (auto& w : shrink_scale_weights) {
        w *= co;
      }
      for (int i = 1; i + 2 < indexes.size(); i++) {
        for (int t = 0; t < 3; t++) {
          ceres::CostFunction *cost_function =
              ShrinkError::Create(shrink_error_weight_ * shrink_scale_weights[i]);
          problem.AddResidualBlock(cost_function,
                                   nullptr,
                                   solution.data() + i * 3,
                                   solution.data() + (i + 1) * 3);
        }
      }
      problem.AddResidualBlock(FixError::Create(shrink_error_weight_ * 2.0 * shrink_scale_weights.front(),
                                                            points_[indexes.front()]),
                               nullptr,
                               solution.data() + 3);
      problem.AddResidualBlock(FixError::Create(shrink_error_weight_ * 2.0 * shrink_scale_weights.back(),
                                                            points_[indexes.back()]),
                               nullptr,
                               solution.data() + (indexes.size() - 2) * 3);
    }
    // Solve & Optimization.
    ceres::Solver::Options options;
    options.max_num_iterations = 1;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    // double build_duration = stop_watch.TimeDuration();
    ceres::Solve(options, &problem, &summary);
    // double solve_duration = stop_watch.TimeDuration();
  }
  else {
    // LOG(FATAL) << "here.";
    std::vector<Eigen::Vector3d> initial_points;
    std::vector<double> errors;
    for (int idx : indexes) {
      initial_points.emplace_back(points_[idx]);
      std::vector<std::pair<double, View *>> valid_views;
      GetSinglePointViewWeight(idx, &valid_views);
      if (valid_views.size() < 2) {
        for (int j = 0; j < indexes.size(); j++) {
          (*solutions)[j] = points_[indexes[j]];
        }
        return 0.0;
      }
      double current_error = 0.0;
      for (const auto &pr : valid_views) {
        double weight = pr.first;
        View *view = pr.second;
        current_error += weight * (view->matching_pixels_[idx] - view->camera_.World2Image(points_[idx])).norm();
      }
      errors.emplace_back(current_error);
    }
    CHECK_EQ(errors.size(), initial_points.size());
    auto curve = std::make_unique<BezierCurve>(initial_points, errors, 2.0, hope_dist_);
    // LOG(INFO) << "Compression ratio: " << curve->points_p_.size() / double(initial_points.size());
    std::vector<double> p_solutions(curve->points_p_.size() * 3, 0.0);
    std::vector<double> t_solutions(curve->points_t_.size() * 3, 0.0);
    for (int i = 0; i < curve->points_p_.size(); i++) {
      for (int k = 0; k < 3; k++) {
        p_solutions[i * 3 + k] = curve->points_p_[i](k);
      }
    }
    for (int i = 0; i < curve->points_t_.size(); i++) {
      for (int k = 0; k < 3; k++) {
        t_solutions[i * 3 + k] = curve->points_t_[i](k);
      }
    }

    ceres::Problem problem;
    CHECK_EQ(indexes.size(), curve->expressions_.size());
    // For debugging
    /*
    for (int i = 0; i < indexes.size(); i++) {
      LOG(INFO) << "----";
      LOG(INFO) << points_[indexes[i]].transpose();
      LOG(INFO) << curve->points_p_[i].transpose();
      LOG(INFO) << curve->points_t_[curve->expressions_[i].t_idx_].transpose();
      LOG(INFO) << curve->expressions_[i].p0_idx_ << " " << curve->expressions_[i].p1_idx_;
      LOG(INFO) << curve->expressions_[i].t_;
    }*/
    // LOG(INFO) << "r: " << double(curve->points_p_.size() + curve->points_t_.size()) / indexes.size();
    const double kStepLen = 5e-1;
    double past_t = -kStepLen - 1e-3;
    double true_past_t = -1.0;
    for (int i = 0; i < indexes.size(); i++) {
      int u = indexes[i];
      const auto& expression = curve->expressions_[i];
      // LOG(INFO) << curve->points_p_.size() << " " << indexes.size();
      if (expression.t_ < true_past_t + 1e-4) {
        past_t = -kStepLen - 1e-3;
      }
      if (expression.t_ < past_t + kStepLen && expression.t_ + 1e-3 < 1.0) {
        true_past_t = expression.t_;
        continue;
      }
      std::vector<std::pair<double, View*>> valid_views;
      GetSinglePointViewWeight(u, &valid_views);
      CHECK(!valid_views.empty());
      // LOG(INFO) << "expression_t: " << expression.t_ << "past_t: " << past_t;
      for (int view_idx = 0; view_idx < valid_views.size(); view_idx++) {
        // LOG(INFO) << "here.";
        const auto &view = valid_views[view_idx].second;
        const double current_weight = std::sqrt(valid_views[view_idx].first);
        CHECK(!std::isnan(current_weight));
        CHECK(view->MatchingCached());
        int view_matching_idx = view->matching_indexes_[u];
        Eigen::Vector2d matching_o = view->extractor_->points_[view_matching_idx];
        Eigen::Vector2d matching_v = view->extractor_->tangs_[view_matching_idx];
        const double tang_weight = (i == 0 || i + 1 == indexes.size()) ? 1.0 : kTangWeight;
        if (expression.t_ > 1e-3 && expression.t_ + 1e-3 < 1.0) {
          ceres::CostFunction *cost_function =
              BezierModelToViewError::Create(view->R_,
                                             view->T_,
                                             matching_o(0),
                                             matching_o(1),
                                             matching_v(0),
                                             matching_v(1),
                                             current_weight,
                                             tang_weight,
                                             view->focal_length_,
                                             view->extractor_->width_,
                                             view->extractor_->height_,
                                             expression.t_);
          CHECK_NE(expression.p0_idx_, expression.p1_idx_);
          CHECK_LT(expression.p0_idx_ * 3, p_solutions.size());
          CHECK_LT(expression.p1_idx_ * 3, p_solutions.size());
          CHECK_LT(expression.t_idx_ * 3, t_solutions.size());
          problem.AddResidualBlock(cost_function,
                                   nullptr,
                                   p_solutions.data() + (expression.p0_idx_ * 3),
                                   t_solutions.data() + (expression.t_idx_ * 3),
                                   p_solutions.data() + (expression.p1_idx_ * 3));
        }
        else if (expression.t_ < 1e-3) {
          ceres::CostFunction *cost_function =
              ModelToViewError::Create(view->R_,
                                       view->T_,
                                       matching_o(0),
                                       matching_o(1),
                                       matching_v(0),
                                       matching_v(1),
                                       current_weight,
                                       tang_weight,
                                       view->focal_length_,
                                       view->extractor_->width_,
                                       view->extractor_->height_);
          problem.AddResidualBlock(cost_function,
                                  nullptr,
                                  p_solutions.data() + (expression.p0_idx_ * 3));
        }
        else {
          ceres::CostFunction *cost_function =
              ModelToViewError::Create(view->R_,
                                       view->T_,
                                       matching_o(0),
                                       matching_o(1),
                                       matching_v(0),
                                       matching_v(1),
                                       current_weight,
                                       tang_weight,
                                       view->focal_length_,
                                       view->extractor_->width_,
                                       view->extractor_->height_);
          problem.AddResidualBlock(cost_function,
                                   nullptr,
                                   p_solutions.data() + (expression.p1_idx_ * 3));
        }
      }
      past_t += kStepLen;
      true_past_t = expression.t_;
    }
    // Smoothness
    CHECK_EQ(curve->points_t_.size() + 1, curve->points_p_.size());
    for (int i = 1; i + 1 < curve->points_p_.size(); i++) {
      double len_a = (curve->points_t_[i - 1] - curve->points_p_[i]).norm();
      double len_b = (curve->points_t_[i] - curve->points_p_[i]).norm();
      if (len_a < 1e-7 || len_b < 1e-7) {
        continue;
      }
      for (int t = 0; t < 3; t++) {
        idx++;
        ceres::CostFunction* cost_function =
            SmoothnessError::Create(hope_dist_weight_ * hope_dist_ / len_a, hope_dist_weight_ * hope_dist_ / len_b);
        problem.AddResidualBlock(cost_function,
                                 nullptr,
                                 t_solutions.data() + (i - 1) * 3,
                                 p_solutions.data() + i * 3,
                                 t_solutions.data() + i * 3);
      }
    }
    for (int i = 0; i < curve->points_t_.size(); i++) {
      double len_a = (curve->points_t_[i] - curve->points_p_[i]).norm();
      double len_b = (curve->points_t_[i] - curve->points_p_[i + 1]).norm();
      if (len_a < 1e-7 || len_b < 1e-7) {
        continue;
      }
      for (int t = 0; t < 3; t++) {
        idx++;
        ceres::CostFunction* cost_function =
            SmoothnessError::Create(hope_dist_weight_ * hope_dist_ / len_a, hope_dist_weight_ * hope_dist_ / len_b);
        problem.AddResidualBlock(cost_function,
                                 nullptr,
                                 p_solutions.data() + i * 3,
                                 t_solutions.data() + i * 3,
                                 p_solutions.data() + (i + 1) * 3);
      }
    }
    // LOG(INFO) << "Problem completed.";
    // Solve & Optimization.
    ceres::Solver::Options options;
    // options.max_num_iterations = 1;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // LOG(INFO) << "Problem solved.";
    solution.clear();
    for (const auto& expression : curve->expressions_) {
      double t = expression.t_;
      double a = (1.0 - t) * (1.0 - t);
      double b = 2.0 * (1.0 - t) * t;
      double c = t * t;
      for (int k = 0; k < 3; k++) {
        solution.emplace_back(a * p_solutions[expression.p0_idx_ * 3 + k] +
                              b * t_solutions[expression.t_idx_ * 3 + k] +
                              c * p_solutions[expression.p1_idx_ * 3 + k]);
        if (std::isnan(solution.back())) {
          for (int j = 0; j < indexes.size(); j++) {
            (*solutions)[j] = points_[indexes[j]];
          }
          return 0.0;
        }
      }
    }
  }
  for (int i = 0; i < indexes.size(); i++) {
    for (int t = 0; t < 3; t++) {
      (*solutions)[i](t) = solution[i * 3 + t];
    }
  }
  return 0.0;
}

void Model::GetControlIndexes(const std::vector<int>& indexes, std::vector<int>* control_indexes) {
  control_indexes->clear();
  const int kSamplingStepForSolving = 1;
  if (indexes.size() < kSamplingStepForSolving * 5) {
    for (int i = 0; i < indexes.size(); i++) {
      control_indexes->emplace_back(i);
    }
  } else {
    for (int i = 0; i < indexes.size(); i += kSamplingStepForSolving) {
      control_indexes->emplace_back(i);
    }
    if (control_indexes->back() != indexes.size() - 1) {
      control_indexes->emplace_back(indexes.size() - 1);
    }
  }
}

void Model::Update3DRadius(double extend_2d_radius) {
  int view_idx = 0;
  int n_views = views_.size();
  auto tree = std::make_unique<SpanningTree>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_, 4);
#pragma omp parallel for private(view_idx) default(none) shared(n_views, points_, tangs_, tang_scores_, hope_dist_, views_, tree)
  for (view_idx = 0; view_idx < n_views; view_idx++) {
    const auto& view = views_[view_idx];
    view->CacheMatchings(points_, tangs_, tang_scores_, tree->Edges(), hope_dist_, false);
  }
  /*nenglun*/
  points_radius_3d_ = std::vector<double>(points_.size(), 0);
  //calculate radius here
  bool calc_radius_by_view_weight = true;
  if (calc_radius_by_view_weight) {
    for (int u = 0; u < n_points_; u++) {
      double u_radius = 0.0;
      std::vector<std::pair<double, View*>> weighted_views;
      GetSinglePointViewWeight(u, &weighted_views);
      for (const auto& pr : weighted_views) {
        View* view = pr.second;
        Eigen::Vector3d cam_p = view->camera_.World2Cam(points_[u]);
        double focal_length = (view->camera_.GetFocalLengthX() + view->camera_.GetFocalLengthY()) / 2;
        double radius_2d = view->matching_radius_[u] + extend_2d_radius;
        double radius_w = radius_2d * cam_p[2] / focal_length;
        u_radius += radius_w * pr.first;
      }
      points_radius_3d_[u] = u_radius;
    }
  }
  else {
    LOG(FATAL) << "deprecated.";
    for (int u = 0; u < n_points_; u++) {
      std::vector<std::pair<double, double>> radius_scores;
      for (const auto &view : views_) {
        if (view->GetMatchingRadius(u) > 0.0) {
          Eigen::Vector3d cam_p = view->camera_.World2Cam(points_[u]);
          double focal_length = (view->camera_.GetFocalLengthX() + view->camera_.GetFocalLengthY()) / 2;
          double radius_w = view->matching_radius_[u] * cam_p[2] / focal_length;
          std::pair<double, double> pr;
          pr.first = radius_w;
          // LOG(INFO) << "..." << view->matched_std_dev_[view->matching_indexes_[u]] / hope_dist_;
          // LOG(INFO) << "+++" << (view->matching_pixels_[u] - view->camera_.World2Image(points_[u])).norm();
          pr.second = std::exp(-view->matched_std_dev_[view->matching_indexes_[u]] / hope_dist_) *
                      std::exp(-(view->matching_pixels_[u] - view->camera_.World2Image(points_[u])).norm());
          radius_scores.emplace_back(pr);
        }
      }
      if (radius_scores.empty()) {
        points_radius_3d_[u] = 0.0;
        continue;
      }
      std::sort(radius_scores.begin(), radius_scores.end());
      radius_scores.resize(int(radius_scores.size() * 0.5 + 1.0));
      std::reverse(radius_scores.begin(), radius_scores.end());
      radius_scores.resize(int(radius_scores.size() * 0.5 + 1.0));
      CHECK(!radius_scores.empty());
      double weight_sum = 0.0;
      double estimated_radius = 0.0;
      for (const auto &pr : radius_scores) {
        weight_sum += pr.second;
        estimated_radius += pr.first * pr.second;
      }  
      // points_radius_3d_[u] = std::accumulate(tmp_radius.begin(), tmp_radius.end(), 0.0) / tmp_radius.size();
      points_radius_3d_[u] = estimated_radius / weight_sum;
      // LOG(INFO) << estimated_radius / weight_sum;
    }
  }
}

// Pls make sure 3d radius is available.
void Model::RadiusSmoothing(Graph* graph) {
  LOG(INFO) << "RadiusSmoothing: Begin";
  bool need_delete_graph = (graph == nullptr);
  if (graph == nullptr) {
    graph = new IronTown(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_);
  }

  // Choose reliable radius. We assume that the radius near a junction is not reliable.
  std::vector<double> near_junction_ratio(n_points_, 1e9);
  std::vector<std::vector<int>> paths;
  graph->GetPaths(&paths);
  for (const auto& path : paths) {
    for (int i = 0; i < path.size(); i++) {
      int u = path[i];
      near_junction_ratio[u] = std::min(near_junction_ratio[u], double(std::min(i, int(path.size()) - 1 - i)) / path.size());
    }
  }
  for (double& ratio : near_junction_ratio) {
    if (ratio > 1.5) {
      ratio = 0.0;
    }
  }

  // Radius smoothing.
  std::vector<std::vector<int>> linked_paths;
  graph->GetLinkedPaths(points_, &linked_paths);

  std::vector<unsigned> coord_r;
  std::vector<unsigned> coord_c;
  std::vector<double> values;
  std::vector<double> B;
  auto AddValue = [&coord_r, &coord_c, &values](int r, int c, double val) {
    coord_r.push_back(r);
    coord_c.push_back(c);
    CHECK(!std::isnan(val));
    values.push_back(val);
  };

  CHECK_EQ(n_points_, points_.size());
  int idx = -1;
  // Data term.
  for (int u = 0; u < n_points_; u++) {
    idx++;
    // if (graph->Degree(u) > 2) {
    //   AddValue(idx, u, 1e2);
    //   B.push_back(points_radius_3d_[u] * 1e2);
    // } else {
    AddValue(idx, u, 1.0);
    B.push_back(points_radius_3d_[u]);
    // }
  }
  // Smoothness term.
  const double smoothing_weight = radius_smoothing_weight_ < -1e-5 ? 1e3 : radius_smoothing_weight_;
  for (const auto& path : linked_paths) {
    std::vector<double> xs;
    std::vector<double> ys;
    double n_valid_points = 0.0;
    double current_x = 0.0;
    for (int i = 0; i < path.size(); i++) {
      int u = path[i];
      if (i > 0) {
        current_x += (points_[u] - points_[path[i - 1]]).norm();
      }
      if (near_junction_ratio[u] < 0.1) {
        continue;
      }
      xs.emplace_back(current_x);
      ys.emplace_back(points_radius_3d_[u]);
      n_valid_points += 1.0;
    }
    double a, b;
    Utils::LinearFitting(xs, ys, &a, &b);
    double relative_error = 0.0;
    for (int i = 0; i < xs.size(); i++) {
      double x = xs[i];
      double y = x * a + b;
      double error = (y - ys[i]) / ys[i];
      relative_error += error * error;
    }
    relative_error = n_valid_points > 1e-3 ? std::sqrt(relative_error / n_valid_points) : 1e9;

    double current_smoothing_weight = smoothing_weight;
    LOG(INFO) << "average relative error: " << relative_error;
    if (relative_error < 0.15) {
      current_smoothing_weight = 1e3;
    }
    for (int i = 1; i + 1 < path.size(); i++) {
      idx++;
      AddValue(idx, path[i], 2.0 * current_smoothing_weight);
      AddValue(idx, path[i - 1], -current_smoothing_weight);
      AddValue(idx, path[i + 1], -current_smoothing_weight);
      B.push_back(0.0);
    }
  }

  Utils::SolveLinearSqrLSQR(
      (int) B.size(), n_points_, coord_r, coord_c, values, B, points_radius_3d_);
  if (need_delete_graph) {
    delete (IronTown*) graph;
  }
  LOG(INFO) << "RadiusSmoothing: End";
}

// Pls make sure matchings have been already cached.
void Model::GetSinglePointViewWeight(int pt_idx,
                                     std::vector<std::pair<double, View*>>* weighted_views) {

  ViewWeightDistributionType current_tp = view_weight_distribution_type_;
  if (current_tp == ViewWeightDistributionType::STD_DEV_THRESHOLD) {
    weighted_views->clear();
    std::vector<double> std_devs;
    for (const auto& view : views_) {
      if (view->matching_indexes_[pt_idx] == -1) {
        continue;
      }
      // double std_dev = view->matched_std_dev_[view->matching_indexes_[pt_idx]];
      // if (std_dev < 5.0 * hope_dist_) {
      // if (!view->is_world_point_self_occluded_[pt_idx]) {
      if (!view->is_self_occluded_[view->matching_indexes_[pt_idx]]) {
        weighted_views->emplace_back(1.0, view);
      }
      // LOG(INFO) << "ratio: " << weighted_views->size() / double(views_.size());
    }
    if (weighted_views->size() < 3) {
      current_tp = ViewWeightDistributionType::STD_DEV;
    } else {
      double weight_sum = 0.0;
      for (auto& pr : *weighted_views) {
        weight_sum += pr.first;
      }
      for (auto& pr : *weighted_views) {
        pr.first /= weight_sum;
      }
      return;
    }
  }
  if (current_tp == ViewWeightDistributionType::STD_DEV) {
    weighted_views->clear();
    std::vector<double> std_devs;
    for (const auto& view : views_) {
      if (view->matching_indexes_[pt_idx] == -1) {
        continue;
      }
      // double std_dev = view->matched_std_dev_[view->matching_indexes_[pt_idx]];
      // CHECK_LT(view->matching_indexes_[pt_idx], view->matched_std_dev_.size());
      // CHECK_GE(view->matching_indexes_[pt_idx], 0);
      double current_std_dev = std::max(view->matched_std_dev_[view->matching_indexes_[pt_idx]], hope_dist_);
      double sigma = (3.0 * hope_dist_);
      double weight = std::exp(-(current_std_dev * current_std_dev / (2 * sigma * sigma)));
      if (weight > 1e-7 && !std::isnan(weight)) {
        std_devs.emplace_back(view->matched_std_dev_[view->matching_indexes_[pt_idx]]);
      }
    }
    if (std_devs.size() > 3) {
      std::sort(std_devs.begin(), std_devs.end());
      // double base_std_dev = std::max(hope_dist_, std_devs[std_devs.size() / 3]);
      for (const auto& view : views_) {
        if (view->matching_indexes_[pt_idx] == -1) {
          continue;
        }
        CHECK_LT(view->matching_indexes_[pt_idx], view->matched_std_dev_.size());
        CHECK_GE(view->matching_indexes_[pt_idx], 0);
        double current_std_dev = std::max(view->matched_std_dev_[view->matching_indexes_[pt_idx]], hope_dist_);
        double sigma = (3.0 * hope_dist_);
        double weight = std::exp(-(current_std_dev * current_std_dev / (2 * sigma * sigma)));
        // if (current_std_dev > hope_dist_ + 1e-7) {
        //   LOG(INFO) << current_std_dev << " " << hope_dist_ << " weight: " << weight;
        // }
        if (weight > 1e-7 && !std::isnan(weight) && current_std_dev < 10.0 * hope_dist_) {
          weighted_views->emplace_back(weight, view);
        }
      }
      std::sort(weighted_views->begin(), weighted_views->end());
      std::reverse(weighted_views->begin(), weighted_views->end());
      if (weighted_views->size() >= 10) {
        weighted_views->resize(std::max(1, int(weighted_views->size()) * 3 / 4));
      }
      double weight_sum = 0.0;
      for (auto& pr : *weighted_views) {
        weight_sum += pr.first;
      }
      for (auto& pr : *weighted_views) {
        pr.first /= weight_sum;
      }
      return;
    }
    else {
      current_tp = ViewWeightDistributionType::UNIFORM;
    }
  }
  if (current_tp == ViewWeightDistributionType::UNIFORM) {
    weighted_views->clear();
    double weight_base = 0.0;
    for (const auto& view : views_) {
      if (view->GetMatchingRadius(pt_idx) > 0.0) {
        weighted_views->emplace_back(view->GetMatchingRadius(pt_idx), view);
        weight_base += view->GetMatchingRadius(pt_idx);
      }
    }
    for (auto& pr : *weighted_views) {
      pr.first /= weight_base;
    }
    return;
  }
}

double Model::GetSinglePoint3dRadius(int pt_idx) {
  std::vector<std::pair<double, View*>> weighted_views;
  GetSinglePointViewWeight(pt_idx, &weighted_views);
  double r = 0.0;
  for (const auto& pr : weighted_views) {
    View* view = pr.second;
    Eigen::Vector3d cam_p = view->camera_.World2Cam(points_[pt_idx]);
    double focal_length = (view->camera_.GetFocalLengthX() + view->camera_.GetFocalLengthY()) / 2;
    // double radius_2d = final_refine ?
    //     view->extractor_->RadiusAt(view->camera_.World2Image(points_[u])(0), view->camera_.World2Image(points_[u])(1)) :
    //     view->matching_radius_[u];
    double radius_2d = view->matching_radius_[pt_idx];
    double radius_w = radius_2d * cam_p[2] / focal_length;
    r += radius_w * pr.first;
  }
  return r;
}

double Model::UpdatePointsGlobal() {
  LOG(FATAL) << "Deprecated.";
  return 0.0;
}

void Model::MoveModelPointsToCenter() {
  Eigen::Vector3d mean_point = Eigen::Vector3d::Zero();
  for (const auto pt : points_) {
    mean_point += pt;
  }
  CHECK(!points_.empty());
  mean_point /= (double) points_.size();
  for (auto& pt : points_) {
    pt -= mean_point;
  }
  UpdateOctree();
  for (auto& view : views_) {
    view->UpdateRT(view->R_, view->T_ + view->R_ * mean_point);
  }
  double ave_depth = 0.0;
  for (const auto& pt : points_) {
    ave_depth += views_.front()->camera_.World2Cam(pt)(2);
  }
  ave_depth /= points_.size();
  for (auto &pt : points_) {
    pt /= ave_depth;
  }
  for (auto& view : views_) {
    view->UpdateRT(view->R_, view->T_ / ave_depth);
  }
}

void Model::UpdateViews() {
  int view_idx = 0;
#pragma omp parallel for private(view_idx) default(none) shared(views_)
  // for (const auto &view : views_) {
  for (view_idx = 0; view_idx < views_.size(); view_idx++) {
    views_[view_idx]->UpdateRTCeres(points_, points_history_, 1);
    // views_[view_idx]->UpdateRT(points_, points_history_, false, 1);
    // view->UpdateRTCeres(points_);
    // view->OutDebugImage("debug", points_, true, 10);
  }
  last_R_ = views_.back()->R_;
  last_T_ = views_.back()->T_;

  views_.back()->OutDebugImage("last", points_, global_data_pool_);
  StopWatch stop_watch;
  while (views_.size() > ::kMaxKeyFrameN) {
    bool remove_by_time_scale = true;
    bool remove_by_similarity = false;
    bool remove_by_std_dev = false;
    if (remove_by_time_scale) {
      std::vector<std::tuple<int, double, int> > marks;
      for (int idx = 1; idx + kSlidingWindowN < views_.size(); idx++) {
        auto& view = views_[idx];
        if (view->is_necessary_) {
          continue;
        }
        int min_dis = 0;
        if (idx > 0) {
          min_dis += view->time_stamp_ - views_[idx - 1]->time_stamp_;
        }
        if (idx + 1 < views_.size()) {
          min_dis += views_[idx + 1]->time_stamp_ - view->time_stamp_;
        }
        // double score = -view->CalcBidirectionProjectionError(points_);
        double score = 0.0;
        marks.emplace_back(min_dis, score, idx);
      }
      if (marks.empty()) {
        LOG(WARNING) << "Can't remove views.";
        break;
      } else {
        std::sort(marks.begin(), marks.end());
        views_.erase(views_.begin() + std::get<2>(marks.front()));
      }
    }
    else if (remove_by_similarity) {
      // Remove by space scale.
      const int kHopeSampledPointN = 500;
      int random_step = std::max(2, n_points_ / kHopeSampledPointN * 2);
      std::vector<std::pair<double, int>> similarity_and_indexes;
      std::vector<int> sampled_u;
      for (int u = std::rand() % random_step; u < n_points_; u += std::rand() % random_step) {
        sampled_u.emplace_back(u);
      }
      for (int view_i = 1; view_i + kSlidingWindowN < views_.size(); view_i++) { // Don't remove the first one and the last one.
        double i_similarity = 0.0;
        for (int view_j = 1; view_j + kSlidingWindowN < views_.size(); view_j++) {
          if (view_i == view_j) {
            continue;
          }
          double valid_base = 0.0;
          double similarity = 0.0;
          for (int u : sampled_u) {
            if (views_[view_i]->GetMatchingRadius(u) < 0.0 || views_[view_j]->GetMatchingRadius(u) < 0.0) {
              if (views_[view_i]->GetMatchingRadius(u) > 0.0 && views_[view_j]->GetMatchingRadius(u) < 0.0) {
                valid_base += 1.0;
                similarity += -1.0;
              }
              continue;
            }
            Eigen::Vector3d ray_i = (views_[view_i]->R_.inverse() * views_[view_i]->GetMatching(u).first).normalized();
            Eigen::Vector3d ray_j = (views_[view_j]->R_.inverse() * views_[view_j]->GetMatching(u).first).normalized();
            valid_base += 1.0;
            similarity += ray_i.dot(ray_j);
          }
          if (valid_base < 1e-5) {
            i_similarity += -1.0;
          } else {
            i_similarity += similarity / valid_base;
          }
        }
        similarity_and_indexes.emplace_back(i_similarity, view_i);
      }
      std::sort(similarity_and_indexes.begin(), similarity_and_indexes.end());
      views_.erase(views_.begin() + similarity_and_indexes.back().second);
    } else if (remove_by_std_dev) {
      LOG(FATAL) << "here.";
    } else {
      LOG(FATAL) << "No such method to remove views.";
    }
  }
  LOG(INFO) << "View size: " << views_.size();
  LOG(INFO) << "Remove views duration: " << stop_watch.TimeDuration();
  // Update average radius;
  for (const auto& view : views_) {
    double depth_sum = 0.0;
    double n_view_points = 0.0;
    for (const auto& pt : points_) {
      if (view->WorldPointOutImageRange(pt)) {
        continue;
        depth_sum += view->camera_.World2Cam(pt)(2);
        n_view_points += 1.0;
      }
    }
    CHECK_LT(n_view_points, 1e-5);
    view->average_depth_ = depth_sum / n_view_points;
  }
  // Debug.
}

void Model::UpdateTangs(Graph* graph) {
  if (graph == nullptr) {
    return;
  }
  const auto& edges = graph->Edges();
  CHECK_EQ(edges.size(), n_points_);
  tangs_.resize(n_points_, Eigen::Vector3d(1.0, 0.0, 0.0));
  tang_scores_.resize(n_points_, 0.0);
  
  bool only_use_neighbors = false;
  if (only_use_neighbors) {
    for (int u = 0; u < n_points_; u++) {
      if (edges[u].size() > 2 || edges[u].size() == 0) {
        tang_scores_[u] = 0.0;
        tangs_[u] = Eigen::Vector3d(1.0, 0.0, 0.0);
      }
      else if (edges[u].size() == 1) {
        tang_scores_[u] = 1.0;
        tangs_[u] = (points_[edges[u].front().first] - points_[u]).normalized();
      }
      else {
        tang_scores_[u] = 1.0;
        tangs_[u] = (points_[edges[u].front().first] - points_[edges[u].back().first]).normalized();
      }
    }
  }
  else {
    // Directed tangs!!!
    std::vector<std::vector<int>> paths;
    graph->GetPaths(&paths);
    for (const auto& path : paths) {
      int path_n = path.size();
      for (int i = 0; i < path_n; i++) {
        if (edges[path[i]].size() > 2) {
          continue;
        }
        int a = std::max(0, i - 3);
        int b = std::min(path_n - 1, i + 3);
        int dis_to_junction_a = std::min(a, path_n - a - 1);
        int dis_to_junction_b = std::min(b, path_n - b - 1);
        if (dis_to_junction_a > dis_to_junction_b) {
          std::swap(a, b);
        }
        tangs_[path[i]] = (points_[path[a]] - points_[path[b]]).normalized();
        tang_scores_[path[i]] = 1.0;
      }
    }
  }
}

SelfDefinedGraph* Model::MergeJunction() {
  auto graph = std::make_unique<IronTown>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_);
  std::vector<std::vector<int>> paths;
  graph->GetPaths(&paths);
  const int kTooShortPathSize = 3;
  std::vector<int> fa(n_points_, 0);
  std::vector<int> size(n_points_, 1);
  std::vector<Eigen::Vector3d> sum_points = points_;
  std::iota(fa.begin(), fa.end(), 0);
  std::function<int(int)> FindRoot;
  FindRoot = [&fa, &FindRoot](int a) -> int {
    return fa[a] == a ? a : (fa[a] = FindRoot(fa[a]));
  };
  for (const auto& path: paths) {
    CHECK(!path.empty());
    if (path.size() > kTooShortPathSize || graph->Degree(path.front()) < 2 ||  graph->Degree(path.back()) < 2) {
      continue;
    }
    int u = FindRoot(path.front());
    for (int v_tmp : path) {
      int v = FindRoot(v_tmp);
      if (u == v) {
        continue;
      }
      size[u] += size[v];
      sum_points[u] += sum_points[v];
      fa[v] = u;
    }
  }
  std::vector<int> new_idx(n_points_, -1);
  int new_idx_cnt = 0;
  for (int u = 0; u < n_points_; u++) {
    if (FindRoot(u) == u) {
      new_idx[u] = new_idx_cnt++;
    }
  }

  // Solve.
  std::vector<unsigned> coord_r;
  std::vector<unsigned> coord_c;
  std::vector<double> values;
  std::vector<double> B;
  auto AddValue = [&coord_r, &coord_c, &values](int r, int c, double val) {
    coord_r.push_back(r);
    coord_c.push_back(c);
    CHECK(!std::isnan(val));
    values.push_back(val);
  };

  // Calc weights.
  std::vector<int> dis(n_points_, -1);
  std::vector<int> que;
  for (int u = 0; u < n_points_; u++) {
    if (size[FindRoot(u)] <= 1) {
      continue;
    }
    dis[FindRoot(u)] = 0;
    que.push_back(u);
  }
  for (int l = 0; l < que.size(); l++) {
    int u = que[l];
    const auto& edges = graph->Edges()[u];
    for (const auto& edge : edges) {
      int v = edge.first;
      if (dis[FindRoot(v)] > -1) {
        continue;
      }
      CHECK_EQ(FindRoot(v), v);
      dis[FindRoot(v)] = dis[FindRoot(u)] + 1;
      que.push_back(v);
    }
  }

  int idx = -1;
  StopWatch stop_watch;
  // Smoothness.
  for (const auto& path : paths) {
    for (int i = 1; i + 1 < path.size(); i++) {
      for (int t = 0; t < 3; t++) {
        idx++;
        AddValue(idx, new_idx[FindRoot(path[i])] * 3 + t, -2.0 * hope_dist_weight_);
        AddValue(idx, new_idx[FindRoot(path[i - 1])] * 3 + t, hope_dist_weight_);
        AddValue(idx, new_idx[FindRoot(path[i + 1])] * 3 + t, hope_dist_weight_);
        B.push_back(0.0);
      }
    }
  }
  // Position.
  for (int u = 0; u < n_points_; u++) {
    if (FindRoot(u) != u) {
      continue;
    }
    // double weight = size[u] * size[u];
    // double weight = 1.0;
    // CHECK(dis[u] >= 0);
    if (dis[u] < 0) {
      dis[u] = (1 << 20);
    }
    double weight = 10.0;
    if (dis[u] > 0) {
      double mu = dis[u] * 0.08;
      weight = mu / (mu + 1.0);
    }
    Eigen::Vector3d hope_pos = sum_points[u] / (double) size[u];
    for (int t = 0; t < 3; t++) {
      idx++;
      AddValue(idx, new_idx[u] * 3 + t, weight);
      B.push_back(hope_pos(t) * weight);
    }
  }

  std::vector<double> solution(new_idx_cnt * 3);
  for (int i = 0; i < new_idx_cnt; i++) {
    for (int t = 0; t < 3; t++) {
      solution[i * 3 + t] = 0.0;
    }
  }
  Utils::SolveLinearSqrLSQR((int) B.size(), new_idx_cnt * 3, coord_r, coord_c, values, B, solution);

  points_.resize(new_idx_cnt);
  n_points_ = new_idx_cnt;
  for (int i = 0; i < new_idx_cnt; i++) {
    for (int t = 0; t < 3; t++) {
      points_[i](t) = solution[i * 3 + t];
    }
  }
  UpdateOctree();
  UpdateTangs();

  std::vector<std::vector<std::pair<int, double>>> new_edges(n_points_); // n_points_ here is new.
  for (int u = 0; u < graph->Edges().size(); u++) {
    for (const auto& pr : graph->Edges()[u]) {
      int v = pr.first;
      if (FindRoot(u) == FindRoot(v)) {
        continue;
      }
      new_edges[new_idx[FindRoot(u)]].emplace_back(new_idx[FindRoot(v)], pr.second);
    }
  }
  std::vector<std::vector<int>> new_paths;
  for (const auto& path : paths) {
    if (path.size() > kTooShortPathSize || graph->Degree(path.front()) < 2 ||  graph->Degree(path.back()) < 2) {
      continue;
    }
    new_paths.emplace_back();
    for (int u : path) {
      new_paths.back().push_back(new_idx[FindRoot(u)]);
    }
  }
  return new SelfDefinedGraph(points_, new_edges, new_paths);
}

IronTown* Model::RadiusBasedMergeJunction() {
  UpdateOctree();
  auto graph = std::make_unique<IronTown>(points_,
                                          octree_,
                                          hope_dist_ * topology_searching_radius_,
                                          hope_dist_,
                                          &points_radius_3d_);
  UpdateTangs(graph.get());

  // Estimate world radius.
  Update3DRadius(radius_dilation_);
  std::vector<double> rs = points_radius_3d_;
  // for (auto& r : rs) { r *= 1.5; }
  Update3DRadius(0.0);
  std::vector<int> banned(n_points_, 0);
  std::vector<int> vis(n_points_, 0);
  std::vector<double> dis(n_points_, 0.0);
  int timestamp = 0;
  for (int u = 0; u < n_points_; u++) {
    if (graph->Degree(u) <= 2) {
      continue;
    }
    banned[u] = 1;
    // Search U.
    vis[u] = ++timestamp;
    dis[u] = 0.0;
    auto cmp = [&dis](int a, int b) {
      return dis[a] < dis[b] - 1e-9 || (dis[a] < dis[b] + 1e-9 && a < b);
    };
    std::set<int, decltype(cmp)> st(cmp);
    st.insert(u);
    auto Intersect = [&st, &graph, &vis, &rs, timestamp, this]() {
      std::set<int> margins;
      for (auto u : st) {
        for (const auto& pr : graph->Edges()[u]) {
          int v = pr.first;
          if (vis[v] != timestamp) {
            margins.emplace(v);
          }
        }
      }

      for (auto iter_u = margins.begin(); iter_u != margins.end(); iter_u++) {
        const auto& pt_u = points_[*iter_u];
        double r_u = rs[*iter_u];
        for (auto iter_v = std::next(iter_u); iter_v != margins.end(); iter_v++) {
          const auto& pt_v = points_[*iter_v];
          double r_v = rs[*iter_v];
          if ((pt_u - pt_v).norm() < r_u + r_v) {
            return true;
          }
        }
      }

      return false;
    };

    while (Intersect()) {
      int current_u = *st.begin();
      if (dis[current_u] > hope_dist_ * 15.0) {
        break;
      }
      st.erase(st.begin());
      for (const auto& pr : graph->Edges()[current_u]) {
        int v = pr.first;
        // double new_dis = dis[current_u] + (points_[v] - points_[current_u]).norm();
        double new_dis = (points_[v] - points_[u]).norm();
        if (graph->Edges()[v].size() <= 1) {
          LOG(INFO) << "new_dis: " << new_dis;
          LOG(INFO) << "rs[v]: " << rs[v];
          LOG(INFO) << "true_r[v]: " << points_radius_3d_[v];
        }
        if (graph->Edges()[v].size() <= 1 && new_dis > points_radius_3d_[v] + points_radius_3d_[u]) {
          // Don't erase leaves.
          continue;
        }
        if (vis[v] != timestamp) {
          banned[v] = 1;
          vis[v] = timestamp;
          st.erase(v);
          dis[v] = new_dis;
          st.insert(v);
        }
      }
    }
  }

  std::vector<int> fa(n_points_, 0);
  std::iota(fa.begin(), fa.end(), 0);
  std::function<int(int)> FindRoot;
  FindRoot = [&fa, &FindRoot](int a) -> int {
    return fa[a] == a ? a : (fa[a] = FindRoot(fa[a]));
  };
  const auto& edges = graph->Edges();
  std::vector<std::map<int, Eigen::Vector3d>> outs(n_points_);
  for (int u = 0; u < n_points_; u++) {
    if (!banned[u]) {
      continue;
    }
    for (const auto& pr : edges[u]) {
      int v = pr.first;
      if (!banned[v]) {
        outs[u].emplace(v, (points_[u] - points_[v]).normalized());
      }
    }
  }
  for (int u = 0; u < n_points_; u++) {
    if (!banned[u]) {
      continue;
    }
    std::vector<int> neighbors;
    for (const auto& pr : graph->Edges()[u]) {
      // octree_->SearchingR(rs[u], points_[u], neighbors);
      // for (int v : neighbors) {
      int v = pr.first;
      if (!banned[v] || FindRoot(u) == FindRoot(v)) {
        continue;
      }
      if (outs[fa[u]].size() > outs[fa[v]].size()) {
        for (auto node : outs[fa[v]]) {
          outs[fa[u]].emplace(node);
        }
        fa[fa[v]] = fa[u];
      }
      else {
        for (auto node : outs[fa[u]]) {
          outs[fa[v]].emplace(node);
        }
        fa[fa[u]] = fa[v];
      }
    }
  }

  std::vector<int> maps(n_points_, -1);
  int new_n_points = 0;
  std::vector<Eigen::Vector3d> new_points;
  std::vector<double> new_radius;

  // Recover
  bool recover = false;
  if (recover) {
    for (int u = 0; u < n_points_; u++) {
      if (banned[u] == 0 || FindRoot(u) != u) {
        continue;
      }
      if (outs[u].size() < 3) {
        continue;
      }
      bool is_valid = true;
      for (const auto& pr_a : outs[u]) {
        bool is_current_valid = false;
        for (const auto& pr_b : outs[u]) {
          if (pr_a.second.dot(pr_b.second) < 0.3) {
            is_current_valid = true;
            break;
          }
        }
        if (!is_current_valid) {
          is_valid = false;
          break;
        }
      }
      if (!is_valid) { // Recover
        banned[u] = -1;
      }
      double max_distance = 0.0;
      for (const auto& pr_a : outs[u]) {
        for (const auto& pr_b : outs[u]) {
          max_distance = std::max(max_distance, (points_[pr_a.first] - points_[pr_b.first]).norm());
        }
      }
      if (max_distance > 30.0 * hope_dist_) {
        banned[u] = -1;
      }
    }

    for (int u = 0; u < n_points_; u++) {
      FindRoot(u);
    }

    for (int u = 0; u < n_points_; u++) {
      if (banned[fa[u]] == -1) {
        outs[u].clear();
        banned[u] = -1;
        fa[u] = u;
      }
    }
    for (int u = 0; u < n_points_; u++) {
      if (banned[u] == -1) {
        banned[u] = 0;
      }
    }

  }

  for (int u = 0; u < n_points_; u++) {
    if (!banned[u]) {
      CHECK_EQ(FindRoot(u), u);
      maps[u] = new_n_points++;
      new_points.emplace_back(points_[u]);
      new_radius.emplace_back(points_radius_3d_[u]);
    }
  }

  // Add new junction
  std::vector<std::vector<std::pair<int, double>>> new_edges(new_n_points);
  for (int u = 0; u < n_points_; u++) {
    if (!banned[u] || FindRoot(u) != u) {
      continue;
    }
    if (outs[u].size() == 0) {
      continue;
    }
    if (outs[u].size() <= 2) {
      maps[u] = new_n_points++;
      new_points.emplace_back(points_[u]);
      new_radius.emplace_back(points_radius_3d_[u]);
      new_edges.emplace_back();
      continue;
    }

    // Solve linear equation.
    std::vector<unsigned> coord_r;
    std::vector<unsigned> coord_c;
    std::vector<double> values;
    std::vector<double> B;
    auto AddValue = [&coord_r, &coord_c, &values](int r, int c, double val) {
      coord_r.push_back(r);
      coord_c.push_back(c);
      CHECK(!std::isnan(val));
      values.push_back(val);
    };
    int idx = -1;
    double radius_sum = 0.0;
    for (const auto& pr : outs[u]) {
      int v = pr.first;
      radius_sum += points_radius_3d_[v];
      Eigen::Vector3d o = points_[v];
      Eigen::Vector3d t = tangs_[v];
      Eigen::Vector3d n0, n1;
      Eigen::Vector3d t_x_o = t.cross(o);
      if (std::abs(t(0)) < 1e-9) {
        n0 = Eigen::Vector3d(0.0, -t(2), t(1));
        n0 /= n0.norm();
      }
      else {
        n0 = Eigen::Vector3d(t(1), -t(0), 0.0);
        n0 /= n0.norm();
      }
      n1 = t.cross(n0);
      idx++;
      for (int k = 0; k < 3; k++) {
        AddValue(idx, k, n0(k));
      }
      B.push_back(o.dot(n0));
      idx++;
      for (int k = 0; k < 3; k++) {
        AddValue(idx, k, n1(k));
      }
      B.push_back(o.dot(n1));
    }
    std::vector<double> solution = { 0.0, 0.0, 0.0 };
    Utils::SolveLinearSqrLSQR((int) B.size(), 3, coord_r, coord_c, values, B, solution);

    Eigen::Vector3d new_u = Eigen::Vector3d(solution[0], solution[1], solution[2]);
    
    bool is_new_u_ok = true;
    for (const auto& pr : outs[u]) {
      int v = pr.first;
      Eigen::Vector3d o = points_[v];
      Eigen::Vector3d t = tangs_[v]; // t is directed to junction
      if (pr.second.dot(t) < 0.0) {
        t = -t;
      }
      // double sin_error = (o - new_u).normalized().cross(t.normalized()).norm();
      //   if (sin_error > 0.4) {
      //   is_new_u_ok = false;
      // }
      double cos_val = (new_u - o).normalized().dot(t);
      LOG(INFO) << "cos val: " << cos_val;
      if (cos_val < 0.9) {
        is_new_u_ok = false;
      }
    }
    
    if (!is_new_u_ok || outs[u].size() > 14) {
      banned[u] = -1;
      continue;
    }
    
    CHECK_EQ(new_n_points, new_edges.size());
    bool dense_junction = (outs[u].size() > 5);
    CHECK(!outs[u].empty());
    maps[u] = new_n_points++;
    new_edges.emplace_back();
    new_points.emplace_back(new_u);
    new_radius.emplace_back(dense_junction ? -1.0 : -2.0); // Current radius is unavalible.
    
    for (const auto& pr : outs[u]) {
      int v = pr.first;
      Eigen::Vector3d o = points_[v];
      // const double out_radius = points_radius_3d_[v];
      int sampling_n = std::max(2, (int) std::ceil((o - new_u).norm() / hope_dist_));
      Eigen::Vector3d vec = (o - new_u).normalized();
      double t_step = (o - new_u).norm() / sampling_n;
      for (int i = 1; i < sampling_n; i++) {
        new_points.emplace_back(new_u + t_step * i * vec);
        new_radius.emplace_back(dense_junction ? -1.0 : -2.0); // Current radius is unavalible.
        new_n_points++;
        new_edges.emplace_back();
        int a = i == 1 ? maps[u] : new_n_points - 2;
        int b = new_n_points - 1;
        new_edges[a].emplace_back(b, t_step);
        new_edges[b].emplace_back(a, t_step);
      }
      CHECK_GE(maps[FindRoot(v)], 0);
      CHECK_LT(maps[FindRoot(v)], new_n_points);
      new_edges[new_n_points - 1].emplace_back(maps[FindRoot(v)], t_step);
      new_edges[maps[FindRoot(v)]].emplace_back(new_n_points - 1, t_step);
    }
  }
  
  for (int u = 0; u < n_points_; u++) {
    if (banned[FindRoot(u)] < 0) {
      banned[u] = -1;
      maps[u] = new_n_points++;
      outs[u].clear();
      new_points.emplace_back(points_[u]);
      new_radius.emplace_back(points_radius_3d_[u]);
      new_edges.emplace_back();
    }
  }
  for (int u = 0; u < n_points_; u++) {
    if (banned[u] < 0) {
      fa[u] = u;
    }
  }
  
  CHECK_EQ(new_n_points, new_edges.size());
  CHECK_EQ(new_n_points, new_points.size());
  CHECK_EQ(new_points.size(), new_radius.size());
  points_ = std::move(new_points);
  points_radius_3d_ = std::move(new_radius);
  n_points_ = points_.size();
  UpdateOctree();
  for (int u = 0; u < maps.size(); u++) {
    int fu = FindRoot(u);
    // CHECK_GE(maps[fu], 0);
    if (maps[fu] < 0) {
      continue;
    }
    for (const auto& pr : graph->Edges()[u]) {
      int v = pr.first;
      int fv = FindRoot(v);
      // CHECK_GE(maps[fv], 0);
      if (maps[fv] < 0) {
        continue;
      }
      if (maps[fu] > maps[fv] && outs[fu].size() <= 2 && outs[fv].size() <= 2) {
        CHECK_GE(maps[fu], 0);
        CHECK_GE(maps[fv], 0);
        CHECK_LT(maps[fu], points_.size());
        CHECK_LT(maps[fv], points_.size());
        new_edges[maps[fu]].emplace_back(maps[fv], (points_[maps[fu]] - points_[maps[fv]]).norm());
        new_edges[maps[fv]].emplace_back(maps[fu], (points_[maps[fu]] - points_[maps[fv]]).norm());
      }
    }
  }
  IronTown* ret_graph = new IronTown(points_, new_edges);
  RecoverUncertainRadius((Graph*) ret_graph);
  return ret_graph;
}

void Model::RecoverUncertainRadius(Graph* graph) {
  // Cache matchings.
  int view_idx = 0;
  int n_views = views_.size();
  auto tree = std::make_unique<SpanningTree>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_, 4);
#pragma omp parallel for private(view_idx) default(none) shared(n_views, points_, tangs_, tang_scores_, hope_dist_, views_, tree)
  for (view_idx = 0; view_idx < n_views; view_idx++) {
    const auto& view = views_[view_idx];
    view->CacheMatchings(points_, tangs_, tang_scores_, tree->Edges(), hope_dist_, false);
  }

  std::vector<double> before_3d_radius = points_radius_3d_;
  Update3DRadius();
  for (int i = 0; i < n_points_; i++) {
    if (std::abs(before_3d_radius[i] + 2.0) < 1e-8) { // If the junction point is not dense.
      before_3d_radius[i] = points_radius_3d_[i];
    }
  }

  points_radius_3d_ = before_3d_radius;
  // Main logic.
  CHECK(graph != nullptr);
  std::vector<std::vector<int>> paths;
  graph->GetLinkedPaths(points_, &paths);
  for (const auto& path : paths) {
    if (path.size() < 3) {
      continue;
    }
    int last_certain_pos = -1;
    std::vector<double> single_len(path.size() + 1);
    single_len[0] = (points_[path[0]] - points_[path[1]]).norm();
    for (int i = 1; i < path.size(); i++) {
      single_len[i] = (points_[path[i]] - points_[path[i - 1]]).norm();
      single_len[i] += single_len[i - 1];
    }
    single_len[path.size()] =
        (points_[path[path.size() - 1]] - points_[path[path.size() - 2]]).norm() + single_len[path.size() - 1];

    for (int idx = 0; idx < path.size(); idx++) {
      if (points_radius_3d_[path[idx]] > -1e-9 ||
          (points_radius_3d_[path[idx]] <= -1e-9 && idx + 1 == path.size())) {
        CHECK(last_certain_pos >= 0 || points_radius_3d_[path[idx]] > -1e-9);
        // double base_radius_l = last_certain_pos < 0 ?
        //     GetSinglePoint3dRadius(path[0]) : points_radius_3d_[path[last_certain_pos]];
        double base_radius_l = last_certain_pos < 0 ?
                               points_radius_3d_[path[idx]] : points_radius_3d_[path[last_certain_pos]];

        double base_radius_r = points_radius_3d_[path[idx]] > -1e-9 ?
                               points_radius_3d_[path[idx]] : points_radius_3d_[path[last_certain_pos]];
        int next_certain_pos = points_radius_3d_[path[idx]] > -1e-9 ? idx : path.size();

        for (int j = last_certain_pos + 1; j < next_certain_pos; j++) {
          points_radius_3d_[path[j]] =
              double(base_radius_l * (next_certain_pos - j) + base_radius_r * (j - last_certain_pos)) /
              double(next_certain_pos - last_certain_pos);
        }
        last_certain_pos = next_certain_pos;
      }
    }
  }
}

// Deprecated.
void Model::PathsSmoothing() {
  LOG(FATAL) << "Deprecated.";
  auto graph = std::make_unique<IronTown>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_);
  std::vector<std::vector<int>> paths;
  graph->GetPaths(&paths);
  std::vector<Eigen::Vector3d> new_points;
  for (const auto& path : paths) {
    if (path.size() < 3) {
      continue;
    }
    std::vector<Eigen::Vector3d> path_points;
    std::vector<Eigen::Vector3d> new_path_points;
    for (const auto idx : path) {
      path_points.emplace_back(points_[idx]);
    }
    Utils::SplineFitting(path_points, hope_dist_ * 10.0, &new_path_points);
    new_points.insert(new_points.end(), new_path_points.begin(), new_path_points.end());
    LOG(INFO) << new_points.size();
  }
  points_ = std::move(new_points);
  n_points_ = points_.size();
  UpdateOctree();
}

void Model::OutputFinalModel(Graph* graph, const std::string& out_file_name) {
  const auto& edges = graph->Edges();
  // Output Curves: Test.
  if (out_curve_format_ == "RAW") {
    std::vector<int> f(n_points_, -1);
    std::vector<int> que;
    std::vector<std::vector<Eigen::Vector3d> > curves;
    std::vector<std::vector<int>> paths;
    graph->GetPaths(&paths);
    for (const auto& path : paths) {
      curves.emplace_back();
      auto& curve = curves.back();
      for (int u : path) {
        curve.emplace_back(points_[u]);
      }
    }
    for (int u = 0; u < n_points_; u++) {
      if (edges[u].size() <= 2) {
        continue;
      }
      for (int i = 0; i < edges[u].size(); i++) {
        for (int j = i + 1; j < edges[u].size(); j++) {
          curves.emplace_back();
          curves.back().emplace_back(points_[u]);
          curves.back().emplace_back(points_[edges[u][i].first]);
          curves.back().emplace_back(points_[edges[u][j].first]);
        }
      }
    }
    Utils::OutputCurves(out_file_name + ".txt", curves);
  }
  else if (out_curve_format_ == "OBJ") {
    std::vector<std::pair<int, int>> out_edges;
    for (int u = 0; u < n_points_; u++) {
      for (const auto& pr : edges[u]) {
        out_edges.emplace_back(u, pr.first);
      }
    }
    Utils::OutputCurvesAsOBJ(out_file_name + ".obj", points_, out_edges);
  }
    /*nenglun*/
  else if(out_curve_format_ == "SWEEP") {
    std::vector<std::pair<int, int>> out_edges;
    for (int u = 0; u < n_points_; u++) {
      for (const auto& pr : edges[u]) {
        out_edges.emplace_back(u, pr.first);
      }
    }
    Utils::OutputCurvesAsOBJ(out_file_name + ".obj", points_, out_edges);

    std::vector<Eigen::MatrixXd> verts_list, verts_list_ave_r;
    std::vector<Eigen::MatrixXi> faces_list, faces_list_ave_r;
    std::vector<std::vector<int>> paths;
    graph->GetLinkedPaths(points_, &paths);
    // graph->GetPaths(&paths);
    // LOG(FATAL) << "here";

    std::vector<std::vector<double>> feed_radius(paths.size());
    for (int i = 0; i < paths.size(); i++) {
      for (int j = 0; j < paths[i].size(); j++) {
        feed_radius[i].emplace_back(points_radius_3d_[paths[i][j]]);
      }
    }
    
    RadiusSmoothing(graph);
    std::vector<std::vector<double>> feed_smooth_radius(paths.size());
    for (int i = 0; i < paths.size(); i++) {
      for (int j = 0; j < paths[i].size(); j++) {
        feed_smooth_radius[i].emplace_back(points_radius_3d_[paths[i][j]]);
      }
    }
    
    {
      SweepSurface ss;
      Eigen::MatrixXd verts;
      Eigen::MatrixXi faces;
      ss.GenSweepSurface(points_, paths, feed_smooth_radius, verts, faces);
      Utils::OutputTriMeshAsOBJ(out_file_name+"_mesh.obj", verts, faces);
    }
  }
  else {
    LOG(FATAL) << "No such out curve format.";
  }
}

void Model::RefinePoints() {
  solver_type_ = SolverType::CERES;
  StopWatch stop_watch;
  LOG(INFO) << "Refine: begin.";

  auto active_view_iter = views_.begin();
  // Recover views
  
  UpdateDataStructure();
  for (auto view_iter = view_pools_.begin(); view_iter != view_pools_.end(); view_iter++) {
    LOG(INFO) << "Refine view: " << view_iter->get()->time_stamp_;
    if (view_iter->get() == *active_view_iter) {
      active_view_iter++;
    }
    else {
      CHECK(view_iter != view_pools_.begin());
      std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> pose_candidates;
      // static.
      auto past_view_iter = std::prev(view_iter);
      pose_candidates.emplace_back((*past_view_iter)->R_, (*past_view_iter)->T_);
      for (auto initial_flow : std::vector<Eigen::Vector2d>
          { Eigen::Vector2d(0.0, 0.0), (*view_iter)->view_center_ - (*past_view_iter)->view_center_ }) {
        int track_state = view_iter->get()-> Track(pose_candidates,
                                                   points_,
                                                   points_history_,
                                                   tangs_,
                                                   tang_scores_,
                                                   curve_network_,
                                                   spanning_tree_,
                                                   octree_,
                                                   initial_flow);
        if (track_state >= 0) {
          break;
        }
      }
    }
  }
  views_.clear();
  for (auto& view : view_pools_) {
    views_.emplace_back(view.get());
  }
  // Clear matching cache.
  for (const auto& view : view_pools_) {
    view->ClearMatchingCache();
    view->ClearSelfOcclusionState();
  }

  shrink_error_weight_ = 0.1;
  hope_dist_weight_ = final_smoothing_weight_;
  topology_searching_radius_ = 5.0;

  UpdateDataStructure();
  UpdatePoints(true, curve_network_, spanning_tree_);

  Update3DRadius();
  LOG(INFO) << "Refine time duration: " << stop_watch.TimeDuration();
  
  // Output camera poses information.
  std::ofstream out_file("cameras.txt");
  for (const auto& view : views_) {
    out_file << view->R_ << std::endl << view->T_.transpose() << std::endl << std::endl;
  }
  out_file.close();
}

void Model::FinalProcess(const std::string& out_file_dname) {
  if (final_process_method_ == "MERGE_SHORT_PATHS") {
    RefinePoints();
    std::unique_ptr<SelfDefinedGraph> graph(MergeJunction());
    OutputFinalModel((Graph*) graph.get());
  }
  else if (final_process_method_ == "NAIVE") {
    RefinePoints();
    Update3DRadius(0.0);
    // UpdateOctree();
    // UpdateTangs();
    auto graph = std::make_unique<IronTown>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_);
    OutputFinalModel((Graph*) graph.get());
  }
  else if (final_process_method_ == "RADIUS_BASED_MERGE_JUNCTION") {
    RefinePoints();
    // auto naive_graph = std::make_unique<IronTown>(points_, octree_, hope_dist_ * topology_searching_radius_, hope_dist_);
    // OutputFinalModel((Graph*) naive_graph.get());
    // std::system("mv curves_mesh.obj curves_mesh_naive.obj");
    // std::system("mv curves.obj curves_naive.obj");
    // std::system("mv points.ply points_naive.ply");
    std::unique_ptr<IronTown> graph(RadiusBasedMergeJunction());
    UpdateDataStructure(false);
    UpdatePoints(true, graph.get(), spanning_tree_);
    OutputFinalModel((Graph*) graph.get());
  }
  else {
    LOG(FATAL) << "No such final process method.";
  }

  // Out view information.
  // for (const auto& view : views_) {
  //   view->OutVisualizationInfo();
  //   view->OutDebugImage(std::to_string(view->time_stamp_) + "_final.png", points_, false, -1);
  // }
}

void Model::UpdateSpanningTree() {
  delete spanning_tree_;
  spanning_tree_ = new SpanningTree(points_, octree_, hope_dist_ * 2.0, hope_dist_, 4);
}

void Model::UpdateDataStructure(bool update_3d_radius) {
  StopWatch stop_watch;
  // First, update spanning tree for matching algorothm.
  UpdateSpanningTree();
  // Update 3d radius.
  if (update_3d_radius) {
    Update3DRadius();
  }
  // LOG(INFO) << "Update 3d radius time: " << stop_watch.TimeDuration();
  // Update graph.
  delete curve_network_;
  curve_network_ = new IronTown(points_,
                                octree_,
                                hope_dist_ * topology_searching_radius_, hope_dist_, &points_radius_3d_);
  LOG(INFO) << "Update data structure time: " << stop_watch.TimeDuration();
}

void Model::UpdateOctree() {
  delete octree_;
  octree_ = new OctreeNew(points_);
}
