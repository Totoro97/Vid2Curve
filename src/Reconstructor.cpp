//
// Created by aska on 2019/4/14.
//

#include "Core/Initializer.h"
#include "Reconstructor.h"
#include "Utils/Common.h"

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/progress.hpp>

Reconstructor::Reconstructor(std::string ini_file_name, GlobalDataPool* global_data_pool) {
  global_data_pool_ = global_data_pool;
  boost::property_tree::ini_parser::read_ini(ini_file_name, ptree_);
  data_path_ = ptree_.get<std::string>("Global.DataPath");
  if (data_path_.back() != '/') {
    data_path_ += "/";
  }
  PropertyTree local_ptree;
  boost::property_tree::ini_parser::read_ini(data_path_ + "local_config.ini", local_ptree);

  // Segmentation image path.
  img_path_ = data_path_ + local_ptree.get<std::string>("Local.ImgPath");
  if (img_path_.back() != '/') {
    img_path_ += "/";
  }
  ptree_.put("Global.ImgPath", img_path_);

  n_views_ = local_ptree.get<int>("Local.NumImages");
  ptree_.put("Global.NumImages", n_views_);
  initial_idx_ = 0;

  focal_length_ = local_ptree.get<double>("Local.FocalLength");
  height_ = local_ptree.get<int>("Local.Height");
  width_ = local_ptree.get<int>("Local.Width");

  std::vector<std::string> parameter_names = {
      "RadiusDilation",
      "FinalSmoothingWeight",
      "IteratingSmoothingWeight",
      "RadiusSmoothingWeight",
      "ShrinkErrorWeight"
  };
  for (const std::string& parameter_name : parameter_names) {
    double value = local_ptree.get<double>
        ("Local." + parameter_name, std::numeric_limits<double>::signaling_NaN());
    if (!std::isnan(value)) {
      ptree_.put("Global." + parameter_name, value);
    }
  }

  ptree_.put("Global.FocalLength", focal_length_);
  ptree_.put("Global.Height", height_);
  ptree_.put("Global.Width", width_);

  init_depth_ = 1.0;
  reconstruction_method_ = "POSE_CANDIDATES";
  use_gpu_ = ptree_.get<bool>("CurveExtractor.UseGPU");
  if (use_gpu_) {
    CHECK_EQ(std::system("nvidia-smi"), 0);
  }
  Initialize();
}

Reconstructor::~Reconstructor() {
}

void Reconstructor::Run() {
  boost::progress_display pd(n_views_);
  // Possible method 1.
  if (reconstruction_method_ == "POSE_CANDIDATES") {
    extractors_.clear();
    PushNewExtractors(1);
    // Greedy.
    std::vector<std::unique_ptr<ModelData>> model_states;
    int initial_frame_id = -1;
    for (int i = 1; i < 10 && i < n_views_; i++) {
      PushNewExtractors(1);
      auto pose_initializer =
          std::make_unique<Initializer>(std::vector<CurveExtractor*>{ extractors_[0].get(),
                                                                      extractors_[i].get() },
                                        focal_length_,
                                        (double) width_,
                                        (double) height_,
                                        0.01,
                                        0.0);
      pose_initializer->GetInitialModelData(&model_states);
      if (!model_states.empty()) {
        initial_frame_id = i;
        break;
      }
    }

    const int kMinTryFrameN = 10;
    const int kMaxTryFrameN = 40;
    const int kMinContinuousN = 10;
    CHECK_GT(n_views_, kMaxTryFrameN + initial_frame_id);
    PushNewExtractors(kMaxTryFrameN);

    double model_score = -1e9;
    std::vector<double> scores_for_debug;
    // model_states.resize(1);
    const double kEraseRatio = 1.3;

    std::set<std::pair<double, Model*>> current_models;
    for (const auto& model_state : model_states) {
      auto current_model =  new Model(model_state.get(), ptree_, global_data_pool_);
      current_model->Update();
      current_models.emplace(-current_model->Score(), current_model);
    }

    Model* model_candidate = nullptr;
    int next_frame_id = initial_frame_id + 1;
    
    for (int i = 0; i < initial_frame_id; i++) {
      ++pd;
    }
    
    for (int frame_id = initial_frame_id + 1; frame_id <= initial_frame_id + kMaxTryFrameN; frame_id++) {
      ++pd;
      std::set<std::pair<double, Model*>> new_models;
      // Actually, the score here is the negative distance.
      double max_score = -1e9;
      next_frame_id = frame_id + 1;
      for (const auto& pr : current_models) {
        auto current_model = pr.second;
        current_model->FeedExtractor(extractors_[frame_id].get());
        current_model->Update();
        double current_score = current_model->Score() * 0.3 + (-pr.first) * 0.7;
        bool has_similar_model = false;
        for (const auto& pr : new_models) {
          if (pr.second->IsCameraMotionSimilar(current_model)) {
            has_similar_model = true;
            break;
          }
        }
        if (has_similar_model) {
#ifdef USE_GUI
          global_data_pool_->DeleteModel((void*) current_model);
#endif
          delete current_model;
        }
        else {
          max_score = std::max(max_score, current_score);
          new_models.emplace(-current_score, current_model);
        }
      }
      double score_threshold = -1e8;
      if (frame_id > initial_frame_id + kMinTryFrameN && new_models.begin()->second->IsCameraMovementSufficient()) {
        score_threshold = max_score * kEraseRatio;
      }
      auto base_pr = std::make_pair(-score_threshold, nullptr);
      for (auto iter = new_models.lower_bound(base_pr); iter != new_models.end(); iter++) {
#ifdef USE_GUI
        global_data_pool_->DeleteModel((void*) iter->second);
#endif
        delete iter->second;
      }
      new_models.erase(new_models.lower_bound(base_pr), new_models.end());

      CHECK(!new_models.empty());
      if (new_models.size() <= 1) {
        model_candidate = new_models.begin()->second;
        model_candidate->model_state_ = ModelState::ITERATING;
        break;
      }
      current_models = new_models;
      LOG(INFO) << "threshold: " << score_threshold;
      for (const auto& pr : current_models) {
        LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!! " << pr.first;
      }
    }

    if (model_candidate == nullptr) {
      model_candidate = current_models.begin()->second;
      model_candidate->model_state_ = ModelState::ITERATING;
      for (const auto& pr : current_models) {
        if (pr.second != model_candidate) {
          delete pr.second;
        }
      }
    }

    for (int frame_id = next_frame_id; frame_id < n_views_; frame_id++) {
      ++pd;
      PushNewExtractors(frame_id + 1 - extractors_.size());
      model_candidate->FeedExtractor(extractors_[frame_id].get());
      model_candidate->Update();
    }
    model_candidate->FinalProcess();
    Utils::SavePointsAsPly("points.ply", model_candidate->points_);
    if (reconstruction_method_ == "PRESSURE_TEST") {
      PressureTest(model_candidate);
    }
    delete model_candidate;
    state_ = State::EXIT_OK;
  }
  else {
    LOG(FATAL) << "Single view initialization is deprecated.";
  }
}

void Reconstructor::Initialize() {
  LOG(INFO) << "Initialize: Begin";
  std::string streamer_type = ptree_.get<std::string>("Streamer.StreamerType");
  if (streamer_type == "IMAGE_STREAMER") {
    streamer_ = std::make_unique<ImageStreamer>(img_path_, n_views_);
  } else {
    LOG(FATAL) << "Unsupported streamer type.";
  }             
  LOG(INFO) << "Initialize: End";
}

void Reconstructor::PushNewExtractors(int n_new_extractors) {
  StopWatch stop_watch;
  for (int i = 0; i < n_new_extractors; i++) {
    extractors_.push_back(std::make_unique<CurveExtractor>(streamer_.get(), &ptree_));
    streamer_->SwitchToNextFrame();
  }
  LOG(INFO) << "Build new extractor time: " << stop_watch.TimeDuration();
}

void Reconstructor::PressureTest(Model *final_model) {
  std::string original_img_path = ptree_.get<std::string>("Global.OriginalImgPath");
  if (original_img_path.back() != '/') {
    original_img_path += "/";
  }
  int original_n_views = ptree_.get<int>("Global.OriginalNumViews");
  ModelData model_state;
  model_state.points = final_model->points_;
  model_state.tangs = final_model->tangs_;
  model_state.tang_scores = final_model->tang_scores_;
  std::vector<double> tracking_steps[4];

  Eigen::Vector3d center_pt(0.0, 0.0, 0.0);
  for (const auto& pt : final_model->points_) {
    center_pt += pt;
  }
  center_pt /= final_model->points_.size();
  for (int left_idx = 0; left_idx < original_n_views; left_idx += 10) {
    LOG(INFO) << "------ " << left_idx << " ------";
    model_state.camera_poses.clear();
    Eigen::Matrix3d left_R = final_model->views_[left_idx / 10]->R_;
    Eigen::Vector3d left_T = final_model->views_[left_idx / 10]->T_;

    model_state.camera_poses.emplace_back(left_R, left_T);
    model_state.curve_extractors.clear();
    auto left_single_image_streamer =
        std::make_unique<SingleImageStreamer>(original_img_path + std::to_string(left_idx) + ".png");
    auto left_extractor = std::make_unique<CurveExtractor>((StreamerBase *) left_single_image_streamer.get(), &ptree_);
    model_state.curve_extractors.emplace_back(left_extractor.get());

    for (auto track_type : std::vector <TrackType> { TrackType::MATCHING, TrackType::NAIVE }) {
      int step = 400;
      int last_good_idx = left_idx + 1;
      Eigen::Matrix3d last_good_R;
      Eigen::Vector3d last_good_T;
      Eigen::Vector3d left_pos = left_R.inverse() * -left_T;
      for (int right_idx = left_idx + 1; right_idx < left_idx + 500;) {
        auto model = std::make_unique<Model>(&model_state, ptree_, global_data_pool_);
        model->track_type_ = track_type;
        auto right_single_image_streamer =
            std::make_unique<SingleImageStreamer>(original_img_path + std::to_string(right_idx) + ".png");
        auto right_extractor = std::make_unique<CurveExtractor>(right_single_image_streamer.get(), &ptree_);
        model->FeedExtractor(right_extractor.get());
        bool is_tracking_good = model->IsTrackingGood();
        if (right_idx == left_idx + 1) {
          CHECK(is_tracking_good);
        }
        if (step < 1) {
          LOG(INFO) << (track_type == TrackType::NAIVE ? "NAIVE" : (track_type == TrackType::MATCHING? "MATCH" : "MATOF"))
              << " " << last_good_idx - left_idx;
          Eigen::Vector3d last_good_pos = last_good_R.inverse() * -last_good_T;
          double average_depth = 0.5 * ((last_good_pos - center_pt).norm() + (left_pos - center_pt).norm());
          tracking_steps[int(track_type)].emplace_back((last_good_pos - left_pos).norm() / average_depth);
          break;
        }
        if (!is_tracking_good) {
          step /= 2;
        } else {
          last_good_idx = right_idx;
          last_good_R = model->views_.back()->R_;
          last_good_T = model->views_.back()->T_;
        }
        right_idx = last_good_idx + step;
      }
      int u = track_type;
      LOG(INFO) << "average: " << (track_type == TrackType::NAIVE ? "NAIVE" : (track_type == TrackType::MATCHING? "MATCH" : "MATOF"))
                << " " << std::accumulate(tracking_steps[u].begin(), tracking_steps[u].end(), 0.0) / tracking_steps[u].size();
    }
  }
  for (int st : tracking_steps[0]) {
    std::cout << st << std::endl;
  }
}
