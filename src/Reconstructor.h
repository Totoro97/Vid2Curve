// Created by aska on 2019/4/14.
//
#pragma once

#include "Core/Model.h"
#include "Core/CurveExtractor.h"
#include "Core/CurveMatcher.h"
#include "Core/View.h"
#include "Utils/Common.h"
#include "Utils/GlobalDataPool.h"
#include "Utils/Streamer.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Reconstructor {
public:
  enum State { RUNNING, EXIT_OK };

  Reconstructor(std::string ini_file_name, GlobalDataPool* global_data_pool);
  ~Reconstructor();
  void Run();
  void Initialize();
  void PushNewExtractors(int n_new_extractors = 1);
  void PressureTest(Model* final_model);

  // Configures
  PropertyTree ptree_;
  std::string img_path_;
  std::string data_path_;
  int n_views_, initial_idx_;
  int height_, width_;
  bool use_gpu_;
  double focal_length_;
  double init_depth_;

  // Configures: For Deforming
  double rigid_weight_;

  std::string reconstruction_method_;

  // Data
  std::vector<std::unique_ptr<CurveExtractor> > extractors_;

  std::unique_ptr<StreamerBase> streamer_;
  GlobalDataPool* global_data_pool_;
  State state_ = State::RUNNING;
};