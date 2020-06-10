//
// Created by aska on 2019/9/26.
//

#include "GlobalDataPool.h"


GlobalDataPool::GlobalDataPool() {}

#ifdef USE_GUI

void GlobalDataPool::DrawModelPoints(pangolin::GlBuffer* gl_buffer) {
  model_points_mutex_.lock();
  glColor3f(1.0, 1.0, 1.0);
  std::vector<float> data_to_draw;
  float x_offset = 0.0;
  for (const auto& pr : mp_model_points_) {
    for (int i = 0; i < pr.second.size(); i += 3) {
      data_to_draw.emplace_back(pr.second[i] + x_offset);
      data_to_draw.emplace_back(pr.second[i + 1]);
      data_to_draw.emplace_back(pr.second[i + 2]);
    }
    x_offset += 1.5;
  }
  gl_buffer->Upload(data_to_draw.data(), data_to_draw.size() * sizeof(float));
  model_points_mutex_.unlock();
}

void GlobalDataPool::UpdateModelPoints(void* model_ptr, const std::vector<Eigen::Vector3d>& points) {
  model_points_mutex_.lock();
  std::vector<float>& model_points = mp_model_points_[model_ptr];
  model_points.resize(points.size() * 3);
  for (int i = 0; i < points.size(); i++) {
    for (int j = 0; j < 3; j++) {
      model_points[i * 3 + j] = points[i](j);
    }
  }
  model_points_mutex_.unlock();
}

int GlobalDataPool::n_points() {
  int ret = 0;
  for (const auto& pr : mp_model_points_) {
    ret += pr.second.size();
  }
  return ret;
}

void GlobalDataPool::DrawImage(pangolin::GlTexture* image_texture) {
  last_frame_mutex_.lock();
  image_texture->Upload(last_frame_.data(), GL_RGB,GL_UNSIGNED_BYTE);
  last_frame_mutex_.unlock();
}

void GlobalDataPool::UpdateImageData(const cv::Mat& image) {
  last_frame_mutex_.lock();
  last_frame_.resize(image.cols * image.rows * image.channels());
  std::memcpy(last_frame_.data(), image.data, last_frame_.size());
  last_frame_mutex_.unlock();
}


void GlobalDataPool::UpdateCameras(void* model_ptr,
                                   const std::vector<Eigen::Matrix3d>& K_invs,
                                   const std::vector<Eigen::Matrix4d>& Wfs,
                                   const std::vector<std::pair<int, int>>& width_and_heights) {
  cameras_mutex_.lock();
  auto& camera_motion_info = mp_cameras_[model_ptr];
  camera_motion_info.K_invs_ = K_invs;
  camera_motion_info.Wfs_ = Wfs;
  camera_motion_info.width_and_heights_ = width_and_heights;
  cameras_mutex_.unlock();
}

void GlobalDataPool::DrawCameras() {
  cameras_mutex_.lock();
  glColor3f(1.0, 1.0, 0.0);
  double offset = 0.0;
  for (const auto& pr : mp_cameras_) {
    const auto& camera_motion_info = pr.second;
    const auto& K_invs_ = camera_motion_info.K_invs_;
    const auto& Wfs_ = camera_motion_info.Wfs_;
    const auto& width_and_heights_ = camera_motion_info.width_and_heights_;
    CHECK_EQ(K_invs_.size(), Wfs_.size());
    CHECK_EQ(K_invs_.size(), width_and_heights_.size());
    for (int i = 0; i < camera_motion_info.K_invs_.size(); i++) {
      Eigen::Matrix4d current_frame = Wfs_[i];
      current_frame(0, 3) += offset;
      pangolin::glDrawFrustum(K_invs_[i], width_and_heights_[i].first, width_and_heights_[i].second, current_frame, 0.05);
    }
    offset += 1.5;
  }
  cameras_mutex_.unlock();
}

void GlobalDataPool::DrawPlotting(pangolin::DataLog* log) {
  plotting_data_mutex_.lock();
  if (plotting_data_refreshed_) {
    if (plotting_data_.size() == 1) {
      log->Log(plotting_data_[0]);
    }
    else if (plotting_data_.size() == 2) {
      log->Log(plotting_data_[0], plotting_data_[1]);
    }
    else {
      LOG(FATAL) << "Two many items.";
    }
    plotting_data_refreshed_ = false;
  }
  plotting_data_mutex_.unlock();
}

void GlobalDataPool::UpdatePlottingData(const std::vector<double>& plotting_data){
  plotting_data_mutex_.lock();
  plotting_data_ = plotting_data;
  plotting_data_refreshed_ = true;
  plotting_data_mutex_.unlock();
}

void GlobalDataPool::DeleteModel(void *model_ptr) {
  cameras_mutex_.lock();
  mp_cameras_.erase(model_ptr);
  cameras_mutex_.unlock();

  model_points_mutex_.lock();
  mp_model_points_.erase(model_ptr);
  model_points_mutex_.unlock();
}

#endif
