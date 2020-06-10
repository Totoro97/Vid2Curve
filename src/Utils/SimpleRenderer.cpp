//
// Created by aska on 2019/4/12.
//

#include "SimpleRenderer.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

SimpleRenderer::SimpleRenderer(int height, int width, double focal_length):
    height_(height), width_(width), focal_length_(focal_length) {
}

void SimpleRenderer::CacheData() {
  n_points_ = points_loader_->size();
  n_traces_ = traces_loader_->size();
  points_.clear();
  for (int i = 0; i < n_points_; i++) {
    points_.push_back((*points_loader_)[i]);
  }
  traces_.clear();
  for (int i = 0; i < n_traces_; i++) {
    traces_.push_back((*traces_loader_)[i]);
  }
}

void SimpleRenderer::CalcRays(int idx, std::vector<Vector3d> &rays, double add_error) {
  rays.clear();
  Vector3d pos = traces_[idx].block(0, 0, 3, 1);
  Vector3d to =  traces_[idx].block(0, 1, 3, 1);
  Vector3d up =  traces_[idx].block(0, 2, 3, 1);
  Vector3d axis_y = -up / up.norm();
  Vector3d axis_z = to / to.norm();
  Vector3d axis_x = axis_y.cross(axis_z);
  auto Rand = [](double L, double R) {
    return (double) std::rand() / (double) RAND_MAX * (R - L) + L;
  };
  for (const auto &point : points_) {
    Vector3d bias = point - pos;
    Vector3d pt(bias.dot(axis_x), bias.dot(axis_y), bias.dot(axis_z));
    pt /= pt(2);
    pt += Eigen::Vector3d(Rand(-add_error, add_error), Rand(-add_error, add_error), 0.0);
    rays.emplace_back(pt);
  }
}

void SimpleRenderer::OutputAllImages(std::string dir_path, double add_noise) {
  if (dir_path.back() != '/') {
    dir_path += '/';
  }
  for (int idx = 0; idx < n_traces_; idx++) {
    cv::Mat img(height_, width_, CV_8UC1);
    std::memset(img.data, 0, height_ * width_);
    Vector3d pos = traces_[idx].block(0, 0, 3, 1);
    Vector3d to =  traces_[idx].block(0, 1, 3, 1);
    Vector3d up =  traces_[idx].block(0, 2, 3, 1);
    Vector3d axis_y = -up / up.norm();
    Vector3d axis_z = to / to.norm();
    Vector3d axis_x = axis_y.cross(axis_z);
    for (const auto &point : points_) {
      Vector3d bias = point - pos;
      Vector3d pt(bias.dot(axis_x), bias.dot(axis_y), bias.dot(axis_z));
      pt /= pt(2);
      int a = std::round(pt(1) * focal_length_ + height_ * 0.5);
      int b = std::round(pt(0) * focal_length_ + width_ * 0.5);
      if (a >= 0 && a < height_ && b >= 0 && b < width_) {
        img.data[a * width_ + b] = 255;
      }
    }
    cv::imwrite(dir_path + std::to_string(idx) + ".png", img);
  }
}

void SimpleRenderer::OutputAllMats(std::string dir_path) {
  if (dir_path.back() != '/') {
    dir_path += '/';
  }
  for (int idx = 0; idx < n_traces_; idx++) {
    cv::Mat img(height_, width_, CV_8UC1);
    std::memset(img.data, 0, height_ * width_);
    Vector3d pos = traces_[idx].block(0, 0, 3, 1);
    Vector3d to =  traces_[idx].block(0, 1, 3, 1);
    Vector3d up =  traces_[idx].block(0, 2, 3, 1);
    Vector3d axis_y = -up / up.norm();
    Vector3d axis_z = to / to.norm();
    Vector3d axis_x = axis_y.cross(axis_z);
    // pt = R(point - pos)
    // pt = R(point) - R(pos);
    Eigen::Matrix3d R;
    R.block(0, 0, 1, 3) = axis_x.transpose();
    R.block(1, 0, 1, 3) = axis_y.transpose();
    R.block(2, 0, 1, 3) = axis_z.transpose();
    Eigen::Vector3d T;
    T = -R * pos;

    std::ofstream f(dir_path + std::to_string(idx) + ".txt");
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        f << R(i, j) << " ";
      }
      f << std::endl;
    }
    for (int i = 0; i < 3; i++) {
      f << T(i) << " ";
    }
    f << std::endl;
    f << width_ << " " << height_ << std::endl << focal_length_ << std::endl;
    f.close();
  }
}
