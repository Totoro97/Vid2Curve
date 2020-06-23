//
// Created by aska on 2019/10/19.
//

#include "QuadTree.h"

#include <numeric>

QuadTree::QuadTree(const std::vector<Eigen::Vector2d>& points) : points_(points) {
  CHECK(!points_.empty());
  n_points_ = points_.size();
  Eigen::Vector2d ma(-1e9, -1e9);
  Eigen::Vector2d mi(1e9, 1e9);
  for (const auto& pt : points) {
    for (int t = 0; t < 2; t++) {
      ma(t) = std::max(ma(t), pt(t));
      mi(t) = std::min(mi(t), pt(t));
    }
  }
  corner_ = mi - Eigen::Vector2d(1e-8, 1e-8);
  double r = std::max(ma(0) - mi(0), ma(1) - mi(1)) * (0.50 + 1e-4);
  p_.resize(n_points_);
  std::iota(p_.begin(), p_.end(), 0);

  root_ = BuildTreeNode(corner_, r, 0, n_points_);
}

QuadTree::~QuadTree() {
  delete root_;
}

QuadTreeNode* QuadTree::BuildTreeNode(const Eigen::Vector2d& corner, double r, int l_bound, int r_bound) {
  if (l_bound >= r_bound) {
    return nullptr;
  }
  auto u = new QuadTreeNode();
  u->r_ = r;
  u->o_ = corner + Eigen::Vector2d(r, r);

  if (l_bound + 1 == r_bound || r < 1e-9) {
    u->u_idx_ = p_[l_bound];
    return u;
  }

  for (int i = l_bound; i < r_bound; i++) {
    int v = p_[i];
    CHECK(corner(0) < points_[v](0) + 1e-7);
    CHECK(corner(1) < points_[v](1) + 1e-7);
    CHECK(corner(0) + r * 2.0 > points_[v](0) - 1e-7);
    CHECK(corner(1) + r * 2.0 > points_[v](1) - 1e-7);
  }

  int mid = l_bound;
  {
    int tmp_l = l_bound;
    int tmp_r = r_bound;
    while (true) {
      while (tmp_l < tmp_r && points_[p_[tmp_l]](0) < u->o_(0)) {
        tmp_l++;
      }
      while (tmp_l < tmp_r && points_[p_[tmp_r - 1]](0) >= u->o_(0)) {
        tmp_r--;
      }
      if (tmp_l >= tmp_r) {
        break;
      }
      std::swap(p_[tmp_l], p_[tmp_r - 1]);
    }
    CHECK_EQ(tmp_l, tmp_r);
    mid = tmp_l;
  }

  int l_sec = l_bound;
  {
    int tmp_l = l_bound;
    int tmp_r = mid;
    while (true) {
      while (tmp_l < tmp_r && points_[p_[tmp_l]](1) < u->o_(1)) {
        tmp_l++;
      }
      while (tmp_l < tmp_r && points_[p_[tmp_r - 1]](1) >= u->o_(1)) {
        tmp_r--;
      }
      if (tmp_l >= tmp_r) {
        break;
      }
      std::swap(p_[tmp_l], p_[tmp_r - 1]);
    }
    CHECK_EQ(tmp_l, tmp_r);
    l_sec = tmp_l;
  }
  int r_sec = mid;
  {
    int tmp_l = mid;
    int tmp_r = r_bound;
    while (true) {
      while (tmp_l < tmp_r && points_[p_[tmp_l]](1) < u->o_(1)) {
        tmp_l++;
      }
      while (tmp_l < tmp_r && points_[p_[tmp_r - 1]](1) >= u->o_(1)) {
        tmp_r--;
      }
      if (tmp_l >= tmp_r) {
        break;
      }
      std::swap(p_[tmp_l], p_[tmp_r - 1]);
    }
    CHECK_EQ(tmp_l, tmp_r);
    r_sec = tmp_l;
  }
  u->sons_[0] = BuildTreeNode(corner, r * 0.5, l_bound, l_sec);
  u->sons_[1] = BuildTreeNode(corner + Eigen::Vector2d(0.0, r), r * 0.5, l_sec, mid);
  u->sons_[2] = BuildTreeNode(corner + Eigen::Vector2d(r, 0.0), r * 0.5, mid, r_sec);
  u->sons_[3] = BuildTreeNode(corner + Eigen::Vector2d(r, r), r * 0.5, r_sec, r_bound);
  return u;
}

Eigen::Vector2d QuadTree::NearestPoint(const Eigen::Vector2d& point) {
  return points_[NearestIdx(point)];
}

int QuadTree::NearestIdx(const Eigen::Vector2d& point) {
  int idx = NearestIdx(root_, point, 1e9);
  return idx;
}

int QuadTree::NearestIdx(QuadTreeNode* u, const Eigen::Vector2d& point, double dis) {
  if (u == nullptr) {
    return -1;
  }
  double bound = std::max(std::abs(point(0) - u->o_(0)), std::abs(point(1) - u->o_(1)));
  // std::cout << point.transpose() << std::endl;
  // std::cout << u->o_.transpose() << std::endl;
  // std::cout << bound << " " << dis << " " << u->r_ << std::endl;
  if (bound > dis + u->r_) {
    return -1;
  }
  if (u->u_idx_ != -1) {
    return u->u_idx_;
  }
  int a = (point(0) > u->o_(0) - 1e-9);
  int b = (point(1) > u->o_(1) - 1e-9);
  int idx = a * 2 + b;
  int ret = -1;
  if (u->sons_[idx] != nullptr) {
    int tmp_ret = NearestIdx(u->sons_[idx], point, dis);
    double tmp_dis = 1e9;
    if (tmp_ret >= 0) {
      tmp_dis = (points_[tmp_ret] - point).norm();
    }
    if (dis > tmp_dis) {
      dis = tmp_dis;
      ret = tmp_ret;
    }
  }
  for (int i = 0; i < 4; i++) {
    if (i != idx && u->sons_[i] != nullptr) {
      int tmp_ret = NearestIdx(u->sons_[i], point, dis);
      double tmp_dis = 1e9;
      if (tmp_ret >= 0) {
        tmp_dis = (points_[tmp_ret] - point).norm();
      }
      if (dis > tmp_dis) {
        dis = tmp_dis;
        ret = tmp_ret;
      }
    }
  }
  return ret;
}

void QuadTree::SearchingR(const Eigen::Vector2d& o, double r, std::vector<int>* neighbors) {
  for (int i = 0; i < n_points_; i++) {
    // if ((points_[i] - o).norm() < r) {
    //   neighbors->emplace_back(i);
    // }
  }
  SearchingR(root_, o, r, neighbors);
}

void QuadTree::SearchingR(QuadTreeNode* u, const Eigen::Vector2d& o, double r, std::vector<int>* neighbors) {
  Eigen::Vector2d bias = o - u->o_;
  double single_bias = 0.0;
  for (int t = 0; t < 2; t++) {
    single_bias = std::max(single_bias, std::abs(bias(t)));
  }
  if (single_bias > r + u->r_) {
    return;
  }
  if (u->u_idx_ != -1) {
    if ((points_[u->u_idx_] - o).norm() < r + 1e-9) {
      neighbors->emplace_back(u->u_idx_);
    }
    return;
  }
  for (int i = 0; i < 4; i++) {
    if (u->sons_[i] != nullptr) {
      SearchingR(u->sons_[i], o, r, neighbors);
    }
  }
}
