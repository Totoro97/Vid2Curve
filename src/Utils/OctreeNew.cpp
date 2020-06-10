//
// Created by aska on 2020/1/1.
//

#include "OctreeNew.h"

OctreeNew::OctreeNew(const std::vector<Eigen::Vector3d>& points) : points_(points) {
  CHECK(!points_.empty());
  n_points_ = points_.size();
  Eigen::Vector3d ma(-1e10, -1e10, -1e10);
  Eigen::Vector3d mi(1e10, 1e10, 1e10);
  for (const auto& pt : points) {
    for (int t = 0; t < 3; t++) {
      ma(t) = std::max(ma(t), pt(t));
      mi(t) = std::min(mi(t), pt(t));
    }
  }
  corner_ = mi - Eigen::Vector3d(1e-8, 1e-8, 1e-8);
  double r = std::max({ ma(0) - mi(0), ma(1) - mi(1), ma(2) - mi(2) }) * (0.50 + 1e-4);
  p_.resize(n_points_);
  std::iota(p_.begin(), p_.end(), 0);

  root_ = BuildTreeNode(corner_, r, 0, n_points_);
}

OctreeNew::~OctreeNew() {
  delete root_;
}

OctreeNode* OctreeNew::BuildTreeNode(const Eigen::Vector3d& corner, double r, int l_bound, int r_bound) {
  if (l_bound >= r_bound) {
    return nullptr;
  }

  // LOG(INFO) << l_bound << " " << r_bound << " " << " " << corner(0) << " " << r;

  auto u = new OctreeNode();
  u->r_ = r;
  u->o_ = corner + Eigen::Vector3d(r, r, r);

  if (l_bound + 1 == r_bound || r < 1e-9) {
    u->u_idx_ = p_[l_bound];
    return u;
  }

  for (int i = l_bound; i < r_bound; i++) {
    int v = p_[i];
    for (int k = 0; k < 3; k++) {
      CHECK(corner(k) < points_[v](k) + 1e-7);
      CHECK(corner(k) + r * 2.0 > points_[v](k) - 1e-7);
    }
  }

  std::vector<int> sub_sizes(8, 0);
  for (int t = l_bound; t < r_bound; t++) {
    const auto& pt = points_[p_[t]];
    int sub_idx = 0;
    for (int k = 0; k < 3; k++) {
      sub_idx |= (int(pt(k) > u->o_(k)) << k);
    }
    sub_sizes[sub_idx]++;
  }
  // LOG(INFO) << "sub_sizes: ";
  // for (int i = 0; i < 8; i++) {
  //   LOG(INFO) << sub_sizes[i];
  // }
  std::vector<int> sub_sizes_cp = sub_sizes;
  for (int i = 1; i < 8; i++) {
    sub_sizes[i] += sub_sizes[i - 1];
  }
  CHECK_EQ(sub_sizes.back(), r_bound - l_bound);
  std::vector<int> p_cp;
  for (int t = l_bound; t < r_bound; t++) {
    p_cp.emplace_back(p_[t]);
  }
  // LOG(INFO) << "bf: " << p_[0] << " " << p_[1] << " " << p_[2] << " " << p_[3] << " " << p_[4];
  // LOG(INFO) << "sub_sizes: " << sub_sizes[0] << " " << sub_sizes[7];
  for (int t = l_bound; t < r_bound; t++) {
    const auto& pt = points_[p_cp[t - l_bound]];
    int sub_idx = 0;
    for (int k = 0; k < 3; k++) {
      sub_idx |= (int(pt(k) > u->o_(k)) << k);
    }
    sub_sizes[sub_idx]--;
    int new_idx = sub_sizes[sub_idx] + l_bound;
    // LOG(INFO) << "sub_idx: " << sub_idx << " new_idx: " << new_idx << " p_cp[t]: " << p_cp[t];
    p_[new_idx] = p_cp[t - l_bound];
  }
  // LOG(INFO) << "af: " << p_[0] << " " << p_[1] << " " << p_[2] << " " << p_[3] << " " << p_[4];

  int acc = 0;
  for (int t = 0; t < 8; t++) {
    u->sons_[t] = BuildTreeNode(corner + Eigen::Vector3d(t & 1, t >> 1 & 1, t >> 2 & 1) * r,
                                r * 0.5,
                                l_bound + acc,
                                l_bound + acc + sub_sizes_cp[t]);
    acc += sub_sizes_cp[t];
  }
  // CHECK_EQ(acc, r_bound - l_bound);
  return u;
}

int OctreeNew::NearestIdx(const Eigen::Vector3d& point) {
  /*
  double dis = 1e10;
  int ret = -1;
  for (int i = 0; i < n_points_; i++) {
    if (dis > (point - points_[i]).norm()) {
      dis = (point - points_[i]).norm();
      ret = i;
    }
  }
  return ret;
  */
  // CHECK_LT((points_[idx] - point).norm(), dis + 1e-6) << idx << " " << ret << " "
  //    << points_[idx].transpose() << " " << points_[ret].transpose() << " " << point.transpose();
  // CHECK_EQ(ret, NearestIdx(root_, point, 1e10));
  int idx = NearestIdx(root_, point, 1e10);
  return idx;
}

int OctreeNew::NearestIdx(OctreeNode* u, const Eigen::Vector3d& point, double dis) {
  if (u == nullptr) {
    return -1;
  }
  double bound =
      std::max({ std::abs(point(0) - u->o_(0)), std::abs(point(1) - u->o_(1)), std::abs(point(2) - u->o_(2))});
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
  int c = (point(2) > u->o_(2) - 1e-9);
  int idx = a + b * 2 + c * 4;
  int ret = -1;
  if (u->sons_[idx] != nullptr) {
    int tmp_ret = NearestIdx(u->sons_[idx], point, dis);
    double tmp_dis = 1e10;
    if (tmp_ret >= 0) {
      tmp_dis = (points_[tmp_ret] - point).norm();
    }
    if (dis > tmp_dis) {
      dis = tmp_dis;
      ret = tmp_ret;
    }
  }
  for (int i = 0; i < 8; i++) {
    if (i != idx && u->sons_[i] != nullptr) {
      int tmp_ret = NearestIdx(u->sons_[i], point, dis);
      double tmp_dis = 1e10;
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

void OctreeNew::SearchingR(const Eigen::Vector3d& o, double r, std::vector<int>* neighbors) {
  // for (int i = 0; i < n_points_; i++) {
    // if ((points_[i] - o).norm() < r) {
    //   neighbors->emplace_back(i);
    // }
  // }
  SearchingR(root_, o, r, neighbors);
}

void OctreeNew::SearchingR(OctreeNode* u, const Eigen::Vector3d& o, double r, std::vector<int>* neighbors) {
  Eigen::Vector3d bias = o - u->o_;
  double single_bias = 0.0;
  for (int t = 0; t < 3; t++) {
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
  for (int i = 0; i < 8; i++) {
    if (u->sons_[i] != nullptr) {
      SearchingR(u->sons_[i], o, r, neighbors);
    }
  }
}
