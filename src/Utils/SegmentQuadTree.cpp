//
// Created by aska on 2019/12/13.
//

#include "SegmentQuadTree.h"

const double kMaxValidIntersectionCosAbs = 0.9;

using Node = SegmentQuadTreeNode;

SegmentQuadTree::SegmentQuadTree(const std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>& segments,
                                 const std::vector<Eigen::Vector2d>& tangs):
    segments_(segments), tangs_(tangs) {
  CHECK(!segments.empty());
  CHECK_EQ(segments.size(), tangs.size());
  Eigen::Vector2d mi;
  Eigen::Vector2d ma;
  std::vector<Eigen::Vector2d> points;
  points.reserve(segments.size() * 2);
  for (const auto& pr : segments) {
    points.push_back(pr.first);
    points.push_back(pr.second);
  }
  std::tie(mi, ma) = Utils::GetAABB(points);
  double r = std::max(ma(0) - mi(0), ma(1) - mi(0)) * 0.55;
  node_pool_.emplace_back(new Node((mi + ma) * 0.5, r));
  root_ = node_pool_.back().get();

  // Insert segments
  for (int i = 0; i < segments_.size(); i++) {
    InsertSegment(root_, i);
  }
}

void SegmentQuadTree::InsertSegment(SegmentQuadTreeNode *u, int seg_idx) {
  const auto& segment = segments_[seg_idx];
  const Eigen::Vector2d& o = u->o_;
  double r = u->r_;
  bool is_inside_sons = false;
  for (int idx = 0; idx < 4; idx++) {
    double hr = r * 0.5;
    Eigen::Vector2d new_o = o + Eigen::Vector2d((idx & 1) - 0.5, ((idx >> 1) & 1) - 0.5) * r;
    if (IsSegmentInsideBox(new_o, hr, segment)) {
      is_inside_sons = true;
      if (u->sons_[idx] == nullptr) {
        Node* new_node = new Node(new_o, hr);
        node_pool_.emplace_back(new_node);
        u->sons_[idx] = new_node;
      }
      InsertSegment(u->sons_[idx], seg_idx);
    }
  }
  if (!is_inside_sons) {
    u->indexes_.emplace_back(seg_idx);
  }
}

void SegmentQuadTree::FindIntersections(const Eigen::Vector2d& o,
                                        const Eigen::Vector2d& v,
                                        double t_min,
                                        double t_max,
                                        std::vector<std::pair<double, Eigen::Vector2d>>* intersections) {
  intersections->clear();
  std::vector<std::pair<double, Eigen::Vector2d>> dup_intersections;
  FindIntersections(root_, o, v, t_min, t_max, &dup_intersections);
  std::sort(dup_intersections.begin(),
            dup_intersections.end(),
            [](const std::pair<double, Eigen::Vector2d>& a, const std::pair<double, Eigen::Vector2d>& b) {
              return a.first < b.first;});
  for (const auto& intersection : dup_intersections) {
    if (intersections->empty() || std::abs(intersections->back().first - intersection.first) > 1e-8) {
      intersections->emplace_back(intersection);
    }
  }
}

void SegmentQuadTree::FindIntersections(SegmentQuadTreeNode* u,
                                        const Eigen::Vector2d& o,
                                        const Eigen::Vector2d& v,
                                        double t_min,
                                        double t_max,
                                        std::vector<std::pair<double, Eigen::Vector2d>>* intersections) {
  // LOG(INFO) << "o: " << u->o_.transpose() << " r: " << u->r_ << " ray_o: " << o.transpose()
  //           << " ray_v: " << v.transpose() << " t_min: " << t_min << " t_max: " << t_max;
  if (!IsSegmentIntersectBox(u->o_, u->r_, o, v, t_min, t_max)) {
    return;
  }
  for (int seg_idx : u->indexes_) {
    // LOG(INFO) << "try: " << segments_[seg_idx].first.transpose() << " " << segments_[seg_idx].second.transpose();
    double t = FindIntersectionsTwoSegment(o, v, t_min, t_max, segments_[seg_idx]);
    if (t > t_min - 1e-9 && t < t_max + 1e-9) {
      intersections->emplace_back(t, tangs_[seg_idx]);
    }
  }
  for (int idx = 0; idx < 4; idx++) {
    if (u->sons_[idx] != nullptr) {
      FindIntersections(u->sons_[idx], o, v, t_min, t_max, intersections);
    }
  }
}

double SegmentQuadTree::FindIntersectionsTwoSegment(const Eigen::Vector2d& o,
                                                    const Eigen::Vector2d& v,
                                                    double t_min,
                                                    double t_max,
                                                    const std::pair<Eigen::Vector2d, Eigen::Vector2d>& segment) {
  Eigen::Vector2d ano_o = segment.first;
  Eigen::Vector2d ano_v = (segment.second - segment.first).normalized();
  auto cross = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) -> double {
    return a(0) * b(1) - a(1) * b(0);
  };
  if (std::abs(cross(v, ano_v)) < 1e-9) {
    return -1e9;
  }
  if (std::abs(v.dot(ano_v)) > kMaxValidIntersectionCosAbs) {
    return -1e9;
  }
  double t = cross(ano_v, (o - ano_o)) / cross(v, ano_v);
  if (t > t_max + 1e-9 || t < t_min - 1e-9) {
    return -1e9;
  }
  Eigen::Vector2d intersected_pt = o + v * t;
  if ((intersected_pt - segment.first).norm() < 1e-5) {
    return t;
  }
  if ((intersected_pt - segment.second).norm() < 1e-5) {
    return t;
  }
  if ((intersected_pt - segment.first).normalized().dot((intersected_pt - segment.second).normalized()) < -1e-9) {
    return t;
  }
  return -1e9;
}

bool SegmentQuadTree::IsSegmentInsideBox(const Eigen::Vector2d& o,
                                         double r,
                                         const std::pair<Eigen::Vector2d, Eigen::Vector2d>& segment) {
  const auto& pt_a = segment.first;
  const auto& pt_b = segment.second;
  if (std::max(std::abs(pt_a(0) - o(0)), std::abs(pt_a(1) - o(1))) > r - 1e-7) {
    return false;
  }
  if (std::max(std::abs(pt_b(0) - o(0)), std::abs(pt_b(1) - o(1))) > r - 1e-7) {
    return false;
  }
  return true;
}

bool SegmentQuadTree::IsSegmentIntersectBox(const Eigen::Vector2d& o,
                                            double r,
                                            const std::pair<Eigen::Vector2d, Eigen::Vector2d>& segment) {
  double t_max = (segment.second - segment.first).norm();
  return IsSegmentIntersectBox(o, r, segment.first, (segment.second - segment.first) / t_max, 0.0, t_max);
}

bool SegmentQuadTree::IsSegmentIntersectBox(const Eigen::Vector2d& o,
                                            double r,
                                            const Eigen::Vector2d& seg_o,
                                            const Eigen::Vector2d& seg_v,
                                            double t_min,
                                            double t_max) {
  for (int k = 0; k < 2; k++) {
    if (std::abs(seg_v(k)) < 1e-9) {
      if (seg_o(k) < o(k) - r - 1e-9 || seg_o(k) > o(k) + r + 1e-9) {
        return false;
      }
      continue;
    }
    double l_bound = (o(k) - r - seg_o(k)) / seg_v(k);
    double r_bound = (o(k) + r - seg_o(k)) / seg_v(k);
    if (l_bound > r_bound) {
      std::swap(l_bound, r_bound);
    }
    t_min = std::max(t_min, l_bound);
    t_max = std::min(t_max, r_bound);
  }
  return t_min - 1e-9 < t_max;
}
