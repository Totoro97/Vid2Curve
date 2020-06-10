//
// Created by aska on 2019/10/22.
//
// This is new.

#include "Graph.h"
#include <numeric>

// ---------------------------------- base graph -----------------------------------------

void Graph::GetLinkedPaths(const std::vector<Eigen::Vector3d>& points,
                           std::vector<std::vector<int>>* paths,
                           double cos_thres,
                           int max_degree,
                           bool jump_junctions) {
  CHECK_EQ(n_points_, points.size());
  paths->clear();
  std::vector<std::vector<int>> original_paths;
  this->GetPaths(&original_paths);
  int n_paths = original_paths.size();
  std::vector<std::vector<std::pair<int, Eigen::Vector3d>>> outs(n_points_);
  for (int path_idx = 0; path_idx < original_paths.size(); path_idx++) {
    const auto& path = original_paths[path_idx];
    if (path.size() < 2) {
      continue;
    }
    if (path.front() == path.back()) {
      // paths->emplace_back(path);
      // The new circle path will be added in the later step. Therefore, we don't add new path here.
      continue;
    }
    int step = std::min(int(path.size() * 0.8), 10);
    // int header_idx = path.size() <= 2 ? 0 : 1;
    int header_idx = 0.0;
    outs[path.front()].emplace_back(path_idx, (points[path[step]] - points[path[header_idx]]).normalized());
    outs[path.back()].emplace_back(path_idx, (points[path[path.size() - step - 1]] - points[path[path.size() - header_idx - 1]]).normalized());
  }

  std::vector<std::vector<int>> path_outs(n_paths);
  std::vector<int> all_linked(n_points_, 0);
  for (int u = 0; u < n_points_; u++) {
    if (outs[u].size() < 2) {
      continue;
    }
    CHECK_LE(outs[u].size(), 30);
    if (outs[u].size() > max_degree) {
      continue;
    }
    int n_link = outs[u].size() / 2;
    std::vector<std::pair<int, double>> matching_pairs;
    double max_cos_val = -1e9;
    double min_cos_val = 1e9;
    for (int i = 0; i < outs[u].size(); i++) {
      for (int j = i + 1; j < outs[u].size(); j++) {
        double cos_val = outs[u][i].second.dot(outs[u][j].second);
        max_cos_val = std::max(max_cos_val, cos_val);
        min_cos_val = std::min(min_cos_val, cos_val);
        if (cos_val < cos_thres) {
          matching_pairs.emplace_back(((1 << i) | (1 << j)), cos_val);
        }
      }
    }

    max_cos_val += 1.0;
    min_cos_val += 1.0;
    if ((max_cos_val - min_cos_val) < 0.3 * max_cos_val) { // Too similar.
      continue;
    }
    std::function<void(int, int, double)> DFS;
    std::vector<int> current_pairs;
    std::vector<int> pairs_solution;
    double overall_cost = 1e9;
    DFS = [n_link, &matching_pairs, &overall_cost, &current_pairs, &pairs_solution, &DFS]
        (int current_n, int current_state, double current_cost) -> void {
      bool can_add = false;
      for (const auto& pr : matching_pairs) {
        int st = pr.first;
        double single_cost = pr.second;
        if ((current_state & st) == 0) {
          current_pairs.emplace_back(st);
          DFS(current_n + 1, current_state | st, std::max(current_cost, single_cost));
          current_pairs.pop_back();
          can_add = true;
        }
      }
      if (!can_add) {
        // current_cost -= 1e3 * current_n;
        if (current_n > 0 && current_cost < overall_cost) {
          overall_cost = current_cost;
          pairs_solution = current_pairs;
        }
        return;
      }
    };
    DFS(0, 0, -1.0);
    // LOG(INFO) << "overall cost: " << overall_cost;
    all_linked[u] = (pairs_solution.size() * 2 == outs[u].size());
    for (const auto& st : pairs_solution) {
      int a = -1;
      int b = -1;
      for (int i = 0; i < outs[u].size(); i++) {
        if ((st >> i & 1) != 0) {
          if (a == -1) {
            a = i;
          } else {
            CHECK_EQ(b, -1);
            b = i;
          }
        }
      }
      CHECK(a != -1) << st;
      CHECK(b != -1) << st;
      CHECK(a != b);
      path_outs[outs[u][a].first].emplace_back(outs[u][b].first);
      path_outs[outs[u][b].first].emplace_back(outs[u][a].first);
    }
  }

  std::vector<int> vis(n_paths, -1);
  std::vector<int> permu(n_paths);
  std::iota(permu.begin(), permu.end(), 0);
  std::sort(permu.begin(),
            permu.end(),
            [&path_outs](int a, int b) { return path_outs[a].size() < path_outs[b].size(); });
  for (int path_idx_p = 0; path_idx_p < n_paths; path_idx_p++) {
    int path_idx = permu[path_idx_p];
    CHECK_LE(path_outs[path_idx].size(), 2);
    if (vis[path_idx] != -1) {
      continue;
    }
    paths->emplace_back();
    auto& new_path = paths->back();
    if (path_outs[path_idx].empty()) {
      new_path.insert(new_path.end(), original_paths[path_idx].begin(), original_paths[path_idx].end());
      continue;
    }
    auto& current_path = original_paths[path_idx];
    auto& next_path = original_paths[path_outs[path_idx][0]];
    if (current_path.front() == next_path.front() || current_path.front() == next_path.back()) {
      std::reverse(current_path.begin(), current_path.end());
    }
    CHECK(current_path.back() == next_path.front() || current_path.back() == next_path.back());
    new_path.insert(new_path.end(), current_path.begin(), current_path.end());
    vis[path_idx] = 1;
    // LOG(INFO) << "+++" << path_outs[path_idx].size();
    int past_u = path_idx;
    for (int u = path_outs[path_idx][0]; u != -1 && vis[u] == -1; ) {
      // LOG(INFO) << "..." << path_outs[u].size();
      CHECK(vis[u] == -1);
      CHECK_LE(path_outs[u].size(), 2);
      CHECK_GE(original_paths[u].size(), 2);
      if (!(original_paths[u].front() == new_path.back() || original_paths[u].back() == new_path.back())) {
        break;
      }
      vis[u] = 1;
      CHECK(original_paths[u].front() == new_path.back() || original_paths[u].back() == new_path.back())
              << original_paths[u].front() << " " << original_paths[u].back() << " "
              << new_path.back() << " " << new_path.size() << " " << original_paths[u].size() << " "
              << original_paths[past_u].front() << " " << original_paths[past_u].back();
      if (original_paths[u].back() != new_path.back()) {
        int junction_u = new_path.back();
        if (jump_junctions || all_linked[junction_u]) {
          new_path.pop_back();
        }
        new_path.insert(new_path.end(), std::next(original_paths[u].begin()), original_paths[u].end());
      } else {
        int junction_u = new_path.back();
        if (jump_junctions || all_linked[junction_u]) {
          new_path.pop_back();
        }
        new_path.insert(new_path.end(), std::next(original_paths[u].rbegin()), original_paths[u].rend());
      }
      int v = -1;
      for (int v_hope : path_outs[u]) {
        if (vis[v_hope] != -1) {
          continue;
        }
        CHECK(v == -1);
        v = v_hope;
      }
      past_u = u;
      u = v;
    }
    // handle circular path.
    if (new_path.size() > 20 && new_path.front() == new_path.back()) {
      std::vector<int> tmp_path;
      for (auto iter = std::next(new_path.begin()); std::next(iter) != new_path.end(); iter++) {
        tmp_path.emplace_back(*iter);
      }
      // LOG(FATAL) << "hello here.";
      new_path = tmp_path;
    }
    if (new_path.size() > 20 && new_path.front() != new_path.back()) {
      Eigen::Vector3d head_pt = points[new_path.front()];
      Eigen::Vector3d head_dir = (points[new_path[0]] - points[new_path[2]]).normalized();
      Eigen::Vector3d tail_pt = points[new_path.back()];
      Eigen::Vector3d tail_dir =
          (points[new_path.back()] - points[new_path[new_path.size() - 3]]).normalized();
      double hope_dist = ((points[new_path[0]] - points[new_path[2]]).norm() +
                          (points[new_path.back()] - points[new_path[new_path.size() - 3]]).norm()) / 4.0;
      if (((head_pt - tail_pt).norm() < hope_dist * 5.0) && (head_dir.dot(tail_dir) < -0.5)) {
        // LOG(FATAL) << "here";
        new_path.emplace_back(new_path.front());
      }
    }
  }
  // LOG(INFO) << "n_path_before: " << n_paths;
  // LOG(INFO) << "n_path_after: " << paths->size();
}

void Graph::GetLinkedPathsWithSameDirection(const std::vector<Eigen::Vector3d>& points,
                                            std::vector<std::vector<int>>* paths,
                                            double cos_thres,
                                            int max_degree,
                                            bool jump_junctions) {
  std::vector<std::vector<int>> original_paths;
  GetLinkedPaths(points, &original_paths, cos_thres, max_degree, jump_junctions);
  paths->clear();
  for (const auto& path : original_paths) {
    const int kMinPathLegnth = 10;
    if (path.size() <= kMinPathLegnth * 2) {
      paths->emplace_back(path);
      continue;
    }
    int l = 0;
    int r = 1;
    int n = path.size();
    while (r < n) {
      if (r + kMinPathLegnth > n) {
        r = n;
        paths->emplace_back(path.data() + l, path.data() + r);
      }
      else if (r - l >= kMinPathLegnth && ((points[path[l + 1]] - points[path[l]]).normalized().
          dot(points[path[r + 1]] - points[path[r]])) < 0.5) {
        paths->emplace_back(path.data() + l, path.data() + r);
        l = r;
      }
      else {
        r++;
      }
    }
  }
}

// --------------------------------- spanning tree ---------------------------------------

SpanningTree::SpanningTree(const std::vector<Eigen::Vector3d>& points,
                           OctreeNew* octree,
                           double r,
                           double hope_dist,
                           int max_degree,
                           const std::vector<double>* const points_radius) :
    points_(&points), r_(r), hope_dist_(hope_dist), max_degree_(max_degree) {
  CHECK(points_radius == nullptr); // Do not use points radius when constructing spanning tree.
  n_points_ = points_->size();
  edges_.clear();
  edges_.resize(n_points_);
  std::vector<std::tuple<double, int, int>> initial_edges;
  std::vector<int> fa(n_points_);
  for (int i = 0; i < n_points_; i++) {
    fa[i] = i;
  }
  std::function<int(int)> FindRoot;
  FindRoot = [&fa, &FindRoot](int a) -> int {
    return fa[a] == a ? a : (fa[a] = FindRoot(fa[a]));
  };
  for (int u = 0; u < n_points_; u++) {
    const auto& p = (*points_)[u];
    std::vector<int> neighbors;
    octree->SearchingR(p, r, &neighbors);
    if (neighbors.size() <= 1) {
      continue;
    }
    for (int v : neighbors) {
      if (v <= u) {
        continue;
      }
      const auto &q = (*points_)[v];
      Eigen::Vector3d bias = q - p;
      if (bias.norm() > r) {
        continue;
      }
      double dis_sqr = bias.squaredNorm();
      double weight = std::exp(-dis_sqr / (r * r) * 0.5);
      if (weight < 1e-9) {
        continue;
      }
      initial_edges.emplace_back(-weight, u, v);
    }
  }

  std::sort(initial_edges.begin(), initial_edges.end());
  for (const auto& edge : initial_edges) {
    int a = std::get<1>(edge);
    int b = std::get<2>(edge);
    double weight = -std::get<0>(edge);
    if (FindRoot(a) == FindRoot(b) || edges_[a].size() >= max_degree || edges_[b].size() >= max_degree) {
      continue;
    }
    fa[fa[a]] = fa[b];
    edges_[a].emplace_back(b, weight);
    edges_[b].emplace_back(a, weight);
  }
}

void SpanningTree::GetPaths(std::vector<std::vector<int>>* paths) {
  // BFS
  std::vector<int> que;
  std::vector<int> f(n_points_, -1);
  paths->clear();
  int L = 0;
  for (int t = 0; t < n_points_; t++) {
    if (f[t] != -1 || edges_[t].size() > 1) {
      continue;
    }
    que.push_back(t);
    for (; L < que.size(); L++) {
      int u = que[L];
      for (const auto& pr : edges_[u]) {
        int v = pr.first;
        if (f[u] == v) {
          continue;
        }
        f[v] = u;
        que.push_back(v);
      }
    }
  }
  std::vector<int> vis(n_points_, 0);
  for (auto iter = que.rbegin(); iter != que.rend(); iter++) {
    int u = *iter;
    if (vis[u]) {
      continue;
    }
    vis[u] = 1;
    paths->emplace_back();
    auto& new_path = paths->back();
    new_path.push_back(u);
    int v = f[u];
    for (; v >= 0 && !vis[v] && edges_[v].size() <= 2; v = f[v]) {
      vis[v] = 1;
      new_path.push_back(v);
    }
    if (v >= 0) {
      new_path.push_back(v);
    }
  }
}

// -------------------------------------- IronTown --------------------------------------------------------

IronTown::IronTown(const std::vector<Eigen::Vector3d>& points,
                   OctreeNew* octree,
                   double r,
                   double hope_dist,
                   const std::vector<double>* const points_radius) : points_(&points), r_(r), hope_dist_(hope_dist) {
  n_points_ = points_->size();
  if (points_radius != nullptr) {
    CHECK_EQ(points_radius->size(), n_points_);
  }
  edges_.clear();
  edges_.resize(n_points_);
  std::vector<std::tuple<double, int, int>> initial_edges;
  std::vector<int> fa(n_points_, 0);
  std::iota(fa.begin(), fa.end(), 0);
  for (int i = 0; i < n_points_; i++) {
    fa[i] = i;
  }
  std::function<int(int)> FindRoot;
  FindRoot = [&fa, &FindRoot](int a) -> int {
    return fa[a] == a ? a : (fa[a] = FindRoot(fa[a]));
  };
  StopWatch stop_watch;
  for (int u = 0; u < n_points_; u++) {
    const auto& p = (*points_)[u];
    std::vector<int> neighbors;
    double current_searching_r = points_radius == nullptr ? r : std::max(r, points_radius->at(u) * 2.0);
    octree->SearchingR(p, current_searching_r, &neighbors);
    if (neighbors.size() <= 1) {
      continue;
    }
    for (int v : neighbors) {
      if (v <= u) {
        continue;
      }
      const auto& q = (*points_)[v];
      Eigen::Vector3d bias = q - p;
      double v_searching_r = points_radius == nullptr ? r : std::max(r, points_radius->at(v) * 2.0);
      if (bias.norm() > v_searching_r || bias.norm() > current_searching_r) {
        continue;
      }
      double dis_sqr = bias.squaredNorm();
      double weight = std::exp(-dis_sqr / (r * r) * 0.5);
      if (weight < 1e-9) {
        continue;
      }
      initial_edges.emplace_back(-weight, u, v);
    }
  }
  // LOG(INFO) << "Gen initial edges duration: " << stop_watch.TimeDuration();

  std::sort(initial_edges.begin(), initial_edges.end());
  // LOG(INFO) << "Sort duration: " << stop_watch.TimeDuration();

  std::vector<double> out_length_sum(n_points_, 0.0);
  std::vector<int> vis(n_points_, 0);
  int timestamp = 0;
  for (const auto& initial_edge : initial_edges) {
    int a = std::get<1>(initial_edge);
    int b = std::get<2>(initial_edge);
    double weight = -std::get<0>(initial_edge);
    if (FindRoot(a) != FindRoot(b)) {
      fa[fa[a]] = fa[b];
      edges_[a].emplace_back(b, weight);
      edges_[b].emplace_back(a, weight);
      Eigen::Vector3d a_to_b = (*points_)[b] - (*points_)[a];
      a_to_b /= a_to_b.norm();
      continue;
    }
    CHECK(!edges_[a].empty() && !edges_[b].empty());
    Eigen::Vector3d a_to_b = (*points_)[b] - (*points_)[a];
    std::vector<std::pair<int, int>> que;
    vis[a] = ++timestamp;
    que.emplace_back(a, 0);
    const int kPathLengthThreshold = std::max(20, (int) std::round(a_to_b.norm() / hope_dist_ * 4.0));
    int L = 0;
    bool found = false;
    while (que.back().second < kPathLengthThreshold && !found) {
      const auto& pr = que[L++];
      int u = pr.first;
      int d = pr.second;
      for (const auto& edge : edges_[u]) {
        int v = edge.first;
        if (vis[v] == timestamp) {
          continue;
        }
        vis[v] = timestamp;
        que.emplace_back(v, d + 1);
        if (v == b) {
          found = true;
          break;
        }
      }
    }

    if (found) {
      continue; // Find small circle.
    }

    // if (edges_[a].size() > 1 && edges_[b].size() > 1) {
    //   if (a_to_b.norm() > out_length_sum[a] / edges_[a].size() * 30000.0 &&
    //      a_to_b.norm() > out_length_sum[b] /  edges_[b].size() * 30000.0) {
    //    continue;
    //  }
    // }

    // Handle this kind of situation:
    // ----------x-x-x-x-x-x-x------------
    //                 |
    //                 |
    // ----------x-x-x-x-x-x-x------------

    if (edges_[a].size() == 2 && edges_[b].size() == 2) {
      auto IsSmooth = [this, &points](int base_u) -> bool {
        const int kLocalArrayLen = 7;
        const int m = kLocalArrayLen / 2;
        std::vector<int> local_array(kLocalArrayLen, -1);
        local_array[m] = base_u;
        local_array[m - 1] = edges_[base_u].front().first;
        local_array[m + 1] = edges_[base_u].back().first;
        for (int i = 1; i < m; i++) {
          for (int k = -1; k <= 1; k += 2) {
            int u = local_array[m + k * i];
            CHECK_GE(u, 0);
            if (edges_[u].size() != 2) {
              return false;
            }
            for (const auto& edge : edges_[u]) {
              int v = edge.first;
              if (v == local_array[m + k * (i - 1)]) {
                continue;
              }
              local_array[m + k * (i + 1)] = v;
            }
          }
        }
        // LOG(INFO) << local_array[0] << " "
        //           << local_array[1] << " "
        //           << local_array[2] << " "
        //           << local_array[3] << " "
        //           << local_array[4] << " "
        //           << local_array[5] << " "
        //           << local_array[6] << " ";
        CHECK_GE(local_array.front(), 0);
        CHECK_GE(local_array.back(), 0);
        Eigen::Vector3d vec_a = (points[base_u] - points[local_array.front()]).normalized();
        Eigen::Vector3d vec_b = (points[base_u] - points[local_array.back()]).normalized();
        return (vec_a.dot(vec_b) < -0.8);
      };
      if (IsSmooth(a) && IsSmooth(b)) {
        continue;
      }
    }

    edges_[a].emplace_back(b, weight);
    edges_[b].emplace_back(a, weight);
    out_length_sum[a] += a_to_b.norm();
    out_length_sum[b] += a_to_b.norm();
  }
  // LOG(INFO) << "Build duration: " << stop_watch.TimeDuration();
}


IronTown::IronTown(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<std::vector<std::pair<int, double>>>& edges) {
  edges_ = edges;
  points_ = &points;
  n_points_ = points.size();
}

void IronTown::GetPaths(std::vector<std::vector<int>>* paths) {
  std::vector<int> vis(n_points_, 0);
  paths->clear();
  for (int u = 0; u < n_points_; u++) {
    int d = Degree(u);
    if (d == 2) {
      continue;
    }
    if (d == 0) {
      paths->emplace_back();
      paths->back().push_back(u);
      continue;
    }
    for (const auto& pr : edges_[u]) {
      int v = pr.first;
      int d_v = Degree(v);
      if ((d_v != 2 && v < u) || (d_v == 2 && vis[v])) {
        continue;
      }
      paths->emplace_back();
      auto& new_path = paths->back();
      new_path.push_back(u);
      while (Degree(v) == 2) {
        vis[v] = 1;
        int new_v = v;
        for (const auto& v_edge : edges_[v]) {
          if (v_edge.first != new_path.back()) {
            new_v = v_edge.first;
            break;
          }
        }
        CHECK(new_v != v) << v << " " <<new_path.back() << " " << edges_[v][0].first << " " << edges_[v][1].first;
        CHECK(Degree(new_v) != 2 || (Degree(new_v) == 2 && !vis[new_v]));
        new_path.push_back(v);
        v = new_v;
      }
      new_path.push_back(v);
    }
  }

  // Process pure loops
  for (int u = 0; u < n_points_; u++) {
    if (Degree(u) != 2 || vis[u]) {
      continue;
    }
    paths->emplace_back();
    auto& new_path = paths->back();
    vis[u] = 1;
    new_path.push_back(u);
    while (true) {
      int v = new_path.back();
      int new_v = -1;
      for (const auto& v_edge : edges_[v]) {
        if (vis[v_edge.first]) {
          continue;
        }
        CHECK(new_v == -1 || v == u);
        new_v = v_edge.first;
        vis[new_v] = 1;
        new_path.push_back(new_v);
      }
      if (new_v == -1) {
        break;
      }
    }
    CHECK(new_path.size() > 1);
    new_path.push_back(u);
  }
}

// -------------------------------------- SelfDefinedGraph -----------------------------------------------------

SelfDefinedGraph::SelfDefinedGraph(const std::vector<Eigen::Vector3d>& points,
                                   const std::vector<std::vector<std::pair<int, double>>>& edges,
                                   const std::vector<std::vector<int>>& paths) {
  edges_ = edges;
  paths_ = paths;
  points_ = &points;
  n_points_ = points.size();
}

void SelfDefinedGraph::GetPaths(std::vector<std::vector<int>>* paths) {
  *paths = paths_;
}
