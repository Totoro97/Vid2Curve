#pragma once
#include "Common.h"
#include <vector>

class SweepSurface{
protected:
  int cross_sec_div_num_ = 32;
public:
  SweepSurface(){}
  ~SweepSurface(){}

  void SetCrossSecDivNum(int num){cross_sec_div_num_ = num;}
  void GenSphere(Eigen::Vector3d center, double radius, int subdiv_num, Eigen::MatrixXd &res_verts, Eigen::MatrixXi & res_faces)
  {
    res_verts.resize(2 + subdiv_num*(subdiv_num - 1), 3);
    //res_faces.resize((subdiv_num-1)*2,3);
    res_faces.resize((subdiv_num-1)*subdiv_num*2, 3);

    Eigen::Vector3d updir(0, 0, 1);
    Eigen::Vector3d xdir(1, 0, 0);
    Eigen::Vector3d ydir(0, 1, 0);

    res_verts.row(0) = center + updir*radius;
    double angular_step = 3.1415926 * 2 / subdiv_num;
    double vert_angular_step = 3.1415926 / subdiv_num;
    for (int i = 1; i < subdiv_num; i++)
    {
      Eigen::Vector3d tmp_center = center + updir *radius*std::cos(vert_angular_step*i);
      double tmp_radius = std::sin(vert_angular_step*i)*radius;
      for (int j = 0; j < subdiv_num; j++)
      {
        double degree = angular_step*j;
        Eigen::Vector3d p = std::cos(degree) * tmp_radius * xdir + std::sin(degree) * tmp_radius * ydir + tmp_center;
        res_verts.row(1+(i-1)*subdiv_num+j)=p;
      }
    }
    res_verts.row(1 + subdiv_num*(subdiv_num - 1)) = center - updir*radius;

    int fid = 0;
    for (int i = 0; i < subdiv_num; i++)
    {
      res_faces.row(fid++) = Eigen::Vector3i(0,1+i, 1+(i + 1) % subdiv_num);
    }

    for (int i = 0; i < subdiv_num-2; i++)
    {
      for (int j = 0; j < subdiv_num; j++)
      {
        if (j != subdiv_num - 1)
        {
          res_faces.row(fid++) = Eigen::Vector3i(1 + i*subdiv_num + j, 1 + (i + 1)*subdiv_num + j, 2 + (i + 1)*subdiv_num + j);
          res_faces.row(fid++) = Eigen::Vector3i(1 + i*subdiv_num + j, 2 + (i + 1)*subdiv_num + j, 2 + i*subdiv_num + j);
        }
        else
        {
          res_faces.row(fid++) = Eigen::Vector3i(1 + i*subdiv_num + j, 1 + (i + 1)*subdiv_num + j, 1 + (i + 1)*subdiv_num + 0);
          res_faces.row(fid++) = Eigen::Vector3i(1 + i*subdiv_num + j, 1 + (i + 1)*subdiv_num + 0, 1 + i*subdiv_num + 0);
        }
      }
    }


    for (int i = 0; i < subdiv_num; i++)
    {
      res_faces.row(fid++) = Eigen::Vector3i(subdiv_num*(subdiv_num - 2) + 1 + i, res_verts.rows() - 1, subdiv_num*(subdiv_num - 2) + 1 + (i + 1) % subdiv_num);
    }

  }
  void GenSweepSurface(std::vector<Eigen::Vector3d>& in_path,
                       std::vector<double>& in_radius,
                       Eigen::MatrixXd& res_mesh_verts,
                       Eigen::MatrixXi& res_mesh_faces){
    int vert_num = cross_sec_div_num_ * in_path.size();
    int face_num = cross_sec_div_num_ * 2 * (in_path.size()-1);
    res_mesh_verts.resize(vert_num, 3);
    res_mesh_faces.resize(face_num, 3);
    double angular_step = 3.1415926 * 2 / cross_sec_div_num_;
    int a[100];
    Eigen::Vector3d pre_dir(0,0,0);

    for(int pi=0; pi < in_path.size(); pi++){
      if (in_path.size() <= 1) {
        continue;
      }
      Eigen::Vector3d tan_dir;
      int a, b, c;
      if (pi > 0 && pi + 1 < in_path.size()) {
        a = pi - 1;
        b = pi;
        c = pi + 1;
        tan_dir = in_path[pi + 1] - in_path[pi - 1];
      } else if ((in_path.front() - in_path.back()).norm() < 1e-7) { // Circular
        // CHECK_GE(in_path.size(), 4);
        a = in_path.size() - 2;
        b = pi;
        c = 1;
        tan_dir = in_path[c] - in_path[a];
      } else if (pi == 0) {
        tan_dir = in_path[pi + 1] - in_path[pi];
      } else if (pi == in_path.size() - 1) { 
        tan_dir = in_path[pi] - in_path[pi - 1];
      }
      else {
        LOG(FATAL) << "here.";
      }
      tan_dir.normalize();
      Eigen::Vector3d para_dir(0, -tan_dir(2), tan_dir(1));
      if (std::abs(tan_dir(1)) < 1e-7 && std::abs(tan_dir(2)) < 1e-7) {
        para_dir(1) = 1;
      }
      if(pi > 0) {
        double lambda = -(pre_dir.dot(tan_dir))/tan_dir.dot(tan_dir);
        para_dir = pre_dir + lambda*tan_dir;
      }
      para_dir.normalize();


      Eigen::Vector3d para_dir2 = tan_dir.cross(para_dir);
      para_dir2.normalize();


      pre_dir = para_dir;
      for(int di=0; di<cross_sec_div_num_; di++){
        double degree = angular_step * di;
        Eigen::Vector3d p = std::cos(degree) * in_radius[pi] * para_dir + std::sin(degree) * in_radius[pi] * para_dir2 + in_path[pi];
        res_mesh_verts.row(pi*cross_sec_div_num_ + di) = p;
      }
    }

    for(int pi=0; pi<in_path.size()-1; pi++){
      for(int di=0; di<cross_sec_div_num_; di++) {
        int fid = cross_sec_div_num_ * 2 * pi + di*2;
        int vid0 = pi*cross_sec_div_num_ + di;
        int vid1 = pi*cross_sec_div_num_ + (di + 1)%cross_sec_div_num_;
        res_mesh_faces.row(fid) = Eigen::Vector3i(vid0, vid1, vid0+cross_sec_div_num_);
        fid++;
        res_mesh_faces.row(fid) = Eigen::Vector3i(vid1, vid1+cross_sec_div_num_, vid0+cross_sec_div_num_);
      }
    }
  }
  void GenSweepSurface(std::vector<Eigen::Vector3d>& in_vertices,
                       std::vector<std::vector<int>>& in_curves,
                       std::vector<std::vector<double>> in_radius,
                       Eigen::MatrixXd& res_mesh_verts,
                       Eigen::MatrixXi& res_mesh_faces) {
    std::vector<Eigen::MatrixXd> mesh_verts_list;
    std::vector<Eigen::MatrixXi> mesh_faces_list;
    std::vector<int>verts_degree(in_vertices.size(),0);
    std::vector<double>verts_radius(in_vertices.size(), 0);
    for (int i = 0; i < in_curves.size(); i++)
    {
      std::vector<int>&curve = in_curves[i];
      verts_degree[curve[0]]++;
      verts_degree[curve.back()]++;
      verts_radius[curve[0]] += in_radius[i][0];
      verts_radius[curve.back()] += in_radius[i].back();
    }
    std::vector<Eigen::Vector3d> ball_centers;
    std::vector<double> ball_radius;
    for (int i = 0; i < verts_degree.size(); i++)
    {
      if (verts_degree[i] > 0)
      {
        verts_radius[i] /= verts_degree[i];
      }
      // if (verts_degree[i] == 1) {
      if (verts_degree[i] != 2 && verts_degree[i] > 0) {
        ball_centers.push_back(in_vertices[i]);
        ball_radius.push_back(verts_radius[i]);
      }
    }
    // ball_centers.clear();
    // ball_radius.clear();
    for (int i = 0; i < in_curves.size(); i++)
    {
      std::vector<int>&curve = in_curves[i];
      int vid0 = curve[0];
      int vid1 = curve.back();
      in_radius[i][0] = verts_radius[vid0];
      in_radius[i][in_radius[i].size()-1] = verts_radius[vid1];
    }

    for (int i = 0; i < in_curves.size(); i++)
    {
      std::vector<int>&curve = in_curves[i];
      std::vector<Eigen::Vector3d> tmpvs;
      for (int j = 0; j < curve.size(); j++)
      {
        tmpvs.push_back(in_vertices[curve[j]]);
      }

      Eigen::MatrixXd tmp_mvs;
      Eigen::MatrixXi tmp_mfs;
      GenSweepSurface(tmpvs, in_radius[i], tmp_mvs, tmp_mfs);
      mesh_verts_list.push_back(tmp_mvs);
      mesh_faces_list.push_back(tmp_mfs);

    }

    for (int i = 0; i < ball_centers.size(); i++)
    {
      Eigen::MatrixXd tmp_mvs;
      Eigen::MatrixXi tmp_mfs;
      GenSphere(ball_centers[i], ball_radius[i], cross_sec_div_num_, tmp_mvs, tmp_mfs);
      mesh_verts_list.push_back(tmp_mvs);
      mesh_faces_list.push_back(tmp_mfs);
    }
    Utils::MergeTriMeshes(mesh_verts_list, mesh_faces_list, res_mesh_verts, res_mesh_faces);
  }

};
