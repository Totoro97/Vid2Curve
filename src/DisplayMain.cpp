//
// Created by aska on 2019/9/26.
//

#include <limits>
#include <iostream>
#include <thread>

#ifdef USE_GUI
#include <pangolin/pangolin.h>
#endif

#include "Reconstructor.h"
#include "Utils/Common.h"
#include "Utils/GlobalDataPool.h"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  FLAGS_minloglevel = 3; // LOG level: 3 -> FATAL
  LOG(INFO) << "Hello Wooden!";

#ifdef USE_GUI
  // Build Global data pool
  auto global_data_pool = std::make_unique<GlobalDataPool>();

  // Build Reconstructor
  auto reconstructor = std::make_unique<Reconstructor>("../config.ini", global_data_pool.get());
  std::thread reconstruct_worker(&Reconstructor::Run, reconstructor.get());

  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 640, 480);

  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
      pangolin::ModelViewLookAt(-1,1,-1, 0,0,0, pangolin::AxisY));

  pangolin::View& d_cam = pangolin::Display("cam")
      .SetBounds(0,1.0f,0,1.0f,-640/480.0)
      .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::View& d_image = pangolin::Display("image")
      .SetBounds(3/4.0f,1.0f,0,1/4.0f, double(reconstructor->width_) / reconstructor->height_)
      .SetLock(pangolin::LockLeft, pangolin::LockBottom);

  pangolin::GlTexture image_texture(reconstructor->width_,
                                    reconstructor->height_,
                                    GL_RGB,
                                    false,
                                    0,
                                    GL_RGB,
                                    GL_UNSIGNED_BYTE);

  while(!pangolin::ShouldQuit() && reconstructor->state_ == Reconstructor::State::RUNNING) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    d_cam.Activate(s_cam);

    // Draw cameras.
    global_data_pool->DrawCameras();

    // Draw points.
    pangolin::GlBuffer glxyz(pangolin::GlArrayBuffer, global_data_pool->n_points(), GL_FLOAT, 3, GL_DYNAMIC_DRAW);

    glColor3f(1.0, 1.0, 1.0);
    global_data_pool->DrawModelPoints(&glxyz);
    global_data_pool->model_points_mutex_.unlock();
    pangolin::RenderVbo(glxyz, GL_POINTS);

    // Draw images.
    global_data_pool->DrawImage(&image_texture);
    d_image.Activate();
    image_texture.RenderToViewport();

    pangolin::FinishFrame();
  }

  pangolin::QuitAll();
  reconstruct_worker.join();
#else
  auto reconstructor = std::make_unique<Reconstructor>("../config.ini", nullptr);
  reconstructor->Run();
#endif
  return 0;
}
