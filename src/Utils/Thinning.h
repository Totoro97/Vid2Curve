/**
 * Code for thinning a binary image using Zhang-Suen algorithm.
 *
 * Author:  Nash (nash [at] opencv-code [dot] com)
 * Website: http://opencv-code.com
 */
#include <opencv2/opencv.hpp>

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * Parameters:
 * 		im    Binary image with range = [0,1]
 * 		iter  0=even, 1=odd
 */
void thinningIteration(cv::Mat& img, int iter)
{
  CV_Assert(img.channels() == 1);
  CV_Assert(img.depth() != sizeof(uchar));
  CV_Assert(img.rows > 3 && img.cols > 3);

  cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

  int nRows = img.rows;
  int nCols = img.cols;

  if (img.isContinuous()) {
    nCols *= nRows;
    nRows = 1;
  }

  int x, y;
  uchar *pAbove;
  uchar *pCurr;
  uchar *pBelow;
  uchar *nw, *no, *ne;    // north (pAbove)
  uchar *we, *me, *ea;
  uchar *sw, *so, *se;    // south (pBelow)

  uchar *pDst;

  // initialize row pointers
  pAbove = NULL;
  pCurr  = img.ptr<uchar>(0);
  pBelow = img.ptr<uchar>(1);

  for (y = 1; y < img.rows-1; ++y) {
    // shift the rows up by one
    pAbove = pCurr;
    pCurr  = pBelow;
    pBelow = img.ptr<uchar>(y+1);

    pDst = marker.ptr<uchar>(y);

    // initialize col pointers
    no = &(pAbove[0]);
    ne = &(pAbove[1]);
    me = &(pCurr[0]);
    ea = &(pCurr[1]);
    so = &(pBelow[0]);
    se = &(pBelow[1]);

    for (x = 1; x < img.cols-1; ++x) {
      // shift col pointers left by one (scan left to right)
      nw = no;
      no = ne;
      ne = &(pAbove[x+1]);
      we = me;
      me = ea;
      ea = &(pCurr[x+1]);
      sw = so;
      so = se;
      se = &(pBelow[x+1]);

      int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
               (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
               (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
               (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
      int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
      int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
      int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

      if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
        pDst[x] = 1;
    }
  }

  img &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * Parameters:
 * 		src  The source image, binary with range = [0,255]
 * 		dst  The destination image
 */
void thinning(const cv::Mat& src, cv::Mat& dst)
{
  dst = src.clone();
  dst /= 255;         // convert to binary image

  cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
  cv::Mat diff;

  do {
    thinningIteration(dst, 0);
    thinningIteration(dst, 1);
    cv::absdiff(dst, prev, diff);
    dst.copyTo(prev);
  }
  while (cv::countNonZero(diff) > 0);

  dst *= 255;
}

double CalcThinnedRatio(const cv::Mat& src, const cv::Mat& dst) {
  int n_edge_pixels = 0;
  int n_thinned_pixels = 0;
  int height = src.rows;
  int width = src.cols;
  for (int a = 0; a < height; a++) {
    for (int b = 0; b < width; b++) {
      if (!src.data[a * width + b]) {
        continue;
      }
      if (!dst.data[a * width + b]) {
        n_thinned_pixels++;
      }
      bool is_edge = false;
      for (int x = std::max(0, a - 1); !is_edge && x <= std::min(height - 1, a + 1); x++) {
        for (int y = std::max(0, b - 1); !is_edge && y <= std::min(width - 1, b + 1); y++) {
          if (!src.data[x * width + b]) {
            is_edge = true;
          }
        }
      }
      if (is_edge) {
        n_edge_pixels++;
      }
    }
  }
  if (!n_edge_pixels) {
    return 0.0;
  }
  return double(n_thinned_pixels) / double(n_edge_pixels);
}

void PartialThinning(const cv::Mat& src, cv::Mat& dst, double thershold = 0.2) {
  dst = src.clone();
  dst /= 255;

  cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
  dst.copyTo(prev);

  double thinned_ratio = 1.0;
  do {
    thinningIteration(dst, 0);
    thinned_ratio = CalcThinnedRatio(prev, dst);
    // std::cout << thinned_ratio << std::endl;
    // cv::imshow("haha", dst * 255);
    // cv::waitKey(-1);
    dst.copyTo(prev);

    thinningIteration(dst, 1);
    thinned_ratio = CalcThinnedRatio(prev, dst);
    // std::cout << thinned_ratio << std::endl;
    // cv::imshow("haha", dst * 255);
    // cv::waitKey(-1);
    dst.copyTo(prev);
  } while (thinned_ratio > thershold);

  dst *= 255;
}

/**
 * This is an example on how to call the thinning funciton above
 */
/*
int main()
{
  cv::Mat src = cv::imread("image.png");
  if (!src.data)
    return -1;

  cv::Mat bw;
  cv::cvtColor(src, bw, CV_BGR2GRAY);
  cv::threshold(bw, bw, 10, 255, CV_THRESH_BINARY);

  thinning(bw, bw);

  cv::imshow("src", src);
  cv::imshow("dst", bw);
  cv::waitKey();
  return 0;
}
*/