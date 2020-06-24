#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};
int resize_uniform(Mat& src, Mat& dst, Size dst_size, object_rect& effect_area);
int crop_effect_area(Mat& uniform_scaled, Mat& dst, Size ori_size, object_rect effect_area);