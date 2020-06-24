#include <math.h>
#include <iostream>
#include "imageTools.h"





int resize_uniform(Mat& src, Mat& dst, Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = Mat(Size(dst_w, dst_h), CV_8UC3, Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    Mat tmp;
    resize(src, tmp, Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) { //高对齐，宽没对齐
        int index_w = floor((dst_w - tmp_w) / 2.0);
        std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) { //宽对齐， 高没有对齐
        int index_h = floor((dst_h - tmp_h) / 2.0);
        std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}

int crop_effect_area(Mat& uniform_scaled, Mat& dst, Size ori_size, object_rect effect_area)
{
    Mat tmp = Mat(Size(effect_area.width, effect_area.height), CV_8UC3, Scalar(0));

    if (effect_area.x == 0 && effect_area.y == 0) {
        resize(uniform_scaled, dst, ori_size);
        return 0;
    }
    else if (effect_area.x == 0) {
        memcpy(tmp.data, uniform_scaled.data + effect_area.y * effect_area.width * 3, effect_area.width * effect_area.height * 3);
    }
    else if (effect_area.y == 0) {
        for (int i = 0; i < effect_area.height; i++) {
            memcpy(tmp.data + i * effect_area.width * 3, uniform_scaled.data + i * uniform_scaled.cols * 3 + effect_area.x * 3, effect_area.width * 3);
        }
    }
    resize(tmp, dst, ori_size);
    return 0;
}


int crop_effect_area_gray(Mat& uniform_scaled, Mat& dst, Size ori_size, object_rect effect_area)
{
    Mat tmp = Mat(Size(effect_area.width, effect_area.height), CV_8U, Scalar(0));

    if (effect_area.x == 0 && effect_area.y == 0) {
        resize(uniform_scaled, dst, ori_size);
        return 0;
    }
    else if (effect_area.x == 0) {
        memcpy(tmp.data, uniform_scaled.data + effect_area.y * effect_area.width , effect_area.width * effect_area.height );
    }
    else if (effect_area.y == 0) {
        for (int i = 0; i < effect_area.height; i++) {
            memcpy(tmp.data + i * effect_area.width , uniform_scaled.data + i * uniform_scaled.cols  + effect_area.x , effect_area.width );
        }
    }
    resize(tmp, dst, ori_size);
    return 0;
}
//int main()
//{
//    Mat img = imread("0.jpg", 3);
//    Mat dst;
//    object_rect res_area;
//    //缩放
//    (void)resize_uniform(img, dst, Size(608, 608), res_area);
//    //imwrite("out.jpg", dst);
//    std::cout << "effectiave area: (" << res_area.x << ", " << res_area.y << ", " << res_area.width << ", " << res_area.height << ")" << std::endl;
//    //Mat out = imread("a.jpg", 3);
//    Mat out = dst;
//    Mat recov;
//    //还原
//    crop_effect_area(out, recov, Size(img.cols, img.rows), res_area);
//    //imwrite("recov.jpg", recov);
//    return 0;
//}