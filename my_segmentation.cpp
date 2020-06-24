
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include "segmentation_onnx.h"
//#include <onnxruntime_cxx_inline.h>
//#include <cuda_provider_factory.h>
//#include <cpu_provider_factory.h>
//#include <onnxruntime_cxx_api.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"


#include <chrono>

static constexpr const int width_ = 640;
static constexpr const int height_ = 480;
static constexpr const int channel = 4;

std::array<float, 1 * width_ * height_ * channel> input_image_{};
//std::array<float, 1 * width_ * height_ * 1>              results_{};
cv::Mat results_ = cv::Mat::zeros(height_, width_, CV_32F);

#ifdef _WIN32
//const wchar_t* model_path = L"4channels384_640.onnx";
//const wchar_t* model_path = L"D:/pengt/code/Cplus/onnx_model/resnet101_21_384x640.onnx";
const wchar_t* model_path = L"D:/pengt/code/Cplus/onnx_model/resnet101_21_480x640.onnx";

#else
const char* model_path = "4channels384_640.onnx";
#endif


#define USE_CUDA

class ONNX_Model
{
public:
#ifdef _WIN32
    ONNX_Model(const wchar_t* model_path)
        : m_env{ ORT_LOGGING_LEVEL_ERROR, "" },
        m_session{ nullptr },
        m_sess_opts{},
        m_mem_info{ Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault) }
#else
    ONNX_Model(char* model_path)
#endif
    {
         //option
         /*  if (true)
            m_sess_opts.EnableCpuMemArena();
        else
            m_sess_opts.DisableCpuMemArena();*/

        //end
        
        /*
        // DirectML
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(m_sess_opts, 0));
        m_sess_opts.DisableMemPattern();
        m_sess_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        m_session = Ort::Session(m_env, model_path, m_sess_opts);
        */

        
        // CPU
       // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(m_sess_opts, 0));
       // m_sess_opts.EnableMemPattern();
       // m_sess_opts.SetIntraOpNumThreads(8);
        //m_sess_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
       // m_sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
       // m_session = Ort::Session(m_env, model_path, m_sess_opts);
        //tensorrt
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(m_sess_opts,0));
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(m_sess_opts, 0));

        //m_sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        m_sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        //m_sess_opts.SetOptimizedModelFilePath(out_optimize_path);  //save optimize

        m_session = Ort::Session(m_env, model_path, m_sess_opts);
        //m_session = Ort::Session(m_env, out_optimize_path, m_sess_opts);
        //end tensorrt
        

         // CUDA
        //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(m_sess_opts, 0));
        //m_sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        //m_session = Ort::Session(m_env, model_path, m_sess_opts);
       
        
        //
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
        //output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, (float*)results_.data, height_ * width_, output_shape_.data(), output_shape_.size());

    }
    void  Run()
    {
        m_session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), &output_tensor_, 1);
        return;
    }

    std::vector<int64_t> get_input_shape_from_session()
    {
        Ort::TypeInfo info = m_session.GetInputTypeInfo(0);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        size_t dim_count = tensor_info.GetDimensionsCount();
        std::vector<int64_t> dims(dim_count);
        tensor_info.GetDimensions(dims.data(), dims.size());
        return dims;
    }
private:
    Ort::Env m_env;
    Ort::Session m_session;
    Ort::SessionOptions m_sess_opts;
    Ort::MemoryInfo m_mem_info;
    //std::vector<const char*> input_names{ "0" };
    //std::vector<const char*> output_names{ "293" };  //Unet

    std::vector<const char*> input_names{ "input.1" };
    std::vector<const char*> output_names{ "1207" };  //resnet101
  
    //std::vector<const char*> input_names{  "input.1" };
    //std::vector<const char*> output_names{  "1683" };  //resnet152

    Ort::Value              input_tensor_{ nullptr };
    std::array<int64_t, 4>  input_shape_{ 1, channel,  height_,width_ };
    Ort::Value              output_tensor_{ nullptr };
    std::array<int64_t, 4>  output_shape_{ 1, 1, height_,width_ };

};


// data need 

//rgbm = rgbm / 255.
//mean = [0.485, 0.456, 0.406, 0]
//std = [0.229, 0.224, 0.225, 1]
//rgbm = rgbm - mean
//rgbm = rgbm / std
//x.transpose(2, 0, 1).astype('float32')

#if 0
void fill_data(cv::Mat input_img, cv::Mat pre_mask, float* output, const int index = 0)
{
    cv::Mat dst_img;
    cv::Mat dst_pre;
    cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
    cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);
    float scale = 0.00392;
    //cv::Scalar mean_ = cv::Scalar(109.125, 102.6, 91.35);
    //cv::Scalar std_ = cv::Scalar(0.0171, 0.01884, 0.01975);
    //if (images.depth() == CV_8U && ddepth == CV_32F)
    //    images.convertTo(images[i], CV_32F);
    //std::swap(mean[0], mean[2]);
    input_img.convertTo(dst_img, CV_32F, scale);
    pre_mask.convertTo(dst_pre, CV_32F, scale);

    dst_img -= mean;
    dst_img /= std;

    //dst_img -= mean_;
    //dst_img *= std_;
    //std::vector<cv::Mat> channels;
    //split(input_img, channels);//²ð·Ö
    //channels.push_back(pre_mask);
    //cv::convertTo(img_float, CV_32F, 1.0 / 255);

    int row = dst_img.rows;
    int col = dst_img.cols;

    //cv::Scalar rgb_mean = cv::mean(dst);
    //std::cout<< (dst_img.ptr<float>(0, 214)[0]) <<std::endl;


    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                //output[c * row * col + i * col + j] = (dst_img.ptr<uchar>(i)[j * 3 + c]);
                //std::cout << "i and j :"<<i<<","<<j<<"="<<(dst_img.ptr<float>(i,j)[c]) << std::endl;
                output[c * row * col + i * col + j] = (dst_img.ptr<float>(i, j)[c]);
            }
        }
    }
    if (index % 20 == 0)
    {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                //output[4 * row * col + i * col + j] = (dst_pre.ptr<uchar>(i)[j]);
                output[3 * row * col + i * col + j] = (dst_pre.ptr<float>(i, j)[0]);
            }
        }
    }

    return;
}
#else
void fill_data(cv::Mat input_img, cv::Mat pre_mask, float* output, const int index = 0)
{
    float mean[3] = { 123.675, 116.28, 103.53 };
    float std[3] = { 0.0171, 0.0175, 0.0174 };
    float allmul[3] = { 2.1145,2.0349,1.8014 };
    float scale = 1;
    float* dst_pre = pre_mask.ptr<float>(0, 0);
    int row = input_img.rows;
    int col = input_img.cols;
    int alllen = row * col;
    //double timeStart = (double)cv::getTickCount();
    //HWC -> CHW
    float* inbuf = output;
    for (int c = 0; c < 3; c++) {
        uchar* img_data = input_img.data;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                *inbuf = img_data[c] * std[c] - allmul[c];//(dst_img.ptr<float>(i, j)[c]);
                img_data += 3;
                inbuf++;
            }
        }
    }

    //double circle = ((double)cv::getTickCount() - timeStart) / cv::getTickFrequency();
    //std::cout << "circle time  ：" << circle << " sec";
    memcpy(inbuf, dst_pre, sizeof(float) * alllen);

    return;
}
#endif

using namespace cv;
std::vector<Vec3b> colors;

void colorizeSegmentation(const Mat& score, Mat& segm)
{
    const int rows = score.size[2];
    const int cols = score.size[3];
    const int chns = score.size[1];

    if (colors.empty()) {
        // Generate colors.
        colors.push_back(Vec3b());
        for (int i = 1; i < chns; ++i) {
            Vec3b color;
            for (int j = 0; j < 3; ++j)
                color[j] = (colors[i - 1][j] + rand() % 256) / 2;
            colors.push_back(color);
        }
    }
    else if (chns != (int)colors.size()) {
        CV_Error(Error::StsError, format("Number of output classes does not match "
            "number of colors (%d != %zu)", chns, colors.size()));
    }

    Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1, score.data);
    for (int ch = 1; ch < chns; ch++) {
        for (int row = 0; row < rows; row++) {
            const float* ptrScore = score.ptr<float>(0, ch, row);
            uint8_t* ptrMaxCl = maxCl.ptr<uint8_t>(row);
            float* ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++) {
                if (ptrScore[col] > ptrMaxVal[col]) {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = (uchar)ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++) {
        const uchar* ptrMaxCl = maxCl.ptr<uchar>(row);
        Vec3b* ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++) {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }
}

// 背景，  前景， mask
cv::Mat replace_and_blend(cv::Mat bkimg, cv::Mat& frame, cv::Mat& mask)
{
    cv::Mat result = cv::Mat::zeros(frame.size(), frame.type());
    int h = frame.rows;
    int w = frame.cols;
    int m = 0;
    double wt = 0;

    int b = 0, g = 0, r = 0;
    int b1 = 0, g1 = 0, r1 = 0;
    int b2 = 0, g2 = 0, r2 = 0;

    for (int row = 0; row < h; row++)
    {
        uchar* current = frame.ptr<uchar>(row);
        uchar* bgrow = bkimg.ptr<uchar>(row);
        uchar* maskrow = mask.ptr<uchar>(row);
        uchar* targetrow = result.ptr<uchar>(row);

        for (int col = 0; col < w; col++)
        {
            m = *maskrow++;
            if (m == 0)   //如果是背景 替换为背景数据
            {
                *targetrow++ = *bgrow++;
                *targetrow++ = *bgrow++;
                *targetrow++ = *bgrow++;
                current += 3;
            }
            else if (m == 255) //如果是前景 保留原来数据
            {
                *targetrow++ = *current++;
                *targetrow++ = *current++;
                *targetrow++ = *current++;
                bgrow += 3;
            }
            else //由于形态学平滑造成的过渡区 颜色采用加权均衡化
            {
                b1 = *bgrow++;
                g1 = *bgrow++;
                r1 = *bgrow++;

                b2 = *current++;
                g2 = *current++;
                r2 = *current++;

                wt = m / 255.0;

                b = b2 * wt + b1 * (1 - wt);
                g = g2 * wt + g1 * (1 - wt);
                r = b2 * wt + r1 * (1 - wt);

                *targetrow++ = b;
                *targetrow++ = g;
                *targetrow++ = r;

            }
        }
    }
    return result;  //返回结果
}


#if 0
int main(void)
{
    ONNX_Model model;
    std::vector<int64_t> dims = model.get_input_shape_from_session();
    std::cout << "Input Shape: (";
    std::cout << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << ")" << std::endl;
    int inputwidth = 640;
    int inputheight = 384;
    cv::Mat pre_mask = cv::Mat::zeros(inputheight, inputwidth, CV_8UC1);  //height  width
    cv::Mat frame, image0;
    frame = cv::imread("D:/pengt/segmetation/test_pic/1.png");
    cv::resize(frame, frame, cv::Size(inputwidth, inputheight));  //width height
    cv::cvtColor(frame, image0, cv::COLOR_BGR2RGB);

    double timeStart = (double)cv::getTickCount();
    float* output = input_image_.data();

    std::fill(input_image_.begin(), input_image_.end(), 0.f);
    fill_data(image0, pre_mask, output);

    model.Run();
    double nTime = ((double)cv::getTickCount() - timeStart) / cv::getTickFrequency();
    std::cout << "running time £º" << nTime << "sec\n" << std::endl;

    cv::Mat segm;
    //colorizeSegmentation(results_, segm);
    cv::threshold(results_, segm, 0.5, 200, cv::THRESH_BINARY);

    cv::imshow("mask", segm);
    cv::imshow("pre_image", frame);
    cv::waitKey(0);

    return 0;
}


#else
int main(void)
{
    ONNX_Model model(model_path);
    std::vector<int64_t> dims = model.get_input_shape_from_session();
    std::cout << "Input Shape: (";
    std::cout << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << ")" << std::endl;
    int inputwidth = width_;
    int inputheight = height_;
    cv::Mat pre_mask = cv::Mat::zeros(height_, width_, CV_32F);//cv::Mat::zeros(inputheight, inputwidth, CV_8UC1);  //height  width
    cv::Mat frame, image0;
    cv::Mat backimg = cv::imread("0.jpg");
    cv::resize(backimg, backimg, cv::Size(inputwidth, inputheight));

    cv::VideoCapture capture(0);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);

    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    int index = 0;
   /// object_rect res_area;

    //需顺时针90°旋转时，transpose(src, tmp) + flip(tmp, dst, 1)

    //需逆时针90°旋转时，transpose(src, tmp) + flip(tmp, dst, 0)

    while (true)
    {
        index += 1;
        capture >> frame;

        //cv::transpose(frame, rotef);
        //cv::Mat rodst;
        //cv::rotate(frame, rodst, 0); //顺时针90
        //cv::rotate(frame, rodst, 2);   //顺时针270


         cv::Mat sizeFrame;
        double timeStart = (double)cv::getTickCount();
        //resize_uniform(frame, sizeFrame, cv::Size(inputwidth, inputheight), res_area);

        cv::resize(frame, sizeFrame, cv::Size(inputwidth, inputheight));  //width height
        cv::cvtColor(sizeFrame, image0, cv::COLOR_BGR2RGB);


        float* output = input_image_.data();

        // std::fill(input_image_.begin(), input_image_.end(), 0.f);  // 暂时去掉
        fill_data(image0, pre_mask, output, index);

        double midpredict = (double)cv::getTickCount();

        double postTime = (midpredict - timeStart) / cv::getTickFrequency();
        std::cout << "  running time post time ：" << postTime << "sec";

        model.Run();
        double stpreTime = (double)cv::getTickCount();
        double preTime = (stpreTime - midpredict) / cv::getTickFrequency();
        std::cout << "  predict time  :" << preTime << "sec";


        cv::Mat segm;
        const int inputMaskValue = 1;
        cv::threshold(results_, segm, 0.5, 1, cv::THRESH_BINARY);

        //////// contour
       // cv::Mat binary = segm * 255;

       // //cv::threshold(segm * 255, binary, 127, 255, cv::THRESH_BINARY);
       // cv::Mat ubinary;
       // binary.convertTo(ubinary, CV_8U);
       ///* cv::Canny(ubinary, ubinary, 60, 255,3); //可以修改canny替换所有
       // cv::imshow("canny", ubinary);*/
       // std::vector<std::vector<cv::Point>> contours;
       // std::vector<cv::Vec4i> hierarchy;

       // cv::findContours(ubinary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
       // cv::Mat imageContours = cv::Mat::zeros(ubinary.size(), CV_8UC1);
       // for (int i = 0; i < contours.size(); i++)
       // {
       //     cv::drawContours(imageContours, contours, i, cv::Scalar(255), 1, 8, hierarchy);
       // }

       // double preTime = ((double)cv::getTickCount() - midpredict) / cv::getTickFrequency();

        //std::cout << "predict time  :" << preTime << "sec\n" << std::endl;
        //cv::imshow("edge", imageContours);

        /// 


        segm.copyTo(pre_mask);
        pre_mask = 0.7 * pre_mask;


        cv::Mat ucharSegem;
        segm = segm * 255;
        segm.convertTo(ucharSegem, CV_8U);

        ///
         //图形学处理 平滑mask;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat dstmask;
        cv::morphologyEx(ucharSegem, dstmask, cv::MORPH_OPEN, kernel);
        //cv::imshow("mask1", dstmask);
        //高斯处理 边缘更平滑 效果更好看
        cv::GaussianBlur(dstmask, dstmask, cv::Size(3, 3), 0, 0);
        //cv::imshow("mask2", dstmask);
        cv::Mat dst = replace_and_blend(backimg, sizeFrame, dstmask);

        //Mat resultdst;
        //crop_effect_area(dst, resultdst, frame.size(), res_area);

        double lastTime = ((double)cv::getTickCount() - stpreTime) / cv::getTickFrequency();

        std::cout << "  last prcess time :" << lastTime << "sec\n" << std::endl;

        cv::imshow("mask", ucharSegem);
       // cv::imshow("resultdst", resultdst);
        cv::imshow("dst", dst);
        //cv::Mat dst = cv::addWeighted(img1, 0.7, img2, 0.3, 0);
        //cv::imshow("mask", segm);

        //crop_effect_area(Mat & uniform_scaled, Mat & dst, Size ori_size, object_rect effect_area);

        //cv::imshow("pre_image", frame);
        cv::waitKey(1);

    }

    return 0;
}


#endif
