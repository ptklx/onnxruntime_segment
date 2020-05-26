
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include "segmentation_onnx.h"
//#include <onnxruntime_cxx_inline.h>
#include <cuda_provider_factory.h>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

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
        /*
        // DirectML
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(m_sess_opts, 0));
        m_sess_opts.DisableMemPattern();
        m_sess_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        m_session = Ort::Session(m_env, model_path, m_sess_opts);
        */

        
        // CPU
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(m_sess_opts, 0));
        m_sess_opts.EnableMemPattern();
        m_sess_opts.SetIntraOpNumThreads(8);
        m_sess_opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        m_sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        m_session = Ort::Session(m_env, model_path, m_sess_opts);
        

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
    cv::Mat pre_mask = cv::Mat::zeros(inputheight, inputwidth, CV_8UC1);  //height  width
    cv::Mat frame, image0;


    cv::VideoCapture capture(0);
    int index = 0;
    while (true)
    {
        index += 1;
        capture >> frame;
        cv::resize(frame, frame, cv::Size(inputwidth, inputheight));  //width height
        cv::cvtColor(frame, image0, cv::COLOR_BGR2RGB);

        double timeStart = (double)cv::getTickCount();
        float* output = input_image_.data();

        std::fill(input_image_.begin(), input_image_.end(), 0.f);  //
        fill_data(image0, pre_mask, output, index);

        model.Run();

        double nTime = ((double)cv::getTickCount() - timeStart) / cv::getTickFrequency();
        std::cout << "running time £º" << nTime << "sec\n" << std::endl;

        cv::Mat segm;

        cv::threshold(results_, segm, 0.5, 200, cv::THRESH_BINARY);
        segm.copyTo(pre_mask);
        cv::Mat ucharSegem;
        segm.convertTo(ucharSegem, CV_8U);

        //cv::imshow("mask", segm);
        cv::imshow("mask", ucharSegem);
        cv::imshow("pre_image", frame);
        cv::waitKey(1);

    }

    return 0;
}


#endif