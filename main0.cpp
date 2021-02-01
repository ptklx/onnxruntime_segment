#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include "segmentation_onnx.h"
#include <cuda_provider_factory.h>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <chrono>


#include "yolov3_post_process.h"
#include "parar_config.h"
#include "count_egg_arthmetic.h"


void fill_data(cv::Mat input_img, std::vector<float>& output)
{
   // float mean[3] = { 123.675, 116.28, 103.53 };
    //float std[3] = { 0.0171, 0.0175, 0.0174 };
    //float allmul[3] = { 2.1145,2.0349,1.8014 };
   // float scale = 1;
    //float* dst_pre = pre_mask.ptr<float>(0, 0);
    int row = input_img.rows;
    int col = input_img.cols;
    //int alllen = row * col;
    //double timeStart = (double)cv::getTickCount();
    //HWC -> CHW
    //float* inbuf = output;
    int n = 0;
    for (int c = 0; c < 3; c++) {
        uchar* img_data = input_img.data;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                // *inbuf = img_data[c] * std[c] - allmul[c];//(dst_img.ptr<float>(i, j)[c]);
                // img_data += 3;
                //inbuf++;
                output[n] = img_data[c] / 255.0;
                img_data += 3;
                n++;
            }
        }
    }

    //double circle = ((double)cv::getTickCount() - timeStart) / cv::getTickFrequency();
    //std::cout << "circle time  ：" << circle << " sec";
    //memcpy(inbuf, dst_pre, sizeof(float) * alllen);

    return;
}



bool comp(const std::string& a, const std::string& b) {
    std::string a_sub = a.substr(a.rfind("\\")+1);  //倒序发现\\第一次出现的子串
    std::string b_sub = b.substr(b.rfind("\\")+1);
    std::string erse_s = "_0.jpg";
    int outa = std::stoi((a_sub.substr(0,a_sub.find(erse_s))).c_str());
    int outb = std::stoi((b_sub.substr(0,b_sub.find(erse_s))).c_str());
    return outa < outb;
    
}

int main(int argc, char* argv[]) {
    //*************************************************************************
    // initialize  enviroment...one enviroment per process
    // enviroment maintains thread pools and other state info
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
    // session (we also need to include cuda_provider_factory.h above which defines it)
    // #include "cuda_provider_factory.h"
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //*************************************************************************
    // create session and load model into memory
#ifdef _WIN32
    const wchar_t* model_path = L"../onnx_model/epoch200.onnx";
#else
    const char* model_path = "squeezenet.onnx";
#endif

    printf("Using Onnxruntime C++ API\n");
    Ort::Session session(env, model_path, session_options);

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

    printf("Number of inputs = %zu\n", num_input_nodes);

    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++) {
        // print input node names
        char* input_name = session.GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        //allocator.Free(input_name);
    }


    // Results should be...
    // Number of inputs = 1
    // Input 0 : name = data_0
    // Input 0 : type = 1
    // Input 0 : num_dims = 4
    // Input 0 : dim 0 = 1
    // Input 0 : dim 1 = 3
    // Input 0 : dim 2 = 224
    // Input 0 : dim 3 = 224

    //*************************************************************************
    // Similar operations to get output node information.
    // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
    // OrtSessionGetOutputTypeInfo() as shown above.

    //*************************************************************************
    // Score the model using sample data, and inspect values

    size_t input_tensor_size = 640 * 384 * 3;  // simplify ... using known dim values to calculate size
                                               // use OrtGetTensorShapeElementCount() to get official size!

    std::vector<float> input_tensor_values(input_tensor_size);
    //std::vector<const char*> output_node_names = { "203" };
   
  
    size_t m_numOutputs = session.GetOutputCount();

    std::vector<const char*> m_outputNodeNames(m_numOutputs);

    std::vector<int64_t> m_outputTensorSizes(m_numOutputs);
    std::vector<std::vector<int64_t>> m_outputShapes(m_numOutputs);
    Ort::AllocatorWithDefaultOptions m_ortAllocator;

    for (int i = 0; i < m_numOutputs; ++i) {
        char* outputName = session.GetOutputName(i, m_ortAllocator);
        printf("output %d : name=%s\n", i, outputName);
        m_outputNodeNames[i] = outputName;

        Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    
        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        printf("output %d : type=%d\n", i, type);

        // print input shapes/dims
        m_outputShapes[i] = tensorInfo.GetShape();
        printf("output %d : num_dims=%zu\n", i, m_outputShapes[i].size());
        for (int j = 0; j < m_outputShapes[i].size(); j++)
            printf("output %d : dim %d=%jd\n", i, j, m_outputShapes[i][j]);
        //m_ortAllocator.Free(outputName);
    }


    int inputwidth = 640;
    int inputheight = 384;

    cv::Mat  image0;// (inputwidth, inputheight, CV_8UC3);
    //cv::VideoCapture capture(0);
    int index = 0;

    init_egg_para();

    //std::string path = "Z:\\data\\train_egg\\egg1m\\testdata\\rktest0\\";
    std::string path = "Z:\\data\\train_egg\\egg1m\\testdata\\first_400\\*.jpg";  //path of folder, you can replace "*.*" by "*.jpg" or "*.png"

    std::vector<cv::String> file_names;
    cv::glob(path, file_names);   //
    sort(file_names.begin(), file_names.end(), comp);
    //while (true)
    //{
    for (int nb = 0; nb < file_names.size(); nb++) {
        
        //capture >> frame;
        //std::string namepath = path + std::to_string(index) + ".jpg";
        std::cout << file_names[nb] << std::endl;
        //index += 1;
        index = nb;

       /* if (index > 154)
        {
            index = 0;
            std::cout << "pic read over !!!";

        }*/
        cv::Mat frame = cv::imread(file_names[nb]);

        cv::resize(frame, frame, cv::Size(inputwidth, inputheight));  //width height
       // cv::cvtColor(frame, image0, cv::COLOR_BGR2RGB);
        cv::cvtColor(frame, image0, CV_BGR2RGB);


        fill_data(image0, input_tensor_values);
        // initialize input data with values in [0.0, 1.0]
        //for (unsigned int i = 0; i < input_tensor_size; i++)
           // input_tensor_values[i] = (float)i / (input_tensor_size + 1);



        // create input tensor object from data values
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
        assert(input_tensor.IsTensor());




        // score model & input tensor, get back output tensor
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, m_outputNodeNames.data(), m_numOutputs);
        assert(output_tensors.size() == m_numOutputs && output_tensors.front().IsTensor());

     
        //std::vector<std::vector<float>> outputData;
        //outputData.reserve(m_numOutputs);

        //int count = 1;
        //for (auto& elem : output_tensors) {
        //    std::cout<<("type of input %d: %s", count++, std::to_string(elem.GetTensorTypeAndShapeInfo().GetElementType()).c_str());
        //   /* outputData.emplace_back(
        //        std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));*/

        //}

        //struct ssd_group* group;
        struct ssd_group g_ssd_group[1];

        int w = inputwidth;
        int h = inputheight;



         //for (int j=0; j<5; j++) {
         //  for (int i=0; i<6; i++) {
         //      printf("_%f",outputData[0][i+j*6]);
         //    } // end of row   
         //  printf("\n");
         //}  
        if (output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount() != LISTSIZE * SPAN * GRIDW0 * GRIDH0)
        {
            std::cout<<"erro!";
        }

        if (output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount() != LISTSIZE * SPAN * GRIDW1 * GRIDH1)
        {
            std::cout << "erro!";
        }
   

        float tempbuffer0 [LISTSIZE * SPAN * GRIDW0 * GRIDH0 ];
        float tempbuffer1 [LISTSIZE * SPAN * GRIDW1 * GRIDH1 ];

        int tempn0 = 0;
        int tempn1 = 0;
        float* out0 = output_tensors[0].GetTensorMutableData<float>();
        float* out1 = output_tensors[1].GetTensorMutableData<float>();
        for (int i = 0; i < LISTSIZE * SPAN; i++) {
            for (int j = 0; j < GRIDW0 * GRIDH0; j++)
            {
          
                tempbuffer0[tempn0] = out0[tempn0];
                tempn0++;
            }
           
            for (int j = 0; j < GRIDW1 * GRIDH1; j++)
            {

                tempbuffer1[tempn1] = out1[tempn1];
                tempn1++;
            }
             
        }

        yolov_post_process_egg((float*)tempbuffer0, (float*)tempbuffer1, w, h, &g_ssd_group[0]);
        for (int k = 0; k < g_ssd_group[0].count; k++)
        {   

             cv::putText(frame,std::to_string(g_ssd_group[0].objects[k].select.prop), cv::Point(g_ssd_group[0].objects[k].select.left, g_ssd_group[0].objects[k].select.top),\
                 cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255),1);

             //cv::rectangle(frame, cv::Rect(g_ssd_group[0].objects[k].select.left, g_ssd_group[0].objects[k].select.top, w,h), cv::Scalar(255, 0, 0), 1, 1, 0);
             cv::rectangle(frame, cv::Point(g_ssd_group[0].objects[k].select.left, g_ssd_group[0].objects[k].select.top), cv::Point(g_ssd_group[0].objects[k].select.right, \
                 g_ssd_group[0].objects[k].select.bottom), cv::Scalar(255, 0, 0), 1, 1, 0);
        }
        

        int eggcount =  compute_egg( w, h, index, g_ssd_group);


        int entranceline = 0.5*h - 60;
        int  outline = 0.5 * h + 60;

 
        cv::line(frame, cv::Point(0, entranceline) , cv::Point(640, entranceline), cv::Scalar(0, 0, 255), 2, 4);

        cv::line(frame, cv::Point(0, outline), cv::Point(640, outline), cv::Scalar(0, 0, 255), 2, 4);

        cv::putText(frame, "eggcount_" + std::to_string(eggcount), cv::Point(4, 384), \
            cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255,255,0), 2);
        cv::imshow("matImage", frame);
     
        cv::waitKey(1);

    }
    printf("Done!\n");
    return 0;
}