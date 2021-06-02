#pragma once
// Created by ch on 2020/12/10.
//
#include <unistd.h>
#include <iostream>
#include <string>
#include "http_client.h"
//#include "opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include "util.hpp"
namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;
#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }
void
inferFromServer_arc(std::vector<cv::Mat> &srcImgList, int BATCH_SIZE, std::string model_name,
                    std::vector<std::vector<float>> &res) {
    if (srcImgList.size() < BATCH_SIZE) {
        BATCH_SIZE = srcImgList.size();
    }
    bool verbose = false;
    bool use_custom_model = false;
    std::string url("localhost:8000");
    nic::Headers http_headers;
    uint32_t client_timeout = 40;
//        std::string model_name = use_custom_model ? "arcface" : "arcface";
    std::string model_version = "";
    // Create a InferenceServerHttpClient instance to communicate with the
    // server using HTTP protocol.
    std::unique_ptr<nic::InferenceServerHttpClient> client;
    FAIL_IF_ERR(
            nic::InferenceServerHttpClient::Create(&client, url, verbose),
            "unable to create http client");
//        const int BATCH_SIZE = 2;
    nic::Error err;
    std::string model_metadata;
    err = client->ModelMetadata(&model_metadata, model_name, model_version, http_headers);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model metadata: " << err << std::endl;
    }
    std::string model_config;
    err = client->ModelConfig(
            &model_config, model_name, model_version, http_headers);
    if (!err.IsOk()) {
        std::cerr << "error: failed to get model config: " << err << std::endl;
    }
    nic::InferInput *input0;
    std::vector<int64_t> shape{BATCH_SIZE, 3, arcface::INPUT_H, arcface::INPUT_W};
    FAIL_IF_ERR(
            nic::InferInput::Create(&input0, "data", shape, "FP32"),
            "unable to get INPUT0");

    std::shared_ptr<nic::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    // prepare input data ---------------------------
    float data[BATCH_SIZE * 3 * arcface::INPUT_H * arcface::INPUT_W];


    cv::Mat blob = cv::dnn::blobFromImages(srcImgList, 0.0078125, cv::Size(arcface::INPUT_W, arcface::INPUT_H),
                                           cv::Scalar(127.5, 127.5, 127.5),
                                           true, false);
    FAIL_IF_ERR(
            input0_ptr->AppendRaw(
                    reinterpret_cast<uint8_t *>(blob.ptr<float>(0)),
                    sizeof(data)),
            "unable to set data for INPUT0");
    // Generate the outputs to be requested.
    nic::InferRequestedOutput *output0;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output0, "prob"),
            "unable to get 'OUTPUT0'");
    std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
    output0_ptr.reset(output0);
    // The inference settings. Will be using default for now.
    nic::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;
    std::vector<nic::InferInput *> inputs = {input0_ptr.get()};
    std::vector<const nic::InferRequestedOutput *> outputs = {output0_ptr.get()};
    nic::InferResult *results;
    FAIL_IF_ERR(
            client->Infer(&results, options, inputs, outputs, http_headers),
            "unable to run model");
    std::shared_ptr<nic::InferResult> results_ptr;
    results_ptr.reset(results);
    // Get pointers to the result returned...
    int32_t *output0_data;
    size_t output0_byte_size;
    FAIL_IF_ERR(
            results_ptr->RawData(
                    "prob", (const uint8_t **) &output0_data, &output0_byte_size),
            "unable to get result data for 'OUTPUT0'");

    for (int b = 0; b < BATCH_SIZE; b++) {
        float *prob = reinterpret_cast<float *>(&output0_data[b * arcface::OUTPUT_SIZE]);
        cv::Mat out(arcface::OUTPUT_SIZE, 1, CV_32FC1, prob);
        cv::Mat out_norm;
        //cv::normalize(out,out_norm,-1,1,)
        cv::normalize(out, out_norm);
        std::vector<float> array;
        if (out_norm.isContinuous()) {
            array.assign((float *) out_norm.datastart, (float *) out_norm.dataend);
        } else {
            for (int i = 0; i < out_norm.rows; ++i) {
                array.insert(array.end(), (float *) out_norm.ptr<uchar>(i), (float *) out_norm.ptr<uchar>(i) + out_norm.cols);
            }
        }
        res.push_back(array);
    }

}






