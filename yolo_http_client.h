
#include <unistd.h>
#include <iostream>
#include <string>
#include "include/http_client.h"
#include "opencv2/opencv.hpp"
#include "util.hpp"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

cv::Rect get_rect(cv::Mat &img, float *bbox) {
    int l, r, t, b;
    float r_w = 608 / (img.cols * 1.0);
    float r_h = 608 / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (608 - r_w * img.rows) / 2;
        b = bbox[3] - (608 - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (608 - r_h * img.cols) / 2;
        r = bbox[2] - (608 - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

void
inferFromServer_arc(std::vector<cv::Mat> &srcImgList, int BATCH_SIZE, std::string model_name
) {
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
    std::vector<int64_t> shape{BATCH_SIZE, 3, 608, 608};
    FAIL_IF_ERR(
            nic::InferInput::Create(&input0, "data", shape, "FP32"),
            "unable to get INPUT0");

    std::shared_ptr<nic::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    // prepare input data ---------------------------
    float data[BATCH_SIZE * 3 * 608 * 608];


    cv::Mat blob = cv::dnn::blobFromImages(srcImgList, 1.0 / 255.0, cv::Size(608, 608), cv::Scalar(0, 0, 0),
                                           true, false);
    FAIL_IF_ERR(
            input0_ptr->AppendRaw(
                    reinterpret_cast<uint8_t *>(blob.ptr<float>(0)),
                    sizeof(data)),
            "unable to set data for INPUT0");
    // Generate the outputs to be requested.

    std::vector<int64_t> shape0{BATCH_SIZE};
    nic::InferRequestedOutput *output0;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output0, "count"),
            "unable to get 'OUTPUT1'");
    std::shared_ptr<nic::InferRequestedOutput> output0_ptr;
    output0_ptr.reset(output0);

    nic::InferRequestedOutput *output1;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output1, "box"),
            "unable to get 'OUTPUT2'");
    std::shared_ptr<nic::InferRequestedOutput> output1_ptr;
    output1_ptr.reset(output1);


    nic::InferRequestedOutput *output2;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output2, "score"),
            "unable to get 'OUTPUT3'");
    std::shared_ptr<nic::InferRequestedOutput> output2_ptr;
    output2_ptr.reset(output2);


    nic::InferRequestedOutput *output3;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output3, "class"),
            "unable to get 'OUTPUT4'");
    std::shared_ptr<nic::InferRequestedOutput> output3_ptr;
    output3_ptr.reset(output3);

    nic::InferRequestedOutput *output4;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output4, "yolo_boxes"),
            "unable to get 'OUTPUT4'");
    std::shared_ptr<nic::InferRequestedOutput> output4_ptr;
    output4_ptr.reset(output4);


    nic::InferRequestedOutput *output5;
    FAIL_IF_ERR(
            nic::InferRequestedOutput::Create(&output5, "yolo_scores"),
            "unable to get 'OUTPUT4'");
    std::shared_ptr<nic::InferRequestedOutput> output5_ptr;
    output5_ptr.reset(output5);

    // The inference settings. Will be using default for now.
    nic::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;
    std::vector<nic::InferInput *> inputs = {input0_ptr.get()};
    std::vector<const nic::InferRequestedOutput *> outputs = {output0_ptr.get(), output1_ptr.get(), output2_ptr.get(),
                                                              output3_ptr.get()};
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
                    "count", (const uint8_t **) &output0_data, &output0_byte_size),
            "unable to get result data for 'OUTPUT0'");
    int32_t *output1_data;
    size_t output1_byte_size;
    FAIL_IF_ERR(
            results_ptr->RawData(
                    "box", (const uint8_t **) &output1_data, &output1_byte_size),
            "unable to get result data for 'OUTPUT0'");
    int32_t *output2_data;
    size_t output2_byte_size;
    FAIL_IF_ERR(
            results_ptr->RawData(
                    "score", (const uint8_t **) &output2_data, &output2_byte_size),
            "unable to get result data for 'OUTPUT0'");
    int32_t *output3_data;
    size_t output3_byte_size;
    FAIL_IF_ERR(
            results_ptr->RawData(
                    "class", (const uint8_t **) &output3_data, &output3_byte_size),
            "unable to get result data for 'OUTPUT0'");


    for (size_t i = 0; i < output0_byte_size / 4; ++i) {
        std::cout << *(output0_data + i) << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < output1_byte_size / 4; ++i) {
        std::cout << *((float *) output1_data + i) << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < output2_byte_size / 4; ++i) {
        std::cout << *((float *) output2_data + i) << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < output3_byte_size / 4; ++i) {
        std::cout << *((float *) output3_data + i) << " ";
    }
    std::cout << std::endl;

    std::cout << "detect count " << output0_data[0] << std::endl;

    cv::Mat img = srcImgList[0];
    for (int j = 0; j < output0_data[0]; j++) {
        float *curBbox = reinterpret_cast<float *>(&output1_data[(0 * 100 + j) * 4]);

        float *curScore = reinterpret_cast<float *>(&output2_data[0 * 100 + j]);
        float *curClass = reinterpret_cast<float *>(&output3_data[0 * 100 + j]);
        cv::Rect r = get_rect(img, curBbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(img, std::to_string(int(*curClass)), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imshow("res", img);
    cv::waitKey(0);
}