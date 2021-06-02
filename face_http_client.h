//
// Created by ch on 2020/12/10.
//

#ifndef TRION_CLIENT_FACE_HTTP_CLIENT_H
#define TRION_CLIENT_FACE_HTTP_CLIENT_H

//
// Created by ch on 2020/12/10.
//

#include <unistd.h>
#include <iostream>
#include <string>
#include "arcface_http_client.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;
#define NMS_THRESH 0.4

void inferFromServer(std::vector<cv::Mat> srcImgList, const int BATCH_SIZE, std::string model_name,
                     std::vector<cv::Mat> &face_list) {

    static const int INPUT_H = decodeplugin::INPUT_H;
    static const int INPUT_W = decodeplugin::INPUT_W;
    bool verbose = false;
    std::string url("localhost:8000");
    nic::Headers http_headers;
    uint32_t client_timeout = 40;
//    std::string model_name = use_custom_model ? "retinaface" : "retinaface";
    std::string model_version = "";
    // Create a InferenceServerHttpClient instance to communicate with the
    // server using HTTP protocol.
    std::unique_ptr<nic::InferenceServerHttpClient> client;
    FAIL_IF_ERR(
            nic::InferenceServerHttpClient::Create(&client, url, verbose),
            "unable to create http client");
//    nic::Error err;
//    std::string model_metadata;
//    err = client->ModelMetadata(&model_metadata, model_name, model_version, http_headers);
//    if (!err.IsOk()) {
//        std::cerr << "error: failed to get model metadata: " << err << std::endl;
//    }
//    std::string model_config;
//    err = client->ModelConfig(
//            &model_config, model_name, model_version, http_headers);
//    if (!err.IsOk()) {
//        std::cerr << "error: failed to get model config: " << err << std::endl;
//    }

    nic::InferInput *input0;
    std::vector<int64_t> shape{BATCH_SIZE, 3, INPUT_H, INPUT_W};
    FAIL_IF_ERR(
            nic::InferInput::Create(&input0, "data", shape, "FP32"),
            "unable to get INPUT0");
    std::shared_ptr<nic::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    // prepare input data -------------------------
    float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    try {
//        std::vector<cv::Mat> srcImgLists;
////        srcImgLists.assign(srcImgList.begin(),srcImgList.end());;
////        std::copy(srcImgLists.begin(),srcImgLists.end(),  std::back_inserter(srcImgList));
////        memcpy(&srcImgLists, &srcImgList, sizeof(cv::Mat) * srcImgList.size());
        cv::Mat blob;
//        vector<cv::Mat>::iterator it;//声明迭代器
//        for (it = srcImgList.begin(); it != srcImgList.end(); ++it) {//遍历v2,赋值给v1
//            cv::Mat temp = it->clone();
//            srcImgLists.push_back(temp);
//        }

        //cv::imwrite("srcImgList_origin.jpg", srcImgList[0]);
        cv::dnn::blobFromImages(srcImgList, blob, 1.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(104, 117, 123),
                                false, false);


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

        double transD[5][2] = {
                {38.2946, 51.6963},
                {73.5318, 51.5014},
                {56.0252, 71.7366},
                {41.5493, 92.3655},
                {70.7299, 92.2041}
        };
        cv::Mat src(5, 2, CV_64F, transD);

        std::vector<std::vector<decodeplugin::Detection>> batch_res(BATCH_SIZE);
        for (int b = 0; b < BATCH_SIZE; b++) {
            auto &res = batch_res[b];
            decodeplugin::nms(res, reinterpret_cast<float *>(&output0_data[b * decodeplugin::OUTPUT_SIZE]), NMS_THRESH);
            std::vector<cv::Mat> arclist;
            cv::Mat img = srcImgList[b];
            cv::Mat landmarkFace;
            for (size_t j = 0; j < res.size(); j++) {
                std::vector<cv::Point2f> landmarkPoints;
                cv::Rect r = decodeplugin::get_rect_adapt_landmark(img, INPUT_W, INPUT_H, res[j].bbox, res[j].landmark);
                landmarkFace = img(r).clone();
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);

                for (int k = 0; k < 10; k += 2) {
                    cv::Point2f lank;
                    lank.x = res[j].landmark[k] - r.x;
                    lank.y = res[j].landmark[k + 1] - r.y;
                    cv::circle(img, cv::Point(res[j].landmark[k], res[j].landmark[k + 1]), 1,
                               cv::Scalar(255 * (k > 2), 255 * (k > 0 && k < 8), 255 * (k < 6)), 4);
                    landmarkPoints.push_back(lank);
                }

//                cv::imshow("res",img);
//                cv::waitKey(0);
                double point5[5][2];
                int i = 0;
                for (auto point : landmarkPoints) {
                    point5[i][0] = point.x;
                    point5[i][1] = point.y;
                    i += 1;
                }
                cv::Mat dst(5, 2, CV_64F, point5);
                cv::Mat M;
                cv::estimateAffinePartial2D(dst, src).copyTo(M);
                cv::Mat warped;
                cv::warpAffine(landmarkFace, warped, M, {arcface::INPUT_W, arcface::INPUT_H});
//                cv::imshow("res",warped);
//                cv::waitKey(0);
                arclist.push_back(warped);

//                cv::imwrite("test.jpg",warped);
            }
            face_list.push_back(img);
        }
    }
    catch (cv::Exception e) {
        std::cout << "msg" << e.what() << std::endl;
    }


}


#endif //TRION_CLIENT_YOLO_HTTP_CLIENT_H
