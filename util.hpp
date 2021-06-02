#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdio.h>
#include <iostream>
#include <string.h>
#include "rapidjson/prettywriter.h"
#include <rapidjson/document.h>
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "opencv2/opencv.hpp"
using namespace rapidjson;
using namespace std;

static std::string  pDBIP;
static unsigned short usDBPort;
static std::string pDBName;
static std::string  pUserName;
static std::string pPassword;
static std::string Deepstream_path;
static std::string  face_config;
static std::string  yolov5_path;
static std::string  yolov5_PRELOAD;
static std::string  savepic_path;
static unsigned short ngix_port;
static std::string index_path;
static unsigned short num_nn_nearst;
static unsigned short num_trees;
static unsigned short face_vector;




#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }
namespace decodeplugin {
    struct alignas(float) Detection {
        float bbox[4];  //x1 y1 x2 y2
        float class_confidence;
        float landmark[10];
    };
    static const int INPUT_H = 480;
    static const int INPUT_W = 640;
    static const int OUTPUT_SIZE =
            (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 + INPUT_H / 32 * INPUT_W / 32) * 2 * 15 + 1;

    static inline cv::Mat preprocess_img(cv::Mat &img, int input_w, int input_h) {
        int w, h, x, y;
        float r_w = input_w / (img.cols * 1.0);
        float r_h = input_h / (img.rows * 1.0);
        if (r_h > r_w) {
            w = input_w;
            h = r_w * img.rows;
            x = 0;
            y = (input_h - h) / 2;
        } else {
            w = r_h * img.cols;
            h = input_h;
            x = (input_w - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, img.type());
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(input_h, input_w, img.type(), cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        return out;
    }

    static cv::Rect get_rect_adapt_landmark(cv::Mat &img, int input_w, int input_h, float bbox[4], float lmk[10]) {
        int l, r, t, b;
        float r_w = input_w / (img.cols * 1.0);
        float r_h = input_h / (img.rows * 1.0);
        if (r_h > r_w) {
            l = bbox[0] / r_w;
            r = bbox[2] / r_w;
            t = (bbox[1] - (input_h - r_w * img.rows) / 2) / r_w;
            b = (bbox[3] - (input_h - r_w * img.rows) / 2) / r_w;
            for (int i = 0; i < 10; i += 2) {
                lmk[i] /= r_w;
                lmk[i + 1] = (lmk[i + 1] - (input_h - r_w * img.rows) / 2) / r_w;
            }
        } else {
            l = (bbox[0] - (input_w - r_h * img.cols) / 2) / r_h;
            r = (bbox[2] - (input_w - r_h * img.cols) / 2) / r_h;
            t = bbox[1] / r_h;
            b = bbox[3] / r_h;
            for (int i = 0; i < 10; i += 2) {
                lmk[i] = (lmk[i] - (input_w - r_h * img.cols) / 2) / r_h;
                lmk[i + 1] /= r_h;
            }
        }
        return cv::Rect(l, t, r - l, b - t);
    }

    static float iou(float lbox[4], float rbox[4]) {
        float interBox[] = {
                std::max(lbox[0], rbox[0]), //left
                std::min(lbox[2], rbox[2]), //right
                std::max(lbox[1], rbox[1]), //top
                std::min(lbox[3], rbox[3]), //bottom
        };

        if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
        return interBoxS /
               ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS +
                0.000001f);
    }

    static bool cmp(const decodeplugin::Detection &a, const decodeplugin::Detection &b) {
        return a.class_confidence > b.class_confidence;
    }

    static inline void nms(std::vector<decodeplugin::Detection> &res, float *output, float nms_thresh = 0.4) {
        std::vector<decodeplugin::Detection> dets;
        for (int i = 0; i < output[0]; i++) {
            if (output[15 * i + 1 + 4] <= 0.1) continue;
            decodeplugin::Detection det;
            memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
            dets.push_back(det);
        }
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto &item = dets[m];
            res.push_back(item);
            //std::cout << item.class_confidence << " bbox " << item.bbox[0] << ", " << item.bbox[1] << ", " << item.bbox[2] << ", " << item.bbox[3] << std::endl;
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

namespace arcface {
    static const int INPUT_H = 112;
    static const int INPUT_W = 112;
    static const int OUTPUT_SIZE = 512;
    cv::Mat preprocess_img(cv::Mat &img, int input_w, int input_h) {
        int w, h, x, y;
        float r_w = input_w / (img.cols * 1.0);
        float r_h = input_h / (img.rows * 1.0);
        if (r_h > r_w) {
            w = input_w;
            h = r_w * img.rows;
            x = 0;
            y = (input_h - h) / 2;
        } else {
            w = r_h * img.cols;
            h = input_h;
            x = (input_w - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        return out;
    }


}

void gettime(char *timeStr) {
    time_t t = time(0);
    strftime(timeStr, 32, "%Y-%m-%d-%H-%M-%S", localtime(&t));

}

struct box_data {
    char *config_path;
    char *config_lib;
    char *Deepstream_path;
};



void str2floatArr(char str[6000], float arr[512])
{
    int count = 0;
    int offset = 0, readCharCount;
    while (sscanf(str + offset, "%f,%n", arr+count, &readCharCount) == 1) {
        //printf("%f, %d\n", arr[count], readCharCount);
        offset += readCharCount;
        count++;
    }
    if (count != 512)
        printf("error : count != 512 nowCcount=%d\n",count);
}

void floatArr2Str(std::vector<float>& array,char str[6000])
{
    char tempStr[16];
    memset(str,0,sizeof(char)*6000);
    for(int k = 0;k<array.size();k++)
    {
        sprintf(tempStr,"%.7f",array[k]);
        strcat(str,tempStr);
        //if(k!=511)
        strcat(str,",");
    }
    std::cout<<"str: "<<str<<std::endl;
    std::cout<<"strlen: "<<strlen(str)<<std::endl;

}

#endif