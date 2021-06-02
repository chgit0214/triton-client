////
//// Created by ch on 2020/12/10.
////


#include "yolo_http_client.h"
#include "util.hpp"
#include "face_http_client.h"
#include "arcface_http_client.h"
#include "annoylib.h"
#include "kissrandom.h"
void floatArr2Str(float arr[512], char str[5120]) {
    char tempStr[16];
    memset(str, 0, sizeof(char) * 5120);
    for (int k = 0; k < 512; k++) {
        sprintf(tempStr, "%.7f", arr[k]);
        strcat(str, tempStr);
        if (k != 511)
            strcat(str, ",");
    }
    std::cout << "str: " << str << std::endl;
}

void Str2floatArr(char str[5120], float arr[512]) {
    char tempStr[16];
    memset(str, 0, sizeof(char) * 5120);
    for (int k = 0; k < 512; k++) {
        sprintf(tempStr, "%.7f", arr[k]);
        strcat(str, tempStr);
        if (k != 511)
            strcat(str, ",");
    }
    std::cout << "str: " << str << std::endl;
}

float calcSimilar(std::vector<float> feature1, std::vector<float> feature2) {
    //assert(feature1.size() == feature2.size());
    float sim = 0.0;
    for (int i = 0; i < feature1.size(); i++)
        sim += feature1[i] * feature2[i];
    return sim;
}
void normalize(std::vector<float> &feature)
{
    float sum = 0;
    for (auto it = feature.begin(); it != feature.end(); it++)
        sum += (float)*it * (float)*it;
    sum = sqrt(sum);
    for (auto it = feature.begin(); it != feature.end(); it++)
        *it /= sum;
}
int main() {
    char imgPath[2][255] = {"../roi344copy.jpg",
                            "../chenhao.jpg"};
    std::vector<cv::Mat> srcImgList;
    for (int i = 0; i < 1; i++) {
        cv::Mat tempImg = cv::imread(imgPath[i]);
        cv::Mat pr_img = decodeplugin::preprocess_img(tempImg, decodeplugin::INPUT_W, decodeplugin::INPUT_H);
        srcImgList.push_back(pr_img);
    }
//    inferFromServer_arc(srcImgList,1,"yolov5");
    std::vector<std::vector<cv::Mat>> face_list;
    face_list.clear();
    inferFromServer(srcImgList, 1, "retinaface", face_list);
    std::cout << face_list.size() << std::endl;
//    std::vector<std::vector<float>> res;
//    std::vector<std::vector<float>> res2;
//    inferFromServer_arc(face_list[0], 2, "arcface", res);
//    inferFromServer_arc(face_list[1], 2, "arcface", res2);
////    normalize(res[0]);
////    normalize(res2[0]);
//
//    AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy> t = AnnoyIndex<int, float, Angular, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy>(
//            512);
//    t.load("/home/ch/CLionProjects/cppserver/config/face_vector.tree");
//    float selarray[512];
//    std::copy(res[0].begin(), res[0].end(), selarray);
//    vector<int> resindex;
//    vector<float> resdis;
//    t.get_nns_by_vector(selarray, 2, 100, &resindex, &resdis);
//    float ss=calcSimilar(res[0], res2[0]);

//    return 0;
}
