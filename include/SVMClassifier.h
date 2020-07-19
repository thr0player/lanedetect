//
//

#ifndef LANEDETECTION_SVMCLASSIFIER_H
#define LANEDETECTION_SVMCLASSIFIER_H

#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

namespace LaneDetect{
    using namespace cv;
    using namespace cv::ml;
    struct ClassSamples{
        int label_;
        std::vector<cv::Mat> frames_;
    };

    struct DetectClass {
        int label_;
        cv::Rect rect_;
        Point2f minpt, maxpt;
        DetectClass(){
            minpt.x = minpt.y = std::numeric_limits<float>::max();
            maxpt.x = maxpt.y = std::numeric_limits<float>::min();
        };
        void GetRect(){
            rect_.x = minpt.x;
            rect_.y = minpt.y;
            rect_.width = maxpt.x - minpt.x;
            rect_.height = maxpt.y - minpt.y;
        }
    };

    class SVMClassifier  {
    public:
        SVMClassifier();

        ~SVMClassifier() {};

        int LoadData();

        int Train();

        int DetectMultiscale(const cv::Mat &test_img, std::vector<DetectClass> &detects);

        int DetectCluster(const cv::Mat &test_img);

        int ModelInit(std::string filename);

        int Detect(const cv::Mat &img);

    public:
        std::string model_path_;
        std::vector<ClassSamples> dataset_;

        Ptr<SVM> svm_;
        HOGDescriptor myHOG_;
        bool init_model_;

    };

}// namespace
#endif //LANEDETECTION_SVMCLASSIFIER_H
