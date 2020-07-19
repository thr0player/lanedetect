//
//

#include <iostream>
#include "SVMClassifier.h"

#define TRAIN_MODE
#define PREDICT_MODE

int  main() {
    LaneDetect::SVMClassifier classifier;

#ifdef TRAIN_MODE
    classifier.LoadData();
    classifier.Train();
#endif
#ifdef PREDICT_MODE
    cv::Mat img = cv::imread("/home/Data/hog/test.png", CV_LOAD_IMAGE_GRAYSCALE);
//    classifier.Detect(img);
    classifier.DetectCluster(img);
#endif
    return 0;
}