//
// from https://blog.csdn.net/icvpr/article/details/10259577
//

#include <iostream>
#include <string>
#include <list>
#include <vector>
#include "utils/common.h"

#include <opencv2/opencv.hpp>
 namespace LaneDetect {

    static void CcaByTwoPass(const cv::Mat &_binImg, cv::Mat &_lableImg, std::set<int>& labels, int col_neighbor=10, int row_neighbor =10) {
        // connected component analysis (4-component)
        // use two-pass algorithm
        // 1. first pass: label each foreground pixel with a label
        // 2. second pass: visit each labeled pixel and merge neighbor labels
        //
        // foreground pixel: _binImg(x,y) >= 1
        // background pixel: _binImg(x,y) = 0

//        int col_neighbor = 10;
//        int row_neighbor = 10;

        if (_binImg.empty() ||
            _binImg.type() != CV_8UC1) {
            return;
        }

        labels.clear();

        // 1. first pass

        cv::Mat bwimg;
        cv::threshold(_binImg, bwimg, 0, 1, cv::THRESH_BINARY);
//        cv::imshow("_lableImg_bw", bwimg);
//        cv::waitKey();

        _lableImg.release();
        bwimg.convertTo(_lableImg, CV_32SC1);


        int label = 1;  // start by 2
        std::vector<int> labelSet;
        labelSet.push_back(0);   // background: 0
        labelSet.push_back(1);   // foreground: 1

        int rows = _binImg.rows - 1;
        int cols = _binImg.cols - 1;
        for (int i = 1; i < rows; i++) {
            int *data_preRow = _lableImg.ptr<int>(i - 1);
            int *data_curRow = _lableImg.ptr<int>(i);
            for (int j = 1; j < cols; j++) {
                if (data_curRow[j] == 1) {
                    std::vector<int> neighborLabels;

                    /////
                    for (int in_row = (i - row_neighbor > 0) ? i - row_neighbor : 0; in_row <=i; ++in_row) {
                        for (int in_col = (j - col_neighbor > 0) ? j - col_neighbor : 0; in_col <=j; ++in_col) {
                            int *data_localRow = _lableImg.ptr<int>(in_row);
                            if (data_localRow[in_col] > 1) {
                                neighborLabels.emplace_back(data_localRow[in_col]);
                            }
                        }
                    }

                    //// find neighbor element
//                    int leftPixel = data_curRow[j - 1];
//                    int upPixel = data_preRow[j];
//
//                    if (leftPixel > 1) {
//                        neighborLabels.push_back(leftPixel);
//                    }
//                    if (upPixel > 1) {
//                        neighborLabels.push_back(upPixel);
//                    }

                    if (neighborLabels.empty()) {
                        labelSet.push_back(++label);  // assign to a new label
                        data_curRow[j] = label; // label_image  assign label
                        labelSet[label] = label;
                    } else {

                        std::sort(neighborLabels.begin(), neighborLabels.end());
                        int smallestLabel = neighborLabels[0];
                        data_curRow[j] = smallestLabel;

                        // save equivalence
                        for (size_t k = 1; k < neighborLabels.size(); k++) {
                            int tempLabel = neighborLabels[k];
                            int &oldSmallestLabel = labelSet[tempLabel];
                            if (oldSmallestLabel > smallestLabel) {
                                labelSet[oldSmallestLabel] = smallestLabel;
                                oldSmallestLabel = smallestLabel;
                            } else if (oldSmallestLabel < smallestLabel) {
                                labelSet[smallestLabel] = oldSmallestLabel;
                            }
                        }
                    }
                }
            }
        }

        // update equivalent labels
        // assigned with the smallest label in each equivalent label set
        for (size_t i = 2; i < labelSet.size(); i++) {
            int curLabel = labelSet[i];
            int preLabel = labelSet[curLabel];
            while (preLabel != curLabel) {
                curLabel = preLabel;
                preLabel = labelSet[preLabel];
            }
            labelSet[i] = curLabel;
        }


        // 2. second pass
        for (int i = 0; i < rows; i++) {
            int *data = _lableImg.ptr<int>(i);
            for (int j = 0; j < cols; j++) {
                int &pixelLabel = data[j];
                pixelLabel = labelSet[pixelLabel];
                labels.insert(pixelLabel);
            }
        }
    }
}// namespace