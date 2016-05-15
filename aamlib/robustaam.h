#ifndef ROBUSTAAM_H
#define ROBUSTAAM_H

#include <opencv2/opencv.hpp>
#include "aam.h"

using namespace std;

class RobustAAM: public AAM
{
private:
    void calcTriangleHessians();
    cv::Mat calcAppeanceUpdate();
    cv::Mat calcShapeUpdate();
    cv::Mat calcWeightedHessian(vector<cv::Mat> triangleHessians);

    cv::Mat applyRobustErrorFunction(cv::Mat img);

    vector<cv::Mat> triangleShapeHessians;
    vector<cv::Mat> triangleAppHessians;

public:
    RobustAAM();

    void train();
    float fit();
};

#endif // ROBUSTAAM_H
