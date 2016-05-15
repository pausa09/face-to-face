#include "sicaam.h"

using namespace cv;

#define fl at<float>

SICAAM::SICAAM()
{
}

void SICAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    //AAM::calcSteepestDescentImages();

    this->initialized = true;
}

float SICAAM::fit() {
    AAM::calcErrorImage();
    AAM::calcSteepestDescentImages();

    Mat SD_sim;
    vconcat(this->steepestDescentImages, this->A, SD_sim);

    Mat Hessian_sim = SD_sim*SD_sim.t();

    Mat deltaq = -Hessian_sim.inv()*SD_sim*this->errorImage.t();

    int numP = this->s.rows+this->s_star.rows;
    int numLambda = this->lambda.rows;
    Mat deltap = deltaq(cv::Rect(0,0,1,numP));
    Mat deltaLambda = deltaq(cv::Rect(0,numP,1,numLambda));

    AAM::updateAppearanceParameters(deltaLambda);
    AAM::updateInverseWarp(deltap);

    return sum(abs(deltaq))[0]/deltaq.rows;
}
