#include "wsicaam.h"

using namespace cv;
#define fl at<float>

WSICAAM::WSICAAM()
{
}

void WSICAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    //AAM::calcSteepestDescentImages();

    this->initialized = true;
}

float WSICAAM::fit() {
    AAM::calcErrorImage();
    AAM::calcSteepestDescentImages();

    Mat SD_sim;
    vconcat(this->steepestDescentImages, this->A, SD_sim);

    Mat weights = this->calcWeights();
    Mat weights2 = repeat(weights, SD_sim.rows, 1);

    Mat Hessian_sim = weights2.mul(SD_sim)*SD_sim.t();

    Mat deltaq = -Hessian_sim.inv()*weights2.mul(SD_sim)*this->errorImage.t();

    int numP = this->s.rows+this->s_star.rows;
    int numLambda = this->lambda.rows;
    Mat deltap = deltaq(cv::Rect(0,0,1,numP));
    Mat deltaLambda = deltaq(cv::Rect(0,numP,1,numLambda));

    AAM::updateAppearanceParameters(deltaLambda);
    AAM::updateInverseWarp(deltap);

    this->steps++;
    std::cout<<"Step: "<<this->steps<<std::endl;
    return sum(abs(deltaq))[0]/deltaq.rows;
}

Mat WSICAAM::calcWeights() {
    Mat img = this->errorImage.clone();
    Mat i = img.clone();

    double median = 0;
    Mat Input = abs(img.reshape(0,1)); // spread Input Mat to single row
    std::vector<double> vecFromMat;
    Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
    std::sort( vecFromMat.begin(), vecFromMat.end() ); // sort vecFromMat

    int start=0;
    for(unsigned int val=0; val<vecFromMat.size(); val++) {
        if(vecFromMat[val] != 0) {
            start = val;
            break;
        }
    }
	std::cout << "Start: " << start << std::endl;

    if (vecFromMat.size()%2==0) {
        median = (vecFromMat[vecFromMat.size()-start/2-1+start]+vecFromMat[vecFromMat.size()-start/2+start])/2;
    } else { // in case of even-numbered matrix
        median = vecFromMat[(vecFromMat.size()-start-1)/2+start]; // odd-number of elements in matrix
    }

    std::cout<<"Median: "<<median<<std::endl;
    double standardDeviation = 1.4826*(1+5/(Input.cols-start - this->s.rows))*median;
    std::cout<<"Standard Deviation: "<<standardDeviation<< std::endl;

	cv::Mat outliers = cv::Mat::zeros(i.rows, i.cols, CV_32FC1);

    for(int row=0; row<i.rows; row++) {
        for(int col=0; col<i.cols; col++) {
            if(abs(i.fl(row, col)) < standardDeviation) {
                i.fl(row, col) = 1;
            } else if(i.fl(row, col) < 3*standardDeviation) {
                i.fl(row, col) = standardDeviation/abs(i.fl(row, col));
            } else {
                i.fl(row, col) = 0;
                outliers.fl(row, col) = 1.0f;
            }
        }
    }

    return i;
}
