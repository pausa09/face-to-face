#include "robustaam.h"

using namespace cv;

#define fl at<float>

RobustAAM::RobustAAM()
{
    this->triangleShapeHessians.clear();
    this->triangleAppHessians.clear();
}

void RobustAAM::train() {
    AAM::calcShapeData();
    AAM::calcAppearanceData();

    AAM::calcGradients();
    AAM::calcJacobian();
    AAM::calcSteepestDescentImages();

    this->calcTriangleHessians();

    this->initialized = true;
}

float RobustAAM::fit() {
    Mat deltaLambda = this->calcAppeanceUpdate();
    AAM::updateAppearanceParameters(deltaLambda);

    Mat deltaShapeParam = this->calcShapeUpdate();
    AAM::updateInverseWarp(deltaShapeParam);

    /*
    namedWindow("reconstruction");
    imshow("reconstruction", this->getAppearanceReconstructionOnFittingImage());
    */

    Mat parameterUpdates;
    vconcat(deltaLambda, deltaShapeParam, parameterUpdates);
    return sum(abs(parameterUpdates))[0]/parameterUpdates.rows;
}

void RobustAAM::calcTriangleHessians() {
    for(int i=0; i<this->triangles.rows; i++) {
        Mat mask = Mat::zeros(this->modelHeight, this->modelWidth, CV_8UC1);

        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        Point2f pa,pb,pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        for(int row = min_y; row < max_y; row++) {
            for(int col = min_x; col < max_x; col++) {
                Point px;
                px.x = col;
                px.y = row;

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

                if((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1)) {
                    mask.at<int>(row,col) = 255;
                }
            }
        }

        mask = mask.reshape(1,1);

        //H^i_p
        Mat sdImg = this->steepestDescentImages.clone();
        for(int j=0; j<sdImg.rows; j++) {
            bitwise_and(sdImg.row(j), sdImg.row(j), sdImg.row(j), mask);
        }

        Mat Hessian = sdImg*sdImg.t();
        this->triangleShapeHessians.push_back(Hessian);

        Hessian.release();
        sdImg.release();

        //H^i_A
        Mat appSdImg = this->A.clone();
        for(int j=0; j<appSdImg.rows; j++) {
            bitwise_and(appSdImg.row(j), appSdImg.row(j), appSdImg.row(j), mask);
        }

        Mat AppHessian = appSdImg*appSdImg.t();

        this->triangleAppHessians.push_back(AppHessian);
    }
}

Mat RobustAAM::applyRobustErrorFunction(Mat img) {
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
    cout<<"Start: "<<start<<endl;

    if (vecFromMat.size()%2==0) {
        median = (vecFromMat[vecFromMat.size()-start/2-1+start]+vecFromMat[vecFromMat.size()-start/2+start])/2;
    } else { // in case of even-numbered matrix
        median = vecFromMat[(vecFromMat.size()-start-1)/2+start]; // odd-number of elements in matrix
    }

    cout<<"Median: "<<median<<endl;
    double standardDeviation = 1.4826*(1+5/(Input.cols-start - this->s.rows))*median;
    cout<<"Standard Deviation: "<<standardDeviation<<endl;

    for(int row=0; row<i.rows; row++) {
        for(int col=0; col<i.cols; col++) {
            if(abs(i.fl(row, col)) < standardDeviation) {
                i.fl(row, col) = 1;
            } else if(i.fl(row, col) < 3*standardDeviation) {
                i.fl(row, col) = standardDeviation/abs(i.fl(row, col));
            } else {
                i.fl(row, col) = 0;
            }
        }
    }

    /*
    float c = 2.385f;
    for(int row=0; row<i.rows; row++) {
        for(int col=0; col<i.cols; col++) {
            i.fl(row, col) = 1/(1+(i.fl(row,col)/c)*(i.fl(row,col)/c));
            if(i.fl(row, col) < threshold) {
                outliers.at<int>(row, col) = 255;
            }
        }
    }
    */

    namedWindow("Weights");
    imshow("Weights", i.reshape(1,this->modelHeight));

    return i;
}

Mat RobustAAM::calcWeightedHessian(vector<Mat> triangleHessians) {
    Mat Hessian = Mat::zeros(triangleHessians.at(0).rows, triangleHessians.at(0).cols, CV_32FC1);
    Mat eImg = this->errorImage.reshape(1, this->modelHeight);
    eImg = this->applyRobustErrorFunction(eImg);

    for(int i=0; i<this->triangles.rows; i++) {
        float sum = 0.0f;
        int n = 0;
        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        Point2f pa,pb,pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
        int min_x = floor(min(pa.x, min(pb.x, pc.x)));
        int max_x = ceil(max(pa.x, max(pb.x, pc.x)));
        int min_y = floor(min(pa.y, min(pb.y, pc.y)));
        int max_y = ceil(max(pa.y, max(pb.y, pc.y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
                Point px;
                px.x = col;
                px.y = row;

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

                if((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1)) {
                    sum += eImg.at<float>(row,col);
                    n++;
                }
            }
        }

        float weight = (sum/(float)n);
        Hessian += weight*triangleHessians.at(i);
    }

    return Hessian;
}

Mat RobustAAM::calcAppeanceUpdate() {
    AAM::calcErrorImage();
    Mat appHessian = this->calcWeightedHessian(this->triangleAppHessians);

    Mat weights = this->applyRobustErrorFunction(this->errorImage);
    weights = repeat(weights, this->A.rows, 1);

    Mat deltaLambda = -appHessian.inv()*weights.mul(this->A)*this->errorImage.t();

    return deltaLambda;
}

Mat RobustAAM::calcShapeUpdate() {
    AAM::calcErrorImage();
    Mat shapeHessian = this->calcWeightedHessian(this->triangleShapeHessians);

    Mat weights = this->applyRobustErrorFunction(this->errorImage);
    weights = repeat(weights, this->steepestDescentImages.rows, 1);

    Mat deltaShapeParam = -shapeHessian.inv()*weights.mul(this->steepestDescentImages)*this->errorImage.t();

    return deltaShapeParam;
}
