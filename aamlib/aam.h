#ifndef AAM_H
#define AAM_H

#include <opencv2/opencv.hpp>

class AAM
{
protected:
    cv::Mat trainingShapes;
	cv::Mat trainingImagesRows;
	std::vector<cv::Mat> trainingImages;

    bool initialized;

    int numShapeParameters;
    int numAppParameters;
    int numPoints;
    int modelWidth;
    int modelHeight;

    int steps;

	cv::Mat fittingImage;
	cv::Mat errorImage;
	cv::Mat fittingShape;

	cv::Mat s0;
	cv::Mat s;
	cv::Mat s_star;
	cv::Mat A0;
	cv::Mat A;

	cv::Mat p;      //shape parameters
	cv::Mat lambda; //appearance parameters

    std::vector <std::vector <int> > triangleLookup;

	cv::Mat gradX, gradY;
	cv::Mat gradXA, gradYA;
	cv::Mat steepestDescentImages;
	cv::Mat jacobians;

	cv::Mat alignShapeData(cv::Mat &shapeData);
	cv::Mat procrustes(const cv::Mat &X, const cv::Mat &Y);
	cv::Mat moveToOrigin(const cv::Mat &A);
	cv::Point2f calcMean(const cv::Mat &A);

	void warpTextureFromTriangle(cv::Point2f srcTri[3], const cv::Mat &originalImage, cv::Point2f dstTri[3], cv::Mat warp_final);

	void calcTriangleStructure(const cv::Mat &s);
    void calcShapeData();
    void calcAppearanceData();
    void calcGradients();
    void calcGradX();
    void calcGradY();
    void calcJacobian();
	cv::Mat derivateWarpToPoint(int vertexId);
    void calcSteepestDescentImages();
    void calcErrorImage();

	void updateAppearanceParameters(cv::Mat deltaLambda);
	void updateInverseWarp(cv::Mat deltaShapeParam);
public:
    AAM();

	void addTrainingData(const cv::Mat &shape, const cv::Mat &image);

	cv::Mat triangles;

	cv::Mat warpImageToModel(const cv::Mat &inputImage, const cv::Mat &inputPoints);
	cv::Mat warpImage(const cv::Mat &inputImage, const cv::Mat &inputPoints, const cv::Mat &outputImage, const cv::Mat &outputPoints);

	int findPointInShape(const cv::Point2f &p);
	cv::Point2f getPointFromMat(const cv::Mat &m, int pointId);
    void setFirstPoint(int id, int &a, int &b, int &c);

    void setNumShapeParameters(int num);
    void setNumAppParameters(int num);

	void setFittingImage(const cv::Mat &fittingImage);
	void setStartingShape(const cv::Mat &shape);
    void resetShape();
	cv::Mat getFittingShape();
	cv::Mat getErrorImage();
    double getErrorPerPixel();

	cv::Mat getAppearanceReconstruction();
	cv::Mat getAppearanceReconstructionOnFittingImage();

    bool isInitialized();
    bool hasFittingImage();

    virtual void train() = 0;
    virtual float fit() = 0;
};

#endif // AAM_H
