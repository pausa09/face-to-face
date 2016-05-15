#ifndef MODEL_H
#define MODEL_H

#define MODEL_MOUTH 1
#define MODEL_LEFTEYE 2
#define MODEL_RIGHTEYE 4
#define MODEL_NOSE 8

#include <opencv2/opencv.hpp>
#include "trainingdata.h"

using namespace std;

#define fl at<float>

class Model: public TrainingData
{
public:
    Model();

    void loadDataFromFile(string fileName);
    //void saveData(string fileName);
	void placeModelInBounds(cv::Rect bounds);
	void placeGroupInBounds(cv::Rect bounds, int group, float scalingFactor);
	void placeMouthInBounds(cv::Rect bounds);
	void placeLeftEyeInBounds(cv::Rect bounds);
	void placeRightEyeInBounds(cv::Rect bounds);
	void placeNoseInBounds(cv::Rect bounds);

    void scaleSelectedPoints(float scale);

	void selectPointsInRect(cv::Rect selection);
    void selectPoint(int id);
    void unselectPoint(int id);
    void moveSelectedVertices(float dx, float dy);
    void unselectAllPoints();
	int findPointToPosition(cv::Point p, int tolerance);
	int findPointToPosition(cv::Point p);
    bool isPointSelected(int id);
    vector <int> getSelectedPoints();

    cv::Mat getTriangles();

	cv::Point2f getPoint(int id);
	cv::Point2f getPointFromMat(cv::Mat p, int id);
private:
	cv::Mat unscaledPoints;
	cv::Mat triangles;
	cv::Mat selected;

    void calcTriangleStructure();
};

#endif // MODEL_H
