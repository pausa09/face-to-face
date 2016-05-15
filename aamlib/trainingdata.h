#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#define MODEL_MOUTH 1
#define MODEL_LEFTEYE 2
#define MODEL_RIGHTEYE 4
#define MODEL_NOSE 8

#include <opencv2/opencv.hpp>

using namespace std;

class TrainingData
{
protected:
    cv::Mat points;
	cv::Mat image;
	cv::Mat groups;
    vector<string> descriptions;

public:
    TrainingData();

    void loadDataFromFile(string fileName);
    void saveDataToFile(string fileName);

	cv::Mat getPoints() const;
	cv::Mat getImage();
	cv::Mat getGroups();
    vector<string> getDescriptions();

	void setPoints(cv::Mat p);
	void setImage(cv::Mat i);
	void setGroups(cv::Mat g);
    void setDescriptions(vector<string> desc);
};

#endif // TRAININGDATA_H
