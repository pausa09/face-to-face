#ifndef MIMICDETECTION_H
#define MIMICDETECTION_H

#include <array>
#include <opencv2/opencv.hpp>
#include <fstream>

class TrainingData;

class MimRec
{
public:
	static int numberOfActionUnits();

	typedef std::vector< std::pair< std::string, std::vector< std::pair< int, float > > > > ActionUnitsT;

	static const ActionUnitsT& actionUnitsDef();

public:
	MimRec(const TrainingData& neutral);
	


public:
	void calcForTrainingData(const cv::Mat& trackingPoints, const std::string& outputFilePath , const cv::Mat fittingImage);
	
	
	const std::vector< float >& getAUDetectionValue() const
	{ return vAUDetectionValue; }

	const std::vector< float >& getBackPropAUDetectionValue() const
	{ return vBackPropAUDetectionValue; }

	const std::vector< char >& getCurrentlyActiveActionUnits() const
	{ return vCurrentlyActiveActionsUnits; }

public:
	static const unsigned int numberOfFFP = 104;
	static const unsigned int numberOFDistances = 5500; //actually 5460 but more for special values
	static const unsigned int numberOfAreas = 5;

private:
	void areaFace();
	void calc_distance();
	void normalizeDistances();
	void calcIntensityScoring(const std::string& outputFilePath, const cv::Mat fittingImage);


	float getX(int i) const
	{ return currentPoints.at< float >(i * 2);  }

	float getY(int i) const
	{ return currentPoints.at< float >(i * 2 + 1); }

//Typen
private:
	typedef std::array< float, 5 > AreaType;

	enum AreaArrayTypePositions
	{
		Full = 0,
		Right = 1,
		Left = 2,
		Up = 3,
		Down = 4
	};

	typedef std::array< float, numberOFDistances > DistancesArrayType;

private:
	cv::Mat currentPoints;
	DistancesArrayType current_distances;
	DistancesArrayType distancesYAxis; 
	AreaType areas; 
	std::vector< float > vAUDetectionValue;
	std::vector< float > vBackPropAUDetectionValue;
	std::vector< char > vCurrentlyActiveActionsUnits;

	AreaType neutral_area;
	DistancesArrayType neutral_distances;

	std::ofstream eigenFace;
	
	float tresholdIntensity = 0.5f;
};

#endif //MIMICDETECTION_H
