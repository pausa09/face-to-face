#ifndef BackProp_H
#define BackProp_H

#include <random>
#include <array>
#include <fstream>

class BackProp
{
public:
	BackProp();
	std::vector< float >& getIntensityScoring(const std::vector< std::vector<float> > &norm_inputdata, const std::string& outputFilePath );
	void Train(int numEpochs);
private:

	static const int numberOfHiddenUnits = 34;
	static const int numberOfPattern = 42;
	static const int numberofActionUnits = 34;

	std::string inputHidden = "C:\\Development\\MimRec\\weightInputHidden.txt";
	std::string hiddenOutput = "C:\\Development\\MimRec\\weightHiddenOutput.txt";

	int patnum;

	float hiddenValues[numberofActionUnits][numberOfHiddenUnits];

	const float learnRate_InputHidden = 0.015f;
	const float learnRate_HiddenOutput = 0.01f;
	

	float outPred[numberOfPattern][numberofActionUnits];
	float errThisPat[numberOfPattern][numberofActionUnits];
	std::vector< float > output;
	float RMSerror[numberofActionUnits];

	// the weights
	std::array<std::array<std::vector<float>, numberOfHiddenUnits>, numberofActionUnits>  weightsInputHidden;
	float weightsHiddenOutput[numberofActionUnits][numberOfHiddenUnits];

	// the data
	std::vector< float > inputtraining;
	std::vector< std::vector<float> > trainInputs;
	
	std::array< float, numberofActionUnits > targetOutput;
	float activation[numberofActionUnits][numberOfHiddenUnits];
	float trainoutput;

public:
	void initWeights();
	void readTrainingDataFromFile(const std::string& filePath, int line);
	void readTargetOutputFromFile(const std::string& filePath, int line);
	void calcNet(int patnum);
	void UpdateWeightHiddenOutput();
	void UpdateWeightInputHidden();
	void calcTotalError();
	void displayResults();
	void saveWeightsToFiles();
	void loadWeightsFromFile(const std::string& filePath, const std::string& filePath2);

private:
	std::default_random_engine random_generator;
	std::uniform_real_distribution<float> valOfWeights;
};


#endif //BackProp_H
