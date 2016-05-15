#include "BackProp.h"
#include <math.h> 
#include <iostream>
#include "MimRec.h"
#include <array>
#include <fstream>
#include <string>
#include <iomanip>

BackProp::BackProp()
{
	output.resize(numberofActionUnits);
}

void BackProp::Train(int numEpochs)
{
	
	
	/* Preload the trainingdata file */
	std::ifstream trainingDataFile("C:\\Development\\MimRec\\normin2.txt");

	if (!trainingDataFile.is_open())
	{
		std::cout << "Please load Trainingsdata" << std::endl;
	}

	const auto& def = MimRec::actionUnitsDef();

	std::vector< decltype(trainInputs) > preLoadedTrainingData;

	while (!trainingDataFile.eof())
	{
		decltype(trainInputs) tmpTrainInputs(def.size());

		for (size_t i = 0; i < def.size(); ++i)
		{
			tmpTrainInputs[i].resize(def[i].second.size());

			for (size_t j = 0; j < def[i].second.size(); ++j)
			{
				trainingDataFile >> tmpTrainInputs[i][j];
				
			}
		}

		preLoadedTrainingData.push_back(std::move(tmpTrainInputs));
	}

	/* Preload the target output file */
	std::ifstream targetOutputFile("C:\\Development\\MimRec\\targetnew.txt");

	if (!targetOutputFile.is_open())
	{
		std::cout << " Please load target outputs" << std::endl;
	}

	std::vector< decltype(targetOutput) > preLoadedTargetOutput;

	while (!targetOutputFile.eof())
	{
		decltype(targetOutput) tmpTargetOutput;

		for (size_t j = 0; j < numberofActionUnits; ++j)
		{
			targetOutputFile >> tmpTargetOutput[j];
		}

		preLoadedTargetOutput.push_back(std::move(tmpTargetOutput));
	}

	/* that loop is to train the network */

	std::uniform_int_distribution<int> number(0, numberOfPattern);
	readTrainingDataFromFile("C:\\Development\\MimRec\\normin2.txt", 1);
	loadWeightsFromFile(hiddenOutput, inputHidden);
	initWeights();

	for (int i = 0; i <= numEpochs; i++)
	{		
		for (int j = 0; j < numberOfPattern; ++j)
		{
			patnum = number(random_generator);						// choose input randomly
			trainInputs = preLoadedTrainingData[ patnum ];			// load train data
			targetOutput = preLoadedTargetOutput[ patnum ];			// load target output
			calcNet(patnum);
			UpdateWeightHiddenOutput();								// calculate the new weights for hidden neuron			
			UpdateWeightInputHidden();								// calculate the new weights for input neuron	
		}
		calcTotalError();
	}

	displayResults();
	saveWeightsToFiles();
}


/*the neuronal network is build here*/
void BackProp::calcNet(int patnum)
{
	for (int k = 0; k < numberofActionUnits; ++k)
	{
		for (int i_numHid = 0; i_numHid < numberOfHiddenUnits; i_numHid++)
		{
			hiddenValues[k][i_numHid] = 0.0;

			for (size_t j = 0; j < trainInputs[k].size(); j++)
			{
				hiddenValues[k][i_numHid] += (trainInputs[k][j] * weightsInputHidden[k][i_numHid][j]);
			}

			hiddenValues[k][i_numHid] = tanh(hiddenValues[k][i_numHid]);
		}
	}
	
	for (int k = 0; k < numberofActionUnits; ++k)
	{
		outPred[patnum][k] = 0.0;
	}

	for (int k = 0; k < numberofActionUnits; ++k)
	{
		for (int i = 0; i<numberOfHiddenUnits; i++)
		{
			outPred[patnum][k] += hiddenValues[k][i] * weightsHiddenOutput[k][i];
		}
		errThisPat[patnum][k] = outPred[patnum][k] - targetOutput[k];
	}
}


void BackProp::UpdateWeightHiddenOutput()
{
	for (int i = 0; i < numberofActionUnits; ++i){

		for (int k = 0; k < numberOfHiddenUnits; k++)
		{
			weightsHiddenOutput[i][k] = 
				weightsHiddenOutput[i][k] - learnRate_HiddenOutput * (errThisPat[patnum][i]) * hiddenValues[i][k];
		}
	}
}

void BackProp::UpdateWeightInputHidden()
{
	for (int i_AU = 0; i_AU < numberofActionUnits; ++i_AU)
	{
		for (int i = 0; i < numberOfHiddenUnits; i++)
		{
			for (size_t k = 0; k < trainInputs[i_AU].size(); k++)
			{
				float x = 1 - (hiddenValues[i_AU][i] * hiddenValues[i_AU][i]);
				x *= weightsHiddenOutput[i_AU][i] * (errThisPat[patnum][i_AU]) * learnRate_InputHidden;
				x *= trainInputs[i_AU][k];

				weightsInputHidden[i_AU][i][k] = weightsInputHidden[i_AU][i][k] - x;
			}
		}
	}
}

/* inits the weights with random values, only need for new traing
	else load the weights from file*/
void BackProp::initWeights()
{
	for (int k = 0; k < numberofActionUnits; ++k)
	{
		for (int numHid = 0; numHid < numberOfHiddenUnits; numHid++)
		{
			weightsHiddenOutput[k][numHid] = valOfWeights(random_generator);

			for (size_t inputCount = 0; inputCount < trainInputs[k].size(); inputCount++)
			{
				weightsInputHidden[k][numHid].push_back(valOfWeights(random_generator));
			}
		}
	}
}


/*loads trainings data*/
void BackProp::readTrainingDataFromFile( const std::string& filePath, int line )
{
	std::ifstream file(filePath);

	if (!file.is_open())
	{
		std::cout << " File could not load. \n Load the file with target outputs." << std::endl;
	}

	while ((--line) >= 0)
	{
		std::string str;
		std::getline(file, str);

		if ( !file )
		{
			std::cout << " Load the file with trainingdata. \n Make sure the number of lines is the same as patnum" << std::endl;
		}
	}

	auto def = MimRec::actionUnitsDef();

	trainInputs.resize(def.size());
	
	for (size_t i = 0; i < def.size(); ++i)
	{
		trainInputs[i].resize(def[i].second.size());

		for (size_t j = 0; j < def[i].second.size(); ++j)
		{
			file >> trainInputs[i][j];
		}
	}
}

void BackProp::readTargetOutputFromFile(const std::string& filePath, int line)
{
	std::ifstream file(filePath);

	if (!file.is_open())
	{
		std::cout << " Load the file with target outputs." << std::endl;
	}

	while ((--line) >= 0)
	{
		std::string str;
		std::getline(file, str);

		if (!file)
		{
			std::cout << " Load the file with target outputs. \n Make sure the number of lines is the same as patnum" << std::endl;
		}
	}
	for (size_t j = 0; j < numberofActionUnits; ++j)
	{
		file >> targetOutput[j];
	}
	
}


void BackProp::calcTotalError()
{
	for (int i = 0; i < numberofActionUnits; i++){
		RMSerror[i] = 0.0;
	}
	
		for (int i = 0; i < numberofActionUnits; i++)
		{
			RMSerror[i] += 0.5f * (targetOutput[i] - outPred[patnum][i]) * (targetOutput[i] - outPred[patnum][i]);				
		}


		/*for (int i = 0; i < numberofActionUnits; i++){
			//RMSerror[i] /= numberOfPattern;
			std::cout << "error pro au" << i<< " : " << RMSerror[i] << std::endl;
		}
		*/
}

/* the results of each round were displyed here, is used for manual check*/
void BackProp::displayResults()
{

	for (int i = 0; i < numberOfPattern; ++i)
	{
		patnum = i;
		std::cout << "Output of Training: " << patnum << std::endl;
		for (int au = 0; au < numberofActionUnits; au++)
		{
			readTrainingDataFromFile("C:\\Development\\MimRec\\normin2.txt", patnum);
			readTargetOutputFromFile("C:\\Development\\MimRec\\targetnew.txt", patnum);

			calcNet(patnum);

			if (outPred[patnum][au] < 0)
			{
				outPred[patnum][au] = 0;
			}
			else if (outPred[patnum][au] > 1)
			{
				outPred[patnum][au] = 1;
			}
			
			std::cout << std::setprecision(1)<< "au: " << au << "   " << outPred[patnum][au] << "\t" << targetOutput[au]  << std::endl;

		}		

	}

}

/* save weights to file. outputfilepath can changed in header file*/
void BackProp::saveWeightsToFiles()
{
	std::ofstream weightIH(inputHidden);
	std::ofstream weightHO(hiddenOutput);
	if (weightIH.is_open() && weightHO.is_open())
	{
		for (int k = 0; k < numberofActionUnits; ++k)
		{
			for (int numHid = 0; numHid < numberOfHiddenUnits; numHid++)
			{
				weightHO << weightsHiddenOutput[k][numHid] << std::endl;

				for (unsigned int inputCount = 0; inputCount < trainInputs[k].size(); inputCount++)
				{
					weightIH << weightsInputHidden[k][numHid][inputCount] << std::endl;
				}
			}
		}
	}
	else
	{
		std::cout << "Unable to open file" << std::endl;
	}
}

void BackProp::loadWeightsFromFile(const std::string& filePath, const std::string& filePath2)
{
	std::ifstream weightsHOFile, weightsIHFile;
	float weightsHO, weightsIH;

	weightsHOFile.open(filePath);
	weightsIHFile.open(filePath2);

	if (weightsHOFile.is_open() && weightsIHFile.is_open())
	{
		for (int k = 0; k < numberofActionUnits; ++k)
		{
			for (int numHid = 0; numHid <numberOfHiddenUnits; numHid++)
			{
				weightsHOFile >> weightsHO;
				weightsHiddenOutput[k][numHid] = weightsHO;

				for (size_t inputCount = 0; inputCount < trainInputs[k].size(); inputCount++)
				{
					weightsIHFile >> weightsIH;
					weightsInputHidden[k][numHid].push_back(weightsIH);
				}
			}
		}
	}
	else
	{
		std::cout << "Unable to open file" << std::endl;
	}

}

/*this function is need for feedforward*/
std::vector< float >& BackProp::getIntensityScoring(const std::vector< std::vector<float> > &norm_inputdata, const std::string& outputFilePath)
{
	
	trainInputs = norm_inputdata;
	loadWeightsFromFile(hiddenOutput, inputHidden);

	for (int i_Au = 0; i_Au < numberofActionUnits; ++i_Au)
	{
		for (int i_numHid = 0; i_numHid < numberOfHiddenUnits; i_numHid++)
		{
			activation[i_Au][i_numHid] = 0.0;

			for (size_t j = 0; j < trainInputs[i_Au].size(); j++)
			{
				activation[i_Au][i_numHid] += (trainInputs[i_Au][j] * weightsInputHidden[i_Au][i_numHid][j]);
			}

			activation[i_Au][i_numHid] = tanh(activation[i_Au][i_numHid]);
		}
	}
	for (int k = 0; k < numberofActionUnits; ++k)
	{
		output[k] = 0.0;
	}


	for (int k = 0; k < numberofActionUnits; ++k)
	{
		for (int i = 0; i<numberOfHiddenUnits; i++)
		{output[k] += activation[k][i] * weightsHiddenOutput[k][i];}

		if (output[k] < 0)
		{output[k] = 0;}

		else if (output[k] > 1)
		{output[k] = 1;}
		std::cout << std::setprecision(2) << k << ":" << output[k] << std::endl;
	}

	return output;

	/*std::ofstream myfile;
	myfile.open(outputFilePath, std::ofstream::trunc);
	myfile << "Code" << ";" << "Soll" << ";" << "Ist BP" << std::endl;
	for (int k = 0; k < numberofActionUnits; ++k)
	{
		myfile << std::setprecision(1) << k << ";" << ";" << output[k] << std::endl;
	}
	*/

}
