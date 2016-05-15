#include "MimRec.h"
#include "BackProp.h"
#include <../AAMlib/trainingdata.h>
#include <iomanip> 
#include <fstream>
#include <KDChartGlobal.h>
#include <QBrush>

// Vector defines the distances for classify Action Units
/* string encodes the description for action units,
first value of pair encodes the number of distance,
second value of pair encodes the maximal displacement
*/
static const std::vector< std::pair< std::string, std::vector< std::pair< int, float > > > >
actionUnits = { 
	{ "Inner Brow Raiser left", { { 1550, 60.f } } }, 
	{ "Inner Brow Raiser right", { { 3450, 60.f } } }, 
	{ "Outer Brow Raiser left", { { 1637, 25.f } } }, 
	{ "Outer Brow Raiser right", { { 3512, 25.f } } },
	{ "Brow Lowerer", { { 322, -30.f }, { 347, -30.f } } },
	{ "Upper Lid Raiser", { { 0, 30.f }, { 1, 30.f } } }, 
	{ "Cheek Raiser", { { 2432, -25.f }, { 4033, -25.f } } },
	{ "Lid Tightener left", { { 1891, -30.f }, { 4380, -20.f } } }, 
	{ "Lid Tightener right", { { 3691, -30.f }, { 4471, -20.f } } },
	{ "Nose Wrinkler", { { 430, 10.f }, { 455, 10.f } } }, 
	{ "Upper Lip Raiser", { { 511, -20.f } } }, 
	{ "Nasolabial Deepener", { { 1, 10000.f } } },
	{ "Lip Corner Puller", { { 5465, 50.f }, { 5466, 50.f } } }, 
	{ "Cheek Puffer", { { 1, 10000 } } },// not defined
	{ "Dimpler", { { 1, 10000 } } }, // not defined
	{ "Lip Corner Depressor", { { 5463, 40.f }, { 5464, 40.f } } },
	{ "Lower Lip Depressor", { { 4997, 10.f }, { 2734, 200.f }, { 5027, 10.f }, { 4260, 30.f }, { 3019, 7000.f }, { 709, -35.f } } }, 
	{ "Chin Raiser", { { 805, 20.f }, { 414, -15.f } } }, 
	{ "Lip Puckerer", { { 681, 20.f }, { 2710, -20.f } } }, 
	{ "not defined", { { 1, 10000.f } } }, // not defined
	{ "Lip Stretcher", { { 2065, -10.f }, { 5473, 25.f }, { 3815 , -10.f} } }, 
	{ "not defined", { { 1, 1000.f } } }, // not defined
	{ "Lip Funneler", { { 5467, -20.f }, { 2710, -20.f } } },
	{ "Lip Tightener", { { 610, -40.f }, { 2710, 0.f } } }, 
	{ "Lip Pressor", { { 511, 20.f }, { 562, 20.f }, { 537, 15.f }, { 610, -25.f }, { 414, -5.f } } }, 
	{ "Lips parted", {  { 3019, 1000.f }} }, 
	{ "Jaw Drop", {  { 514, 30.f } } },
	{ "Mouth Stretch", {  { 514, 50.f } } }, 
	{ "Lip Suck", { { 610, 0.f } } }, 
	{ "Lip Corner Puller right", { { 5466, 50.f }, } },
	{ "Lip Corner Puller left", { { 5465, 50.f }, } },
	{ "Lid droops", {  { 1721, 60.f }, { 3571, 60.f } } },
	{ "Slit", { { 1721, 150.f }, { 3571, 150.f } } },
	{ "Eyes closed", { { 1721, 200.f }, { 3571, 200.f } } },
};


int MimRec::numberOfActionUnits()
{
	return actionUnits.size();
}

const  MimRec::ActionUnitsT& MimRec::actionUnitsDef()
{
	return actionUnits;
}

MimRec::MimRec(const TrainingData& neutral)
{
	currentPoints = neutral.getPoints();

	vAUDetectionValue.resize(actionUnits.size());
	vBackPropAUDetectionValue.resize(actionUnits.size());

	areaFace();
	calc_distance();

	neutral_area = areas;
	neutral_distances.swap(current_distances);
	
}

void MimRec::calcForTrainingData(const cv::Mat& trackingPoints, const std::string& outputFilePath, const cv::Mat fittingImage)
{
	currentPoints = trackingPoints;
	areaFace();
	calc_distance();
	normalizeDistances();
	calcIntensityScoring(outputFilePath, fittingImage);
}

//defines FacialFeaturePoints for specific face area

static const int indexOfFFP[16] = { 0, 35, 36, 38, 53, 52, 54, 56, 9, 31, 29, 27, 28, 13, 11, 10 };
static const int indexOfFFP_Left[9] = { 0, 9, 31, 29, 27, 28, 13, 11, 10 };
static const int indexOfFFP_Up[9] = { 0, 35, 36, 38, 53, 28, 13, 11, 10 };
static const int indexOfFFP_Down[9] = { 53, 52, 54, 56, 9, 31, 29, 27, 28 };

void MimRec::areaFace()
{
	/*calculate the five areas in face: whole face,  left,  right, up , bottom*/

	// get FFP for whole area
	float xPts[16], yPts[16];
	for (int i = 0; i < 16; ++i)
	{
		xPts[i] = { getX(indexOfFFP[i]) };
		yPts[i] = { getY(indexOfFFP[i]) };
	};


	//get FFP for right area
	float xPtsRight[9], yPtsRight[9];
	for (int i = 0; i < 9; ++i)
	{
		xPtsRight[i] = { getX(indexOfFFP[i]) };
		yPtsRight[i] = { getY(indexOfFFP[i]) };
	}
	

	//get FFP for left area
	float xPtsLeft[9], yPtsLeft[9];
	for (int i = 0; i < 9; ++i)
	{
		xPtsLeft[i] = { getX(indexOfFFP_Left[i]) };
		yPtsLeft[i] = { getY(indexOfFFP_Left[i]) };
	}

	//get FFP for upper area
	float xPtsUp[9], yPtsUp[9];
	for (int i = 0; i < 9; ++i)
	{
		xPtsUp[i] = { getY(indexOfFFP_Up[i]) };
		yPtsUp[i] = { getY(indexOfFFP_Up[i]) };
	}
	
	//get FFP for bottom area
	float xPtsDown[9], yPtsDown[9];
	for (int i = 0; i < 9; ++i)
	{
		xPtsDown[i] = { getX(indexOfFFP_Down[i]) };
		yPtsDown[i] = { getY(indexOfFFP_Down[i]) };
	}

	for (size_t i = 0; i < areas.size(); ++i)
	{
		areas[i] = 0.0f;
	}

	for (int i = 0; i < 16; ++i)
	{
		areas[Full] += (yPts[i % 16] + yPts[(i + 1) % 16])*(xPts[i % 16] - xPts[(i + 1) % 16]);
	}
	
	areas[Full] /= 2.0f;

	// Gauss's area formula (Trapezformel)
	for (int i = 0; i < 9; i++)
	{
		areas[Right] += (yPtsRight[i % 9] + yPtsRight[(i + 1) % 9])*
			(xPtsRight[i % 9] - xPtsRight[(i + 1) % 9]);

		areas[Left] += (yPtsLeft[i % 9] + yPtsLeft[(i + 1) % 9])*
			(xPtsLeft[i % 9] - xPtsLeft[(i + 1) % 9]);

		areas[Up] += (yPtsUp[i % 9] + yPtsUp[(i + 1) % 9])*
			(xPtsUp[i % 9] - xPtsUp[(i + 1) % 9]);

		areas[Down] += (yPtsDown[i % 9] + yPtsDown[(i + 1) % 9])*
			(xPtsDown[i % 9] - xPtsDown[(i + 1) % 9]);
	}

	areas[Right] /= 2.0f;
	areas[Left] /= 2.0f;
	areas[Down] /= 2.0f;
	areas[Up] /= 2.0f;
}

void MimRec::calc_distance()
{
	int countPoints = -1;
	for (int i = 0; i < numberOfFFP; i++)
	{
		cv::Vec2f pts1(getX(i), getY(i));
		float ptsYAxis1 = getY(i);
		for (int j = i; j < numberOfFFP; j++)
		{
			countPoints++;
			cv::Vec2f pts2(getX(j), getY(j));

			current_distances[countPoints] = cv::norm(pts1, pts2);

			float ptsYAxis2 = getY(j);
			distancesYAxis[countPoints] = ptsYAxis1 - ptsYAxis2;
		}

	}

	//special values for Lid Tightener
	cv::Vec2f middlePointrightEye((getX(47) + getX(44)) / 2, (getY(47) + getY(44)) / 2);
	cv::Vec2f ptlidright(getX(45), getY(45));
	current_distances[0] = cv::norm(middlePointrightEye, ptlidright);

	cv::Vec2f middlePointLeftEye((getX(22) + getX(19)) / 2, (getY(22) + getY(19)) / 2);
	cv::Vec2f ptlidLeft(getX(20), getY(20));
	current_distances[1] = cv::norm(middlePointLeftEye, ptlidLeft);


	cv::Vec2f ptLowerlidright(getX(46), getY(46));
	current_distances[5461] = cv::norm(middlePointrightEye, ptLowerlidright);

	cv::Vec2f ptLowerlidLeft(getX(21), getY(21));
	current_distances[5462] = cv::norm(middlePointLeftEye, ptLowerlidLeft);


	//angle between 7-8-30 for Lip Corner Depressor
	current_distances[5463] =
		acos((pow(current_distances[826], 2) + pow(current_distances[708], 2) - pow(current_distances[730], 2)) / (2 * abs(current_distances[826]) * abs(current_distances[708])));
	current_distances[5463] *= 180.f / 3.14f;

	current_distances[5464] =
		acos((pow(current_distances[851], 2) + pow(current_distances[708], 2) - pow(current_distances[755], 2)) / (2 * abs(current_distances[708]) * abs(current_distances[851])));
	current_distances[5464] *= 180.f / 3.14f;
	

	//angle between 6-5-30 for Lip Corner Puller
	current_distances[5465] =
		acos((pow(current_distances[535], 2) + pow(current_distances[511], 2) - pow(current_distances[633], 2)) / (2 * abs(current_distances[535]) * abs(current_distances[511])));
	current_distances[5465] *= 180.f / 3.14f;

	current_distances[5466] =
		acos((pow(current_distances[560], 2) + pow(current_distances[511], 2) - pow(current_distances[658], 2)) / (2 * abs(current_distances[560]) * abs(current_distances[511])));
	current_distances[5466] *= 180.f / 3.14f;

	//angle between 6 -55 -30 for funneler
	current_distances[5467] =
		acos((pow(current_distances[633], 2) + pow(current_distances[658], 2) - pow(current_distances[2710], 2)) / (2 * abs(current_distances[633]) * abs(current_distances[658])));
	current_distances[5467] *= 180 / 3.14f;
	
	//angle between 22-19-16 for Inner Brow Raiser left
	current_distances[5468] =
		acos((pow(current_distances[1808], 2) + pow(current_distances[1547], 2) - pow(current_distances[1550], 2)) / (2 * abs(current_distances[1808]) * abs(current_distances[1547])));
	current_distances[5468] *= 180.f / 3.14f;
	//angle between 41-44-47 für Inner Brow Raiser right
	current_distances[5469] =
		acos((pow(current_distances[3447], 2) + pow(current_distances[3633], 2) - pow(current_distances[3450], 2)) / (2 * abs(current_distances[3447]) * abs(current_distances[3633])));
	current_distances[5469] *= 180.f / 3.14f;

	//angle between 22-17-19 for Outer Brow Raiser left
	current_distances[5470] =
		acos((pow(current_distances[1637], 2) + pow(current_distances[1808], 2) - pow(current_distances[1634], 2)) / (2 * abs(current_distances[1637]) * abs(current_distances[1808])));
	current_distances[5470] *= 180.f / 3.14f;
	//angle between 44-42-47 für Outer Brow Raiser right
	current_distances[5471] =
		acos((pow(current_distances[3633], 2) + pow(current_distances[3512], 2) - pow(current_distances[3509], 2)) / (2 * abs(current_distances[3633]) * abs(current_distances[3512])));
	current_distances[5471] *= 180.f / 3.14f;

	//cheek raiser
	cv::Vec2f cheek1(getX(26), getY(26));
	current_distances[5472] = cv::norm(middlePointrightEye, cheek1);

	//lip stretcher
	current_distances[5473] = abs(getX(55) - getX(30));

}

#include <QtWidgets/QInputDialog>

void MimRec::calcIntensityScoring(const std::string& outputFilePath, const cv::Mat fittingImage)
{
	vCurrentlyActiveActionsUnits.clear();
	float rel_distance[numberOFDistances];
	
	for (size_t i = 0; i < current_distances.size(); i++)
	{
		const float tmp = current_distances[i] - neutral_distances[i];
		rel_distance[i] = (tmp * 100 / neutral_distances[i]);							
	}
	
	int count = -1;

	std::vector< std::vector<float> > norm_inputdata(35, std::vector<float>());
	
	for (unsigned char au = 0; au < actionUnits.size(); ++au)
	{
		++count;
		const auto& tests = actionUnits.at(au);
		int numberOfValidTests			=	0;
		float intensityScoringOfEachDis =	0;
		float intensityScoring			=	0;
		bool valid						=	true;
		vAUDetectionValue[au]			=	0;
			
		for (size_t test = 0; test < tests.second.size(); ++test)
		{
			auto& p = tests.second.at(test);
			
			const auto distance =
				(current_distances[p.first] - neutral_distances[p.first]) * 100 / neutral_distances[p.first];
			
			norm_inputdata[count].push_back(distance / 100);
			
			if (p.second < 0.0f && distance < 0.0f)
			{
				
				if(distance < p.second)
				{
					intensityScoringOfEachDis += 1;
					++numberOfValidTests;
				}
				else
				{ 
					intensityScoringOfEachDis += tanh(distance / p.second);
				}
			}

			if (p.second > 0.0f && distance > 0.0f)
			{
				
				if(distance > p.second)
				{
					intensityScoringOfEachDis += 1;
					++numberOfValidTests;
				}
				else
				{
					intensityScoringOfEachDis += tanh(distance / p.second);
				}
			}
			else
			{
				intensityScoringOfEachDis += 0;
			}
		}
	
		
		//vAUDetectionValue[au] = static_cast< float >(numberOfValidTests) /static_cast< float >( tests.second.size() );
		
 		vAUDetectionValue[au] = static_cast< float > (intensityScoringOfEachDis) / tests.second.size();
	

		if (vAUDetectionValue[au] < 0.0f)
		{
			vAUDetectionValue[au] = 0.0f;
		}
		else if (vAUDetectionValue[au] > 0.99f)
		{
			vCurrentlyActiveActionsUnits.push_back( au );
		}
	}
	
	BackProp intensity;
	vBackPropAUDetectionValue = intensity.getIntensityScoring(norm_inputdata, outputFilePath);

	vCurrentlyActiveActionsUnits.clear();

	for (size_t i = 0; i < vBackPropAUDetectionValue.size(); ++i)
	{
		if (vBackPropAUDetectionValue[i] > tresholdIntensity)
		{
			vCurrentlyActiveActionsUnits.push_back(static_cast< char >( i ));
		}
	}

	static const std::vector< std::pair< std::string, std::vector< int > > >
		emotions = { 
			{ "happy", { 6, 12, 7, 8, 25 } },
			{ "sad", { 0, 1, 15 } },
			{ "surprised", { 0,1,2,3, 5, 25} },
			{ "fearful", { 0,1,2,3, 4, 5, 20, 25} },
			{ "angry", {4, 5, 7, 23} },
			{ "disgusted", { 9, 15 } },
			{ "contemptuously left", { 29 } },
			{ "contemptuously left", { 30 } }
		};

	for (size_t e = 0; e < emotions.size(); ++e)
	{
		const auto& emotion = emotions.at(e);
		bool allTestsValid = true;

		for (size_t test = 0; allTestsValid && (test < emotion.second.size()); ++test)
		{
			allTestsValid = vAUDetectionValue[emotion.second.at(test)];
		}
		if (allTestsValid)
		{
			std::cout << "You look " << emotion.first << std::endl;
		}
	}
	
	/*const QString str = QInputDialog::getText(nullptr, "Mimic", "What Mimic do you want to send?");

	auto l = str.splitRef(' ');

	vCurrentlyActiveActionsUnits.clear();

	for (const auto& i : l)
	{
		vCurrentlyActiveActionsUnits.push_back(i.toInt());
	}

	std::sort(vCurrentlyActiveActionsUnits.begin(), vCurrentlyActiveActionsUnits.end());*/
}

void MimRec::normalizeDistances()
{
	/*distances have to normalized so it is comparable with the neutral face */

	float diffsToNeutral[numberOfAreas];
	for(int i = 0; i < numberOfAreas; ++i)
	{
		diffsToNeutral[i] = { abs(areas[i]) - abs(neutral_area[i]) * 100 / abs(neutral_area[i]) };
	};
	
	float ratioFaceareasToFullneutral[numberOfAreas ], ratioFaceareasToFullArea[numberOfAreas],
		diffRatioAreasToNeutral[numberOfAreas];

	for (int i = 1; i < numberOfAreas; ++i)
	{
		ratioFaceareasToFullneutral[i] = { abs(neutral_area[i]) * 100 / abs(neutral_area[Full]) };
		ratioFaceareasToFullArea[i] = { abs(areas[i]) * 100 / abs(areas[Full]) };
		diffRatioAreasToNeutral[i] = { ratioFaceareasToFullneutral[i] - ratioFaceareasToFullArea[i] };
	}

	float ratio_Up_Down = diffRatioAreasToNeutral[Down] - diffRatioAreasToNeutral[Up];
	float ratio_Right_Left = diffRatioAreasToNeutral[Right] - diffRatioAreasToNeutral[Left];
	float ng_full, ng_right, ng_left, ng_up, ng_down, ag;
	float bm_right, bm_left, bm_up, bm_down; // new benchmarks

	std::vector<float> at, at_right, at_left, at_up, at_down, nt;
	ag = sqrt(abs(neutral_area[Full]));				// entire length in neutralface
						
	// new entire length for each sector
	ng_down = sqrt(abs(areas[Down]));
	ng_up = sqrt(abs(areas[Up]));
	ng_right = sqrt(abs(areas[Right]));
	ng_left = sqrt(abs(areas[Left]));

	/*
	the head is turning left/right or back/forward if the ratios are changing
	new length are calculate with nt=ng/ag*at
	*/
	static const int distanceLeft[15] = { 1550, 1637, 322,0,2432,1891,4380,430,5465,5463,4997,2734,2065,1721,537 };
	static const int distanceRight[15] = { 3450,3512,347,1,4033,3691,4471,455,5466,5464,5027,4260,3815,3571,562 };



	if (abs(ratio_Up_Down) < 10 && abs(ratio_Right_Left) < 10)
	{
		ng_full = sqrt(abs(areas[Full]));					// new entire length
		const float bm_full = 100 * ng_full / ag;

		for (size_t i = 0; i < current_distances.size(); i++)
		{
			current_distances[i] = current_distances[i] / bm_full * 100;
		}
	}
		
	// person look to left or right side
	else if (ratio_Right_Left > 1 && abs(ratio_Up_Down) < 1)
	{
		std::cout << "You look to left side" << endl;
		bm_right = 100 * ng_right / ag;
		bm_left = 100 * ng_left / ag;
		for (size_t i = 0; i < 15; i++)
		{
			current_distances[distanceLeft[i]] = current_distances[distanceLeft[i]] / bm_left * 100;
			current_distances[distanceRight[i]] = current_distances[distanceRight[i]] / bm_right * 100;
		}
		
		std::cout << "New Benchmarks \n Right:" << bm_right << "\n Left:" << bm_left << endl;
	}
	else if (ratio_Right_Left < 1 && abs(ratio_Up_Down) < 1)
	{
		std::cout << "You look to right side" << endl;
		bm_right = 100 * ng_right / ag;
		bm_left = 100 * ng_left / ag;
		for (size_t i = 0; i < 15; i++)
		{
			current_distances[distanceLeft[i]] = current_distances[distanceLeft[i]] / bm_left * 100;
			current_distances[distanceRight[i]] = current_distances[distanceRight[i]] / bm_right * 100;
		}
		std::cout << "New Benchmarks \n Right:" << bm_right << "\n Left:" << bm_left << endl;
	}

	/*person looks up or down but not left or right*/
	else if (ratio_Up_Down > 1 && abs(ratio_Right_Left) < 1)
	{
		std::cout << "You look up" << endl;
		bm_down = 100 * ng_down / ag;
		bm_up = 100 * ng_up / ag;
		std::cout << "New Benchmarks \n Down:" << bm_down << "\n Up:" << bm_up << endl;
		for (size_t i = 0; i < current_distances.size(); i++)
		{
			//			nt.push_back(distance[i] / bm_full * 100);
		}
	}

	else if (ratio_Up_Down < 1 && abs(ratio_Right_Left) < 1)
	{
		std::cout << "You look down" << endl;
		bm_down = 100 * ng_down / ag;
		bm_up = 100 * ng_up / ag;
		std::cout << "New Benchmarks \n Down:" << bm_down << "\n Up:" << bm_up << endl;
	}
	else
	{
		std::cout << "Detection failed" << endl;
	}

	
}
