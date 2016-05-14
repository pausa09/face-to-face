#ifndef FACETRACKING_H
#define FACETRACKING_H

class MimRec;
class PrisonersDilemma;
class ChartWidget;

#include <../AAMlib/icaam.h>
#include <QtCore/QObject>
#include <QtNetwork/QTcpServer>

class QTcpSocket;

class FaceTracking : public QObject
{
	Q_OBJECT
public:
	FaceTracking();
	~FaceTracking();
public:
	void run();
	
	void newConnection();
	void processDataFromSocket(bool forceSend = false );

private:
	void loadTrainingDataNew(const std::string &fileName);
	void drawShape(const cv::Mat& image, const cv::Mat& points);

public:
	MimRec* calculator = nullptr;
	PrisonersDilemma* pd = nullptr;
	ChartWidget* pStaticCalcChartWidget;
	ChartWidget* pBackPropChartWidget;
	ICAAM aam;

	std::vector<std::string> descriptions;
	cv::Mat groups;

	QTcpServer theServer;

	QTcpSocket* theSocket = nullptr;
};

#endif //FACETRACKING_H