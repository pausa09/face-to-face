#ifndef CHARTWIDGET_H
#define CHARTWIDGET_H

#include <QtWidgets/QWidget>

#include "BarChartModel.h"

class MimRec;
namespace KDChart
{
	class Chart;
	class BarDiagram;
}

class ChartWidget : public QWidget
{
	Q_OBJECT
public:
	explicit ChartWidget(MimRec* rec, bool staticCalculation, QWidget* parent = 0);
	void emitDataChanged()
	{ model->emitDataChanged(); }

private:
	BarChartModel* model;
	
};

#endif //CHARTWIDGET_H
