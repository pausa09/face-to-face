#include "ChartWidget.h"
#include "BarChartModel.h"

#include <KDChartChart>
#include <KDChartBarDiagram>
#include <KDChartLineDiagram>
#include <KDChartDataValueAttributes>
#include <KDChartGridAttributes>
#include <QtGui/QStandardItemModel>

ChartWidget::ChartWidget( MimRec* rec, bool staticCalculation, QWidget* parent)
	: QWidget(parent)
{
	KDChart::BarDiagram* diagram = new KDChart::BarDiagram( this );
	model = new BarChartModel(rec, staticCalculation, diagram);

	setWindowTitle( staticCalculation ? QStringLiteral("static") : QStringLiteral("backprop") );

	diagram->setModel(model);
	diagram->setPen(QPen(Qt::black, 0));
	
	KDChart::Chart* chart = new KDChart::Chart(this);

	chart->coordinatePlane()->replaceDiagram(diagram);
	
	QVBoxLayout* l = new QVBoxLayout(this);
	l->addWidget(chart);
	setLayout(l);

	KDChart::CartesianAxis *xAxis = new KDChart::CartesianAxis(diagram);
	KDChart::CartesianAxis *yAxis = new KDChart::CartesianAxis(diagram);
	xAxis->setPosition(KDChart::CartesianAxis::Bottom);
	yAxis->setPosition(KDChart::CartesianAxis::Left);

	diagram->addAxis(xAxis);
	diagram->addAxis(yAxis);

	KDChart::BarAttributes ba(diagram->barAttributes());
	
	diagram->setBarAttributes(ba);

	// display the values
	KDChart::DataValueAttributes dva(diagram->dataValueAttributes());
	KDChart::TextAttributes ta(dva.textAttributes());
	//rotate if you wish
	ta.setRotation( 0 );
	ta.setFont(QFont("Arial", 9));
	ta.setPen(QPen(QColor(Qt::black)));
	ta.setVisible(true);
	dva.setTextAttributes(ta);
	dva.setVisible(true);
	diagram->setDataValueAttributes(dva);
	
	

	KDChart::CartesianCoordinatePlane* plane = static_cast <KDChart::CartesianCoordinatePlane*>
		(diagram->coordinatePlane());
	
	//KDChart::HorizontalLineLayoutItem* line = new KDChart::HorizontalLineLayoutItem();
	
	//LineDiagram* lines = new LineDiagram(this, plane);

	//lines->setModel();


//	plane->addDiagram();
	QPen gridPen(Qt::blue);
	
	// Horizontal

	KDChart::GridAttributes gh = plane->gridAttributes(Qt::Vertical);
	gh.setGridPen(gridPen);
	gh.setGridStepWidth(0.135);
	plane->setGridAttributes(Qt::Vertical, gh);
	

}
