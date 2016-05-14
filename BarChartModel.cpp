#include "BarChartModel.h"
#include "MimRec.h"

#include <KDChartGlobal.h>
#include <QBrush>

#include <QDebug>

BarChartModel::BarChartModel(MimRec*rec, bool staticCalculation, QObject* parent)
	:
QAbstractTableModel( parent ),
m_rec( rec ),
bStaticCalculation( staticCalculation )
{

}

int BarChartModel::columnCount(const QModelIndex & parent) const
{
	if ( parent.isValid() )
	{
		Q_ASSERT(false);
		return -1;
	}

	return 1;
}

int BarChartModel::rowCount(const QModelIndex & parent) const
{
	if (parent.isValid())
	{
		Q_ASSERT(false);
		return -1;
	}

	return m_rec->numberOfActionUnits();
}

QVariant BarChartModel::data(const QModelIndex & index, int role) const
{
	QImage colorramp;
	if (!index.isValid() || index.column() != 0 || (index.row() >= m_rec->numberOfActionUnits()))
	{
		return QVariant();
	}

	switch( role )
	{
	case Qt::DisplayRole:
		return ( bStaticCalculation ? m_rec->getAUDetectionValue() : m_rec->getBackPropAUDetectionValue() ).at(index.row());
	case KDChart::DatasetBrushRole:
		if (!colorramp.load("C://Development//colorramp.png"))
		{
			qDebug() << "Cannor load colorramp";
		}

		float f = ( bStaticCalculation ? m_rec->getAUDetectionValue() : m_rec->getBackPropAUDetectionValue() ).at(index.row());

		return QBrush(QColor(colorramp.pixel(std::min(f, 0.99f) * colorramp.width(), 0)));
		//return (m_rec->getAUDetectionValue().at( index.row() ) > 0.99f) ? QBrush(Qt::green) : QBrush(Qt::red);
	}


	return QVariant();
}

QVariant BarChartModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if ( role == Qt::DisplayRole)
	{
		return section;
	}

	return QAbstractTableModel::headerData(section, orientation, role);
}
