#ifndef BARCHARTMODEL_H
#define BARCHARTMODEL_H

#include <QAbstractTableModel>

class MimRec;

class BarChartModel : public QAbstractTableModel
{
	Q_OBJECT

public:
	BarChartModel(MimRec* rec, bool staticCalculation, QObject* parent);
	

public:
	int columnCount(const QModelIndex & parent = QModelIndex()) const override;
	int rowCount(const QModelIndex & parent = QModelIndex()) const override;

	QVariant data(const QModelIndex & index, int role = Qt::DisplayRole) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;

	

public:
	void emitDataChanged()
	{ emit dataChanged(index(0, 0), index(rowCount() - 1, 0), { Qt::DisplayRole }); }
	
private:
	MimRec const * m_rec;
	bool bStaticCalculation;
	
};

#endif //MIMICDETECTION_H