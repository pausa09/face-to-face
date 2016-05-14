
#include "PrisonersDilemma.h"
#include "MimRec.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <array>
#include <math.h>
#include <QtWidgets/QInputDialog>

int count_train = 0;

template< typename T >
T& operator <<(T& str, const Mimic& s)
{
	int size = s.size();

	str << "{ ";

	for( auto iter = s.begin(); iter != s.end(); ++iter)
	{
		str << static_cast< int >( *iter );

		if (--size > 0)
		{
			str << ", ";
		}
	}

	str << " }";

	return str;
}

PrisonersDilemma::PrisonersDilemma(MimRec* rec)
	:
	m_rec(rec),
	epsilonDistribution(0.0, 1.0),
	epsilonDistributionCoop(0.0, 1.0),
	mimicDistribution(0, 6),
	actionDistribution(0,1),
	auDistribution(0, 31)
{
	std::vector< char > baseEmotions[] = { 
		{}, //neutral
		{ 0, 1, 4, 15 },//sad
		{ 4, 5, 7, 8, 23 },//
		{ 0, 1, 2, 3, 4, 5, 7, 8, 20, 26 },// fear
		{ 9, 15 },// disgust
		{ 6, 12 }, // happy
		{ 0, 1, 2, 3, 5, 26 },//surprise
	

	};

	for (const auto& i : baseEmotions)
	{
		getState(i);
	}

	current_agent_state = 0;
	current_user_state = 0;
}


size_t PrisonersDilemma::getState(const Mimic& m)
{
	
	auto iter =
		std::find_if(Q_value.begin(), Q_value.end(), [&m](const QValueRow& test){ return m == test.first; });

	if (iter == Q_value.end())
	{
		const auto oldSize = Q_value.size();
		Q_value.resize(oldSize + 1);
		Q_value.back().first = m;

		auto& newRow = Q_value.back().second;
		newRow.reserve(oldSize + 1);

		for (size_t i = 0; i < oldSize; ++i )
		{
			Q_value[ i ].second.push_back({ epsilonDistribution(generator), epsilonDistribution(generator) });
			newRow.push_back({ epsilonDistributionCoop(generator), epsilonDistribution(generator) });
		}
		newRow.push_back({ epsilonDistributionCoop(generator), epsilonDistribution(generator) });

		return oldSize;
	}

	return iter - Q_value.begin();
}

size_t PrisonersDilemma::chooseAgentState() 
{
	count_train++;
	auto sumer = [](const QValueRow& row)
	{
		double d = 0.0;

		for (auto& i : row.second)
		{
			for (int a = 0; a < maxAction; ++a)
			{
				d += i.at(a);
			}
		}		return d;
	};

	auto currentSum = sumer(Q_value.front());
	

	size_t largest = 0;

	// 15 rounds for training
	const auto mim = epsilonDistribution(generator);
	if (count_train < 15)
		
	{		
		//largest = 0;
		largest = mimicDistribution(generator);

	}
	else{


		if (mim < 0.3)
		{
			largest = mimicDistribution(generator);
		}
		else{

			std::vector<float> tmpSum;

			for (size_t i = 1; i < Q_value.size(); ++i)
			{
				double test = sumer(Q_value.at(i));

				if (test > currentSum)
				{
				largest = i;
				}
				
			}

		}

	}
	
	return largest;
}

#include <QMessageBox>
#include <QtCore/QTextStream>

double PrisonersDilemma::calcReward(int useraction, int agentAction)
{
	QMessageBox msgBox;
	msgBox.setWindowTitle("Prisoners Dilemma");
	msgBox.setInformativeText("Show the next expression");
	msgBox.setWindowModality(Qt::NonModal);

	double value = 0.0;

	if (useraction == 0 && agentAction == 0)
	{		
		msgBox.setText("The two of you decided to cooperate : Each of you gets 1 year in prison!");
		value = 0.75;
	}
	else if (useraction == 0 && agentAction == 1)
	{
		msgBox.setText("The agent betrayed you : You get 3 years in prison , the agent remains unpunished");
		value = 1;
	}
	else if (useraction == 1 && agentAction == 0)
	{
		msgBox.setText("You betrayed the agent , he was silent : You remain unpunished , he gets 3 years in prison");
		value = -1;
	}
	else if (useraction == 1 && agentAction == 1)
	{
		msgBox.setText("The both of you defected: each of you gets 2 years in prison");
		value =  0.5;
	}
	else
	{ assert(false); }

	QString str;

	QTextStream(&str) << Q_value.at(current_agent_state).first;
	msgBox.setText( str+ msgBox.text() );

	msgBox.show();
	msgBox.exec();
	
	return value;
}

void PrisonersDilemma::beginGame()
{
	//softmax
	auto & val = Q_value.at(current_agent_state).second.at(current_user_state);

	double coop = val.at(0);
	double def = val.at(1);

	const auto probForCoop =
		std::exp(val.at(0)) / (std::exp(val.at(0)) + std::exp(val.at(1)));
	
	int current_agent_action = 1;
	const auto epsilon = epsilonDistribution(generator);


	if (epsilon < probForCoop)
	{ current_agent_action = 0; }


	int user_action = 0;
	
	QString str;

	QTextStream(&str) << Q_value.at( current_agent_state ).first;

	const int ret = QMessageBox::question(nullptr, "Prisoners Dilemma", "Do you want to cooperate?" + str);

	switch (ret)
	{
	case QMessageBox::Yes:
		user_action = 0;
		break;
	case QMessageBox::No:
		user_action = 1;
		break;
	}

	const double reward = calcReward(user_action, current_agent_action );

	const auto new_agent_state = chooseAgentState();

	const auto new_user_state = getState(m_rec->getCurrentlyActiveActionUnits());

	const auto newActionPair = Q_value.at(new_agent_state).second.at(new_user_state);
	const auto maxAction = std::max(newActionPair.at(0), newActionPair.at(1));
	
	auto& sink = Q_value[current_agent_state].second[current_user_state][current_agent_action];

	sink = sink + alpha * (reward + gamma * maxAction - sink);

	current_agent_state = new_agent_state;
	current_user_state = new_user_state;
	
}
