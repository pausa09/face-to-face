#ifndef PRISONERSDILEMMA_H
#define PRISONERSDILEMMA_H

#include <random>
#include <array>
#include <unordered_map>

typedef std::vector< char > Mimic;
namespace std {
	template<>
	class hash<Mimic> {
	public:
		size_t operator()(const Mimic& s) const
		{
			size_t h = 0;

			for (auto iter = s.begin(); iter != s.end(); ++iter)
			{
				h ^= std::hash< Mimic::value_type >()(*iter);
			}

			return h;
		}
	};
}

class MimRec;

class PrisonersDilemma
{
public:
	
	PrisonersDilemma( MimRec* rec );

	int useraction, agentAction;
	static const unsigned int maxAction = 2;
	const double epsilonChoose = 0.5;
	const double gamma = 0.5;
	const double alpha = 0.2;
	const int maxEpisodes = 100;

	void beginGame();
	void userAction();
	


	const Mimic& getCurrentAgentAction()
	{
		return Q_value[ current_agent_state ].first;
	}

private:
	float calcprobalyAction();
	double calcReward(int useraction, int agentAction);

private:
	MimRec const* m_rec;

	size_t current_agent_state;
	size_t current_user_state;

	std::default_random_engine generator;
	std::uniform_real_distribution<double> epsilonDistribution;
	std::uniform_real_distribution<double> epsilonDistributionCoop;
	std::uniform_int_distribution<int> mimicDistribution;
	std::uniform_int_distribution<int> actionDistribution;
	std::uniform_int_distribution<int> auDistribution;

	typedef std::pair< Mimic, std::vector< std::array< double, maxAction > > > QValueRow;

	std::vector< QValueRow > Q_value;

	size_t getState(const Mimic& m);
	size_t chooseAgentState() ;
};

#endif //PRISONERSDILEMMA_H
