#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include "SampleDecoder.h"
#include "MTRand.h"
#include "BRKGA.h"
#include "bossa_timer.h"
#include "ArgPack.hpp"
#include <fstream>

using namespace std;

vector<vector<double> > points;

int main(int argc, char* argv[]) {


	ArgPack single_ap(argc, argv);

	unsigned n = 0;			// size of chromosomes
	const unsigned p = ArgPack::ap().population;		// size of population
	const double pe = ArgPack::ap().populationElite;		// fraction of population to be the elite-set
	const double pm = ArgPack::ap().populationMutants;		// fraction of population to be replaced by mutants
	const double rhoe = ArgPack::ap().rhoe;	// probability that offspring inherit an allele from elite parent
	const unsigned K = ArgPack::ap().K;		// number of independent populations
	const unsigned MAXT = ArgPack::ap().threads;	// number of threads for parallel decoding

	const double cutoff_time = ArgPack::ap().time;

	// Reading instance
	string s;
	string file = ArgPack::ap().inputFile;

	ifstream f(file);

	int n_points, dim;
	f >> n_points >> dim;

	points = vector<vector<double> > (n_points);
	for(int i=0; i<n_points; ++i){
		points[i] = vector<double> (dim);
	}

	for(int i=0; i<n_points ;++i) {
		for(int d=0; d < dim ; ++d) {
			f >> points[i][d];
		}
	}

	SampleDecoder decoder;			// initialize the decoder
	const long unsigned rngSeed = ArgPack::ap().rngSeed;	// seed to the random number generator
	MTRand rng(rngSeed);				// initialize the random number generator

	n = n_points;

	// initialize the BRKGA-based heuristic
	BossaTimer timer;

	double bestValue = -1;
	double timerToBest;
	bool verbose = ArgPack::ap().verbose;
	int k_max = sqrt(n_points); // 2 <= k <= sqrt(n)
	int number_pop = k_max - 1;
	vector<BRKGA<SampleDecoder, MTRand>*> NPopulations(number_pop);
	vector<SampleDecoder*> vec_decoders(number_pop);
	vector<double> best_values(number_pop);

	int best_population = 0;
	double sum_best= 0.0;

	if(verbose){
		cout << "Initializing populations\n";
	}

	// Escolher K populações para serem escolhida
	int k = max((0.1 * number_pop),1.0);

	timer.start(); 
	double best_temp;
	for(int i=0; i < number_pop; ++i) {
		vec_decoders[i] = new SampleDecoder();
		vec_decoders[i]->set_k(i+2);
		NPopulations[i] = new BRKGA<SampleDecoder, MTRand> (n, p, pe, pm, rhoe, *vec_decoders[i], rng, K, MAXT);
		best_temp = (-1)*NPopulations[i]->getBestFitness();
		if(best_temp > bestValue) {
			timerToBest = timer.getTime();
			best_population = i;
			bestValue = best_temp;
		}
	}

	// Define os K populações a serem evoluidas;
	set<int> KChoosen;
	while( KChoosen.size() < k ){
		KChoosen.insert(rng.randInt(number_pop-1));
	}

	int idxBest = *KChoosen.begin();
	for ( auto i = KChoosen.begin() ; i != KChoosen.end() ; i++ ){
		if( NPopulations[idxBest]->getBestFitness() > NPopulations[*i]->getBestFitness() ){
			idxBest = *i;
		}
	}

	cout << "Best solution " << bestValue << " from pop " << best_population << endl;
	unsigned generation = 0;		// current generation
	const unsigned X_INTVL =  ArgPack::ap().exchangeBest;	// exchange best individuals at every 100 generations
	const unsigned X_NUMBER = ArgPack::ap().exchangeTop;	// exchange top 2 best
	const unsigned MAX_GENS = ArgPack::ap().generations;	// run for 1000 gens

	do {
		++generation;
		if(verbose){
			cout << "Envolving generation " << generation << " ";
		}
		NPopulations[idxBest]->evolve();	// evolve the population for one generation
		best_temp = (-1)*NPopulations[idxBest]->getBestFitness();
		if(best_temp > bestValue) {
			timerToBest = timer.getTime();
			bestValue = best_temp;
			best_population = idxBest;
		}
	} while (generation < MAX_GENS and timer.getTime() < cutoff_time);
	timer.pause();

	cout << "Best solution " << bestValue << " from pop " << best_population << " k = " << best_population +2 << endl;
	cout << "Total time = " << timer.getTime() << endl;
	cout << "Time to Best ttb = " << timerToBest << endl;

	for(int i=0;i<number_pop; ++i) {
		delete NPopulations[i];
		delete vec_decoders[i];
	}


	return 0;
}
