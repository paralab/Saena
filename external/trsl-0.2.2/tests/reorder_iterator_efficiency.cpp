// (C) Copyright Renaud Detry   2007-2008.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <trsl/random_permutation_iterator.hpp>
#include <tests/common.hpp>
using namespace trsl::test;

class HeavyPickCountParticle : public PickCountParticle
{
public:
  HeavyPickCountParticle(double weight, double x, double y) :
    PickCountParticle(weight, x, y) {}
  HeavyPickCountParticle(const PickCountParticle &p) :
    PickCountParticle(p) {}
private:
  double payload[0];
};


int main()
{
  // BSD has two different random generators
  unsigned long random_seed = time(NULL)*getpid();
  srandom(random_seed);
  srand(random_seed);
  
  typedef std::vector<HeavyPickCountParticle> ParticleArray;
  
  const unsigned N_ROUNDS = 1000;
  const size_t POPULATION_SIZE = 10000;

  std::vector<HeavyPickCountParticle> cpop;
  generatePopulation(POPULATION_SIZE, cpop);

  // ---------------------------------------------------- //
  // Test 1:  --------------------------- //
  // ---------------------------------------------------- //
  {
    
    // Type definitions, once and for all.

    typedef trsl::reorder_iterator
      <ParticleArray::iterator> permutation_iterator;

    //-----------------------//
    // Generate a population //
    //-----------------------//
    
    std::vector<HeavyPickCountParticle> population = cpop;
    
    //------------------------------------------------//
    // Test 1a:  //
    //------------------------------------------------//
    {
      unsigned pickCount = 0;
            
      clock_t start = clock();
      for (unsigned round = 0; round < N_ROUNDS; round++)
      {        
        permutation_iterator sb =
          trsl::random_permutation_iterator(population.begin(),
                                            population.end());
        permutation_iterator se = sb.end();
        for (permutation_iterator si = sb; si != se; ++si)
        {
          si->pick();
          pickCount++;
        }
      }
      clock_t end = clock(); std::cerr << end-start << std::endl;
    }
  }

  // ---------------------------------------------------- //
  // Test 1:  --------------------------- //
  // ---------------------------------------------------- //
  {
    
    // Type definitions, once and for all.

    typedef trsl::reorder_iterator
      <ParticleArray::iterator> permutation_iterator;

    //-----------------------//
    // Generate a population //
    //-----------------------//
    
    std::vector<HeavyPickCountParticle> population = cpop;
    
    //------------------------------------------------//
    // Test 1a:  //
    //------------------------------------------------//
    {
      unsigned pickCount = 0;
            
      clock_t start = clock();
      for (unsigned round = 0; round < N_ROUNDS; round++)
      {        
        std::random_shuffle(population.begin(), population.end(), trsl::rand_gen::uniform_int);

        for (ParticleArray::iterator si = population.begin(); si != population.end(); ++si)
        {
          si->pick();
          pickCount++;
        }
      }
      clock_t end = clock(); std::cerr << end-start << std::endl;
    }
  }
  return 0;
}
