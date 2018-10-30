// (C) Copyright Renaud Detry   2007-2008.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <cassert>
#include <functional>

#include <trsl/is_picked_systematic.hpp>
#include <examples/Particle.hpp>
#include <examples/ParticleCollection.hpp>

int main()
{
  using namespace trsl::example;
  const size_t POPULATION_SIZE = 100;
  const size_t SAMPLE_SIZE = 10;

  //-----------------------//
  // Generate a population //
  //-----------------------//
  
  ParticleCollection population;
  for (size_t i = 0; i < POPULATION_SIZE; ++i)
  {
    Particle p(double(rand())/RAND_MAX,  // weight
               double(rand())/RAND_MAX,  // position (x)
               double(rand())/RAND_MAX); // position (y)
    population.add(p);
  }
  
  //----------------------------//
  // Sample from the population //
  //----------------------------//
  
  ParticleCollection sample;

  //-- population contains 100 elements. --//

  for (ParticleCollection::const_sample_iterator
         si = population.sample_begin(SAMPLE_SIZE),
         sb = si,
         se = population.sample_end();
       si != se; ++si)
  {
    std::cout << "sample_" << std::distance(sb, si) << "'s weight = " <<
      si->getWeight() << std::endl;

    Particle p = *si;
    p.setWeight(1);
    sample.add(p);

    // ... or do something else with *si ...
  }
  
  //-- sample contains 10 elements. --//
  
  assert(sample.size() == SAMPLE_SIZE);
  return 0;
}
