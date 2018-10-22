// (C) Copyright Renaud Detry   2007-2008.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/** @file */

#ifndef TRSL_TEST_COMMON_HPP
#define TRSL_TEST_COMMON_HPP

#include <trsl/is_picked_systematic.hpp>
#include <trsl/persistent_filter_iterator.hpp>
#include <examples/Particle.hpp>

#include <list>
#include <vector>
#include <iostream>
#include <numeric> // accumulate
#include <ctime>

#include <boost/random.hpp>

namespace trsl {
  namespace test {
  

#define TEST_VERBOSE 0

#define TRSL_TEST_DRED         "\033[1;31m"
#define TRSL_TEST_NOCOLOR      "\033[0m"
#define TRSL_TEST_FAILURE \
    { \
        std::cout << TRSL_TEST_DRED; \
        std::cout << __FILE__ << ":" << __LINE__ << ": test failure." << std::endl; \
        std::cout << TRSL_TEST_NOCOLOR; \
        std::cout << TRSL_NVP(random_seed) << std::endl; \
    }
#define TRSL_NVP(x) #x << ": " << x

    inline std::ostream&
    operator<<(std::ostream& out, const trsl::example::Particle& p)
    {
      out << "[(" << p.getX() << "," << p.getY() << ") " << p.getWeight() << "]";
      return out;
    }

    class PickCountParticle : public trsl::example::Particle
    {
    public:
      PickCountParticle(double weight, double x, double y) :
        Particle(weight, x, y), pickCount_(0) {}
    
      void pick() { pickCount_++; }
      unsigned getPickCount() const { return pickCount_; }
    private:
      unsigned pickCount_;
      double dump[16];
    };

    template<typename ContainerType>
    void generatePopulation(const size_t POPULATION_SIZE,
                            ContainerType& population,
                            bool normalizeWeights = true)
    {
      double totalWeight = 0;
      for (size_t i = 0; i < POPULATION_SIZE; ++i)
      {
        PickCountParticle p(double(rand())/RAND_MAX,
                            double(rand())/RAND_MAX,
                            double(rand())/RAND_MAX);
        totalWeight += p.getWeight();
        population.push_back(p);
      }
      // Normalize total weight.
      if (normalizeWeights)
        for (typename ContainerType::iterator i = population.begin();
             i != population.end(); ++i)
          i->setWeight(i->getWeight()/totalWeight);
    }

    inline double wac_function(const PickCountParticle& p)
    {
      return p.getWeight();
    }

    double wac_function_no_inline(const PickCountParticle& p);

    struct wac_functor
    {
      double operator()(const PickCountParticle& p) const
        {
          return p.getWeight();
        }
    };

    struct wac_functor_no_inline
    {
      double operator()(const PickCountParticle& p) const;
    };

  }
}

#endif // include guard
