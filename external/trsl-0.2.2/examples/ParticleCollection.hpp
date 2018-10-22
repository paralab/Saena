// (C) Copyright Renaud Detry   2007-2008.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/** @file */

#ifndef TRSL_PARTICLECOLLECTION_HPP
#define TRSL_PARTICLECOLLECTION_HPP

#include <vector>
#include <trsl/is_picked_systematic.hpp>
#include <trsl/ppfilter_iterator.hpp>
#include <examples/Particle.hpp>

namespace trsl {
  namespace example {
  

    // Example population class

    class ParticleCollection
    {
    public:
      typedef trsl::is_picked_systematic<
        Particle> is_picked;

      typedef trsl::ppfilter_iterator<
        is_picked, std::vector<Particle>::iterator
      > sample_iterator;
      typedef trsl::ppfilter_iterator<
        is_picked, std::vector<Particle>::const_iterator
      > const_sample_iterator;

      ParticleCollection(): totalWeight_(0) {}
  
      void add(const Particle& p)
        {
          totalWeight_ += p.getWeight();
          particles_.push_back(p);
        }
  
      size_t size() const { return particles_.size(); }
  
      sample_iterator sample_begin(size_t sampleSize)
        {
          is_picked predicate(sampleSize, totalWeight_, &Particle::getWeight);
          return sample_iterator(predicate, particles_.begin(), particles_.end());
        }

      sample_iterator sample_end()
        {
          // For an end of range filter_iterator, the predicate operator()
          // will never be called. We can put anything for sampleSize and
          // populationWeight.  A "random" number should be provided, to
          // avoid a useless call to random().
          is_picked predicate(1, 1, 0, &Particle::getWeight);
          return sample_iterator(predicate, particles_.end(), particles_.end());
        }

      const_sample_iterator sample_begin(size_t sampleSize) const
        {
          is_picked predicate(sampleSize, totalWeight_, &Particle::getWeight);
          return const_sample_iterator(predicate, particles_.begin(), particles_.end());
        }

      const_sample_iterator sample_end() const
        {
          // For an end of range filter_iterator, the predicate operator()
          // will never be called. We can put anything for sampleSize and
          // populationWeight.  A "random" number should be provided, to
          // avoid a useless call to random().
          is_picked predicate(1, 1, 0, &Particle::getWeight);
          return const_sample_iterator(predicate, particles_.end(), particles_.end());
        }
    private:
      std::vector<Particle> particles_;
      double totalWeight_;
    };

  }
}

#endif // include guard
