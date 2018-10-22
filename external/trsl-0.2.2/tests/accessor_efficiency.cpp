// (C) Copyright Renaud Detry   2007-2008.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <tests/common.hpp>
using namespace trsl::test;
#include <string>
#include <cassert>

static const size_t NB_ROUNDS = 100000;
static const size_t POPULATION_SIZE = 1000;
static const size_t SAMPLE_SIZE = 10;

unsigned long random_seed = time(NULL)*getpid();

template<typename WeightAccessor>
void trsl_loop(WeightAccessor acc,
               const std::string msg)
{
  typedef trsl::is_picked_systematic<
    PickCountParticle,
    double,
    WeightAccessor
    > is_picked;

  typedef trsl::persistent_filter_iterator
    <is_picked, std::vector<PickCountParticle>::const_iterator> sample_iterator;

  //-----------------------//
  // Generate a population //
  //-----------------------//
    
  std::vector<PickCountParticle> population;
  generatePopulation(POPULATION_SIZE, population);
  std::vector<PickCountParticle> const& const_pop = population;
    
  //------------------------------//
  // Benchmark it --------------- //
  //------------------------------//
  {
    // Create the systemtatic sampling functor.
    is_picked predicate(SAMPLE_SIZE, 1.0, acc);
      
    sample_iterator sb = sample_iterator(predicate, const_pop.begin(), const_pop.end());
    sample_iterator se = sample_iterator(predicate, const_pop.end(),   const_pop.end());
    sample_iterator si = sb;
    clock_t clock_start = clock();
    for (size_t count = 0; count < NB_ROUNDS; count++)
      for (si = sb; si != se; ++si)
      {
      }
    std::cout << "Bench for " << msg << ": " << clock() - clock_start << std::endl;
  }
}

template<typename WeightAccessor>
void stl_loop(WeightAccessor acc,
              const std::string msg)
{
  //-----------------------//
  // Generate a population //
  //-----------------------//
    
  std::vector<PickCountParticle> population;
  generatePopulation(POPULATION_SIZE, population);
  std::vector<PickCountParticle> const& const_pop = population;
    
  //------------------------------//
  // Benchmark it --------------- //
  //------------------------------//
  {

    double sum = 0;
    clock_t clock_start = clock();
    for (size_t count = 0; count < NB_ROUNDS; count++)
      for (std::vector<PickCountParticle>::const_iterator i = const_pop.begin();
           i != const_pop.end(); ++i)
      {
        sum += acc(*i);
      }
    std::cout << "Bench for " << msg << ": " << clock() - clock_start << std::endl;
    // avoid nop-ing the loop:
    assert(fabs(sum-NB_ROUNDS) < 1e-6*NB_ROUNDS);
  }
}

void drop_in_call()
{
  //-----------------------//
  // Generate a population //
  //-----------------------//
    
  std::vector<PickCountParticle> population;
  generatePopulation(POPULATION_SIZE, population);
  std::vector<PickCountParticle> const& const_pop = population;
    
  //------------------------------//
  // Benchmark it --------------- //
  //------------------------------//
#define TRSL_TEST_DROP_BODY(init, iter, msg) \
  { \
    double sum = 0; \
    init; \
    clock_t clock_start = clock(); \
    for (size_t count = 0; count < NB_ROUNDS; count++) \
      for (std::vector<PickCountParticle>::const_iterator i = const_pop.begin(); \
           i != const_pop.end(); ++i) \
      { \
        iter; \
      } \
    clock_t clock_dur = clock() - clock_start; \
    std::cout << "Bench for " msg ": " << clock_dur << std::endl; \
    assert(fabs(sum-NB_ROUNDS) < 1e-6*NB_ROUNDS); \
  }
  
  TRSL_TEST_DROP_BODY( ,sum += wac_function(*i), "wac_function");
  TRSL_TEST_DROP_BODY( ,sum += wac_function_no_inline(*i), "wac_function_no_inline");
  TRSL_TEST_DROP_BODY(double (*ptr)(const PickCountParticle& p) = &wac_function,sum += ptr(*i), "wac_function_pointer");
  typedef std::pointer_to_unary_function<const PickCountParticle&, double>
    wac_pointer_to_unary_function;
  TRSL_TEST_DROP_BODY(wac_pointer_to_unary_function ptr = std::ptr_fun(wac_function),
                      sum += ptr(*i), "wac_pointer_to_unary_function");
  TRSL_TEST_DROP_BODY(wac_pointer_to_unary_function ptr = std::ptr_fun(wac_function_no_inline),
                      sum += ptr(*i), "wac_pointer_to_unary_function_no_inline");
}
int main()
{
  // BSD has two different random generators
  srandom(random_seed);
  srand(random_seed);
  
    
  std::cout << "trsl_loop:" << std::endl;
  
  trsl_loop
    <std::pointer_to_unary_function<const PickCountParticle&, double> >
    (std::ptr_fun(wac_function), "wac_function");

  trsl_loop
    <std::pointer_to_unary_function<const PickCountParticle&, double> >
    (std::ptr_fun(wac_function_no_inline), "wac_function_no_inline");

  trsl_loop
    <wac_functor>
    (wac_functor(), "wac_functor");

  trsl_loop
    <wac_functor_no_inline>
    (wac_functor_no_inline(), "wac_functor_no_inline");

  trsl_loop
    <trsl::mp_weight_accessor<double, PickCountParticle> >
    (&PickCountParticle::getWeight, "mp_weight_accessor");

  std::cout << "stl_loop:" << std::endl;

  stl_loop
    <std::pointer_to_unary_function<const PickCountParticle&, double> >
    (std::ptr_fun(wac_function), "wac_function");

  stl_loop
    <std::pointer_to_unary_function<const PickCountParticle&, double> >
    (std::ptr_fun(wac_function_no_inline), "wac_function_no_inline");

  stl_loop
    <wac_functor>
    (wac_functor(), "wac_functor");

  stl_loop
    <wac_functor_no_inline>
    (wac_functor_no_inline(), "wac_functor_no_inline");

  stl_loop
    <trsl::mp_weight_accessor<double, PickCountParticle> >
    (&PickCountParticle::getWeight, "mp_weight_accessor");

  std::cout << "drop_in_call" << std::endl;
    
  drop_in_call();
    
  return 0;
}
