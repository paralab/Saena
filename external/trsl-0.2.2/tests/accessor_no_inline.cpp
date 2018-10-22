// (C) Copyright Renaud Detry   2007-2008.
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <tests/common.hpp>

namespace trsl
{
  namespace test
  {
    double wac_function_no_inline(const PickCountParticle& p)
    {
      return p.getWeight();
    }

    double wac_functor_no_inline::operator()(const PickCountParticle& p) const
    {
      return p.getWeight();
    }
  }
}
