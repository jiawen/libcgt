#pragma once

#include <chrono>

namespace libcgt { namespace core { namespace timeutils {

// t1 - t0 in milliseconds, where t0 and t1 a std::chrono::<clock>::time_point.
template< typename TimePoint >
int64_t dtMS( TimePoint t0, TimePoint t1 );

// t1 - t0 in microseconds, where t0 and t1 a std::chrono::<clock>::time_point.
template< typename TimePoint >
int64_t dtUS( TimePoint t0, TimePoint t1 );

// t1 - t0 in nano, where t0 and t1 a std::chrono::<clock>::time_point.
template< typename TimePoint >
int64_t dtNS( TimePoint t0, TimePoint t1 );

} } } // timeutils, core, libcgt

#include "TimeUtils.inl"
