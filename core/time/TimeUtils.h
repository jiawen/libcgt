#pragma once

#include <chrono>

namespace libcgt { namespace core { namespace time {

// t1 - t0 in milliseconds, where t0 and t1 a std::chrono::<clock>::time_point.
template< typename TimePoint >
int64_t dtMS( TimePoint t0, TimePoint t1 );

// t1 - t0 in microseconds, where t0 and t1 a std::chrono::<clock>::time_point.
template< typename TimePoint >
int64_t dtUS( TimePoint t0, TimePoint t1 );

// t1 - t0 in nano, where t0 and t1 a std::chrono::<clock>::time_point.
template< typename TimePoint >
int64_t dtNS( TimePoint t0, TimePoint t1 );

// Convert milliseconds to nanoseconds.
int64_t msToNS( int64_t ms );

// Convert milliseconds to microseconds.
int64_t msToUS( int64_t ms );

// Convert microseconds to nanoseconds.
int64_t usToNS( int64_t us );

} } } // time, core, libcgt

#include "libcgt/core/time/TimeUtils.inl"
