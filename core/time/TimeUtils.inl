namespace libcgt { namespace core { namespace time {

template< typename TimePoint >
int64_t dtMS( TimePoint t0, TimePoint t1 )
{
    return std::chrono::duration_cast<std::chrono::milliseconds>
        ( t1 - t0 ).count();
}

template< typename TimePoint >
int64_t dtUS( TimePoint t0, TimePoint t1 )
{
    return std::chrono::duration_cast<std::chrono::microseconds>
        ( t1 - t0 ).count();
}

template< typename TimePoint >
int64_t dtNS( TimePoint t0, TimePoint t1 )
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>
        ( t1 - t0 ).count();
}

inline int64_t msToNS( int64_t ms )
{
    return ms * 1000000;
}

inline int64_t msToUS( int64_t ms )
{
    return ms * 1000;
}

inline int64_t usToNS( int64_t us )
{
    return us * 1000;
}

} } } // time, core, libcgt
