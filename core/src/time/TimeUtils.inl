namespace libcgt { namespace core { namespace timeutils {

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

} } } // timeutils, core, libcgt
