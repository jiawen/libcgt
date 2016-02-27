#pragma once

#include <initializer_list>
#include <string>

// A 1D range at integer coordinates
// Considered a *half-open* interval:
// [x, x + width)
class Range1i
{
public:

    Range1i(); // (0,0), a null range.
    explicit Range1i( int size ); // (0, size)
    // { origin, size }
    Range1i( std::initializer_list< int > os );

    Range1i( const Range1i& copy ); // copy constructor
    Range1i& operator = ( const Range1i& copy ); // assignment operator

    // The origin coordinate, as is.
    int origin() const;
    int& origin();

    // The size value, as is: it may be negative.
    int size() const;
    int& size();

    // TODO: make these clear
    int left() const; // origin.x
    int right() const; // origin.x + size

    // abs(size()), >= 0.
    int width() const;

    // TODO: --> exactCenter()?
    float center() const;

    // Returns true if size() >= 0.
    // Call standardized() to return a valid range with the endpoints flipped
    bool isStandardized() const;

    // Returns the same range but with size() >= 0.
    Range1i standardized() const;

    std::string toString() const;

    // Whether x is in this half-open interval.
    bool contains( int x );

    // Returns the smallest Range1i that contains both r0 and r1.
    // r0 and r1 do not have to be standard.
    static Range1i united( const Range1i& r0, const Range1i& r1 );

private:

    int m_origin;
    int m_size;

};
