#pragma once

// Magic template redirector to store TSTored* for most types TWrapped,
// and const TStored* for const TWrapped.
//
// Usage: typename WrapConstPointerT< TWrapped, TStored >::pointer m_p;
// Example: typename WrapConstPointerT< Vector4f, uint8_t >::pointer m_p;
template< typename TWrapped, typename TStored >
struct WrapConstPointerT {
    typedef TStored* pointer;
};

// Partially specialize on const TWrapped.
template< typename TWrapped, typename TStored >
struct WrapConstPointerT< const TWrapped, TStored > {
    typedef const TStored* pointer;
};
