#pragma once

#include <type_traits>

namespace libcgt
{
namespace core
{
namespace common
{

template< class T >
struct not_const : std::true_type
{

};

template< class T >
struct not_const< const T > : std::false_type
{

};

}; // namespace common
}; // namespace core
}; // namespace libcgt