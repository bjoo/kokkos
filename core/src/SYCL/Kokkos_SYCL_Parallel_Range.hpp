#include<SYCL/Kokkos_SYCL_KernelLaunch.hpp>
#include <algorithm>
#include <functional>
namespace Kokkos {
namespace Impl {

template< class FunctorType , class ... Traits >
class ParallelFor< FunctorType
                 , Kokkos::RangePolicy< Traits ... >
                 , Kokkos::Experimental::SYCL
                 >
{
public:

  typedef Kokkos::RangePolicy< Traits ... > Policy;
  FunctorType  m_functor ;
  Policy       m_policy ;
  
// inline
  void execute() //const
  {
	  std::cerr << "in execute: ptr_d =  " << m_functor.ptr_d << std::endl;
	  Kokkos::Experimental::Impl::sycl_launch(*this);
    }

  ParallelFor( const FunctorType & arg_functor ,
               const Policy   &     arg_policy )
    : m_functor( arg_functor )
    , m_policy(  arg_policy )
    { }

};

}
}
