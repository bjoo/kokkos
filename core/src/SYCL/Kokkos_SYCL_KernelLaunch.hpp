#include <SYCL/Kokkos_SYCL_Error.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<class Driver>
void sycl_launch(Driver driver_in) {



  std::cerr << "In sycl_launch" << std::endl;
  
 // cl::sycl::queue* q = driver_in.m_policy.space().impl_internal_space_instance()->m_queue;
  cl::sycl::queue* q = Kokkos::Experimental::SYCL().impl_internal_space_instance()->m_queue;
  std::cerr << "Queue pointer is: " << (unsigned long) q << std::endl;

  std::cerr << "range=" << driver_in.m_policy.end()-driver_in.m_policy.begin() << std::endl;
  std::cerr << "driver_in.ptr_d = " << (unsigned long)(driver_in.m_functor.ptr_d) << std::endl;

  q->submit([&](cl::sycl::handler& cgh) {
	  cl::sycl::stream out(1024,256,cgh);
	  cgh.parallel_for (
		  cl::sycl::range<1>(driver_in.m_policy.end()-driver_in.m_policy.begin()),
		  [=] (cl::sycl::id<1> item) {
	     	 size_t idx = item[0];
	     	 if (idx == 2 ) { // stop threads overwriting
	     		 out << "idx = " << idx << " PF ptr_d = " << (unsigned long)driver_in.m_functor.ptr_d << cl::sycl::endl;
	     	 }
	     	 driver_in.m_functor(idx, out);
         });
      });
      q->wait_and_throw();
}


} // namespace
} // namespace
} // namespace
