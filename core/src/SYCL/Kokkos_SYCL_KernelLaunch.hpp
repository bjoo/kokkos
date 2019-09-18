#include <SYCL/Kokkos_SYCL_Error.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<class Driver>
void sycl_launch(Driver driver_in) {



  std::cerr << "In sycl_launch" << std::endl;
  
  cl::sycl::queue* q = Kokkos::Experimental::SYCL().impl_internal_space_instance()->m_queue;
  std::cerr << "Queue pointer is: " << (unsigned long) q << std::endl;

#ifdef NO_LAMBDA
  std::cerr << "range=" << driver_in.m_policy.end()-driver_in.m_policy.begin() << std::endl;
  std::cerr << "driver_in.ptr_d = " << (unsigned long)(driver_in.m_functor.ptr_d) << std::endl;
  std::cerr << "Functor: " << driver_in.m_functor << std::endl;
#endif


#ifdef DO_LOCALCOPY
      bool localcopy = true;
#else
      bool localcopy = false;
#endif

#ifdef NO_LAMBDA
      bool printfunctor=true;
#else
      bool printfunctor=false;
#endif
      q->submit([&](cl::sycl::handler& cgh) {
	  cl::sycl::stream out(1024,256,cgh);

	  auto localfunctor = driver_in.m_functor;
#ifdef NO_LAMBDA
	  std::cout << "Before par for: ptr_d=" << driver_in.m_functor << std::endl;
	  std::cout << "Before par for: local copy = " << localfunctor << std::endl;
#endif
	  cgh.parallel_for (
		  cl::sycl::range<1>(driver_in.m_policy.end()-driver_in.m_policy.begin()),
		  [=] (cl::sycl::id<1> item) {
	     	 size_t idx = item[0];
	     	 if (idx == 2 ) { // stop threads overwriting
#ifdef NO_LAMBDA
		   out << "In par for: Before Kernel call: idx = " << idx << " argument functor : " <<  driver_in.m_functor << cl::sycl::endl;
		   out << "In par for: Before Kernel call: idx = " << idx << " localcopy : " <<  localfunctor << cl::sycl::endl;
#endif
	     	 }

		 if ( localcopy  )  {
		    if ( idx == 2 ) {  out << "Calling Localfunctor " << cl::sycl::endl; }
		    localfunctor(idx,out);

		 }
		 else {
		    if ( idx == 2 ) { out << "Calling driver_in.m_functor " << cl::sycl::endl; }
		    driver_in.m_functor(idx,out);
		 }

         });
      });
      q->wait_and_throw();
}


} // namespace
} // namespace
} // namespace
