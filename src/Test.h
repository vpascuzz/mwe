//
// Test.h
//

#include <CL/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

static const unsigned int kNumElements = 10;
static const CONSTANT char kPrintf[] = "  device_ele[%d] = %d\n";

class Test {
 public:
  Test() : eles_(nullptr), eles_device_(nullptr) {}
  ~Test() {
    if (eles_device_) {
      cl::sycl::free(eles_device_, ctx_);
    }
    if (eles_) {
      free(eles_);
    }
  }

  // Allocate host- and device-side memory using USM.
  bool AllocMem(cl::sycl::device* dev);
  // Copy data to device.
  bool LoadToDevice();

 private:
  cl::sycl::context ctx_;
  unsigned int* eles_;
  unsigned int* eles_device_;
};
