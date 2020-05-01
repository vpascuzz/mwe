//
// Test.h
//

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
