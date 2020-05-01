// https://github.com/intel/llvm/issues/1617
// DIBlockByRefStruct on DICompositeType is no longer supported #1617
// MWE reproducing the warning "DIBlockByRefStruct on DICompositeType is no
// longer supported" when compiling SYCL code with dpcpp/Beta06 and using CPU or
// GPU devices. This warning was not issued in versions < Beta06.
// This MWE is based on:
// https://github.com/vpascuzz/FastCaloSycl/blob/master/src/Geo.cc
//
// Compile for Intel device(s):
// clang++ -g -fsycl [-DGPU_DEVICE | -DCPU_DEVICE] \
// -o mwe_1617 mwe_1617.cc
//
// Compile for CUDA device:
// clang++ -g -fsycl -DCUDA_DEVICE \
// -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version \
// -o mwe_1617 mwe_1617.cc
//
// Omitting `-g` suppresses the warning. No warning seen in any case with a CUDA
// device (though there's a floating point runtime exception).

#include <CL/sycl.hpp>
#include <iostream>

#include "Test.h"

#ifdef CUDA_DEVICE
class CUDASelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& Device) const override {
    using namespace cl::sycl::info;

    const std::string DeviceName = Device.get_info<device::name>();
    const std::string DeviceVendor = Device.get_info<device::vendor>();
    const std::string DeviceDriver =
        Device.get_info<cl::sycl::info::device::driver_version>();

    if (Device.is_gpu() && (DeviceVendor.find("NVIDIA") != std::string::npos) &&
        (DeviceDriver.find("CUDA") != std::string::npos)) {
      return 1;
    };
    return -1;
  }
};
#endif

// Main program
int main() {
  Test* te = new Test();
  // Ensure data was transferred to device.
  if (!te->LoadToDevice()) {
    std::cout << "Test::LoadToDevice() failed!\n";
  }
  if (te) {
    delete te;
    te = nullptr;
  }
  return 0;
}