// https://github.com/intel/llvm/issues/1617
// DIBlockByRefStruct on DICompositeType is no longer supported #1617
// MWE reproducing the warning "DIBlockByRefStruct on DICompositeType is no
// longer supported" when compiling SYCL code with dpcpp/Beta06 and using CPU or
// GPU devices. This warning was not issued in versions < Beta06.
//
// Compile for Intel device(s):
// clang++ -g -fsycl [-DGPU_DEVICE | -DCPU_DEVICE | -DCUDA_DEVICE] \
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

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

static const unsigned int kNumElements = 10;
static const CONSTANT char kPrintf[] = "  device_ele[%d] = %d\n";

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
  };

  // Allocate host- and device-side memory.
  bool AllocMem(cl::sycl::device* dev) {
    eles_ = (unsigned int*)malloc(kNumElements * sizeof(int));
    if (!eles_) {
      std::cout << "Cannot allocate host-side memory!\n";
      return false;
    }
    eles_device_ =
        (unsigned int*)malloc_device(kNumElements * sizeof(int), *dev, ctx_);
    if (!eles_device_) {
      std::cout << "Cannot allocate device-side memory!\n";
      return false;
    }
    return true;
  }

  bool LoadToDevice() {
// Device, queue and context setup
#ifdef CPU_DEVICE
    cl::sycl::cpu_selector dev_selector;
#elif GPU_DEVICE
    cl::sycl::gpu_selector dev_selector;
#elif CUDA_DEVICE
    CUDASelector dev_selector;
#else
    cl::sycl::default_selector dev_selector;
#endif
    cl::sycl::device dev(dev_selector);
    cl::sycl::queue queue(dev);
    ctx_ = queue.get_context();
    // Name of the device to run on
    std::string dev_name =
        queue.get_device().get_info<cl::sycl::info::device::name>();
    std::cout << "Using device \"" << dev_name << "\"" << std::endl;

    // Ensure device can handle USM device allocations.
    if (!queue.get_device()
             .get_info<cl::sycl::info::device::usm_device_allocations>()) {
      std::cout << "ERROR :: device \"" << dev_name
                << "\" does not support usm_device_allocations!" << std::endl;
      return false;
    }

    // Allocate memory
    if (!AllocMem(&dev)) {
      std::cout << "Could not allocate host- or device-side memory!"
                << std::endl;
      return false;
    }

    // Fill host-side memory with dummy Elements
    for (unsigned int iel = 0; iel < kNumElements; ++iel) {
      eles_[iel] = iel;
    }

    // Copy host memory to device
    auto ev_cpy_cells =
        queue.memcpy(eles_device_, &eles_[0], kNumElements * sizeof(int));
    ev_cpy_cells.wait_and_throw();

#ifndef CUDA_DEVICE
    // Read device memory, unless a CUDA device (which currently don't support
    // experimental::printf())
    std::cout << "Test device cells..." << std::endl;
    auto ev_cellinfo = queue.submit([&](cl::sycl::handler& cgh) {
      cgh.parallel_for<class Dummy>(
          cl::sycl::range<1>(kNumElements),
          [=, dev_cells_local = this->eles_device_](cl::sycl::id<1> idx) {
            unsigned int id = (int)idx[0];
            int val = dev_cells_local[id];
            cl::sycl::intel::experimental::printf(kPrintf, id, val);
          });
    });
    ev_cellinfo.wait_and_throw();
#endif  // !CUDA_DEVICE

    return true;
  }

 private:
  cl::sycl::context ctx_;
  unsigned int* eles_;
  unsigned int* eles_device_;
};

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