// https://software.intel.com/en-us/forums/oneapi-data-parallel-c-compiler/topic/856418
// Question regarding device-side memory outside of SYCL kernels.
// When running on a machine with an Intel iGPU, memory allocated with
// malloc_device is accessible outside a kernel -- this is unexpected.
// Turns out, this *is* expected since the Intel Unified Memory Architecture
// uses shared memory for the CPU and iGPU; see:
// https://software.intel.com/content/dam/develop/public/us/en/documents/the-architecture-of-intel-processor-graphics-gen11-r1new.pdf
// When compiling this code with Intel's llvm for CUDA, a segfault occurs when
// attempting to access the malloc_device memory outside a kernel (as expected).
// This is checked by instead using a cl::sycl::stream to output the data in
// device-side memory.
//
// Compile for Intel device(s):
// clang++ -fsycl [-DUSE_SYCL_CPU | -DUSE_SYCL_GPU] \
// -o mwe_1617 mwe_1617.cc
//
// Compile for CUDA device:
// clang++ -fsycl -DUSE_PI_CUDA \
// -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version \
// -o mwe_malloc_device /<...>/mwe_malloc_device.cc

#include <CL/sycl.hpp>
#include <iostream>

#ifdef USE_PI_CUDA
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

int main() {
  // Catch asynchronous exceptions
  auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception during generation:\n"
                  << e.what() << std::endl;
      }
    }
  };
  // Initialize device, queue and context
  cl::sycl::device dev;
#ifdef USE_PI_CUDA
  CUDASelector cuda_selector;
  try {
    dev = cl::sycl::device(cuda_selector);
  } catch (...) {
  }
#elif USE_SYCL_CPU
  dev = cl::sycl::device(cl::sycl::cpu_selector());
#elif USE_SYCL_GPU
  dev = cl::sycl::device(cl::sycl::gpu_selector());
#else
  dev = cl::sycl::device(cl::sycl::default_selector());
#endif
  cl::sycl::queue queue = cl::sycl::queue(dev, exception_handler);
  cl::sycl::context ctx = queue.get_context();
  // Name of the device to run on
  std::string dev_name =
      queue.get_device().get_info<cl::sycl::info::device::name>();
  std::cout << "Using device \"" << dev_name << "\"" << std::endl;

  // Ensure device can handle USM device allocations.
  if (!queue.get_device()
           .get_info<cl::sycl::info::device::usm_device_allocations>()) {
    std::cout << "ERROR :: device \"" << dev_name
              << "\" does not support usm_device_allocations!" << std::endl;
    return 1;
  }
  int hostArray[42];
  int* deviceArray = (int*)malloc_device(42 * sizeof(int), dev, ctx);
  for (int i = 0; i < 42; i++) hostArray[i] = 42;
  queue
      .submit([&](cl::sycl::handler& h) {
        // copy hostArray to deviceArray
        h.memcpy(deviceArray, &hostArray[0], 42 * sizeof(int));
      })
      .wait();

#ifdef USE_PI_CUDA
  queue
      .submit([&](cl::sycl::handler& cgh) {
        cl::sycl::stream out(1024, 256, cgh);
        cgh.single_task<class print1>([=] {
          out << "[Before mod] deviceArray[10] = " << deviceArray[10]
              << cl::sycl::endl;
        });
      })
      .wait_and_throw();
#else
  std::cout << "[Before mod] deviceArray[10] = " << deviceArray[10]
            << std::endl;
#endif

  queue.submit([&](cl::sycl::handler& h) {
    h.parallel_for<class foo>(cl::sycl::range<1>{42}, [=](cl::sycl::id<1> ID) {
      int i = ID[0];
      deviceArray[i]++;
    });
  });
  queue.wait();

#ifdef USE_PI_CUDA
  queue
      .submit([&](cl::sycl::handler& cgh) {
        cl::sycl::stream out(1024, 256, cgh);
        cgh.single_task<class print2>([=] {
          out << "[Before mod] deviceArray[10] = " << deviceArray[10]
              << cl::sycl::endl;
        });
      })
      .wait_and_throw();
#else
  std::cout << "[After mod] deviceArray[10] = " << deviceArray[10] << std::endl;
#endif

  return 0;
}