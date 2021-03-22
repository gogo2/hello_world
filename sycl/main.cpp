#include <CL/sycl.hpp>
#include <array>
#include <iostream>


constexpr int SIZE = 4;
constexpr int N = 4;


struct  array_wrapper{
    std::array<int, SIZE> arr;
};

using VEC_T = array_wrapper;

int main() {
    std::array<VEC_T, N> vec_a{1,2,3,4}, vec_b{5,6,7,8}, vec_c{};

    sycl::queue queue{sycl::default_selector()};

    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::range<1> rng{N};
    {
        sycl::buffer<VEC_T, 1> a_buff(vec_a.data(), rng);
        sycl::buffer<VEC_T, 1> b_buff(vec_b.data(), rng);
        sycl::buffer<VEC_T, 1> c_buff(vec_c.data(), rng);

        queue.submit([&](sycl::handler &cgh) {
            auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buff.get_access<sycl::access::mode::read>(cgh);
            auto c_acc = c_buff.get_access<sycl::access::mode::read_write>(cgh);

            auto kernel = [=](sycl::id<1> id) {
                for (uint16_t i = 0; i < SIZE; ++i) {
                    c_acc[id].arr[i] = a_acc[id].arr[i] + b_acc[id].arr[i];
                }
            };
            cgh.parallel_for<class VectorAdd>(rng, kernel);
        });
    }
    for (const auto &c:vec_c) {
        for (uint16_t i = 0; i < SIZE; ++i) {
            std::cout << c.arr[i] << ' ';
        }
    } // 6 8 10 12
    std::cout << std::endl;

}
