#include <SYCL/sycl.hpp>
#include <array>
#include <iostream>

int main() {
    constexpr int SIZE = 4;
    std::array<int, SIZE> vec_a{1, 2, 3, 4}, vec_b{5, 6, 7, 8}, vec_c{};

    sycl::queue queue{sycl::default_selector()};

    std::cout << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    sycl::range<1> rng{SIZE};
    {
        sycl::buffer<int, 1> a_buff(vec_a.data(), rng);
        sycl::buffer<int, 1> b_buff(vec_b.data(), rng);
        sycl::buffer<int, 1> c_buff(vec_c.data(), rng);

        queue.submit([&](sycl::handler &cgh) {
            auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buff.get_access<sycl::access::mode::read>(cgh);
            auto c_acc = c_buff.get_access<sycl::access::mode::read_write>(cgh);

            auto kernel = [=](sycl::id<1> id) {
                c_acc[id] = a_acc[id] + b_acc[id];
            };
            cgh.parallel_for<class VectorAdd>(rng, kernel);
        });
    }
    for (const auto &c:vec_c) {
        std::cout << c << ' ';
    } // 6 8 10 12
    std::cout << std::endl;

}
