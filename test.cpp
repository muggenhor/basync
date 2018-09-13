#include <future.h>
#include <iostream>
#include <thread>
#include <vector>

int main()
{
  auto f = basync::async([] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << " " << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return "World";
  });

  auto x = std::move(f).then([](auto f) {
    std::cout << std::move(f).get() << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return '!';
  });
  std::cout << "Hello" << std::flush
            << std::move(basync::when_all([&x] {
                           std::vector<basync::future<char> > v;
                           v.emplace_back(std::move(x));
                           return v;
                         }())
                           .get()
                           .at(0))
                 .get()
            << '\n';
  basync::async([] { std::this_thread::sleep_for(std::chrono::milliseconds(750)); })
    .then([](auto) { std::cout << "Mooh!\n"; })
    .get();
  std::cout << [] { return basync::promise<const char*>().get_future(); }().get() << '\n';
}
