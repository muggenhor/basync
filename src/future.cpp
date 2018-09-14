#include <basync/future.hpp>
#include <atomic>
#include <queue>
#include <thread>
#include <vector>

namespace basync {

struct threadpool_executor : executor
{
  threadpool_executor()
  {
    for (size_t n = 0; n < std::max(std::thread::hardware_concurrency() * 2, 4u); ++n)
    {
      threads.emplace_back([this] { this->worker(); });
    }
  }
  ~threadpool_executor()
  {
    while (!tasks.empty())
    {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    shutdown.store(true);
    for (auto& t : threads)
    {
      t.join();
    }
  }
  void queue(unique_function<void()> func) override
  {
    std::lock_guard<std::mutex> l(m);
    tasks.push(std::move(func));
  }
  std::queue<unique_function<void()>> tasks;
  std::vector<std::thread>            threads;
  std::mutex                          m;
  std::atomic<bool>                   shutdown;
  void                                worker()
  {
    while (!shutdown.load())
    {
      bool                    foundTask = false;
      unique_function<void()> f;
      {
        std::lock_guard<std::mutex> l(m);
        if (!tasks.empty())
        {
          f = std::move(tasks.front());
          tasks.pop();
          foundTask = true;
        }
      }

      if (foundTask)
      {
        f();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    }
  }
};

executor& default_executor()
{
  static threadpool_executor exec;
  return exec;
}
}
