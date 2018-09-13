#pragma once

#include <cassert>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace basync {

template <typename Signature>
class unique_function;

template <typename R, typename... Args>
class unique_function<R(Args...)>
{
private:
  template <typename F, typename Alloc>
  struct func_impl;

public:
  unique_function() = default;

  template <typename Alloc, typename F>
  unique_function(std::allocator_arg_t, const Alloc& alloc, F&& f)
  {
    using impl_type = func_impl<std::decay_t<F>, Alloc>;
    typename std::allocator_traits<Alloc>::template rebind_alloc<impl_type> a(alloc);
    using traits = std::allocator_traits<decltype(a)>;

    // Use RAII to deallocate memory when the constructor throws
    struct mem_handle
    {
      ~mem_handle()
      {
        if (mem)
          traits::deallocate(alloc, mem, 1);
      }

      impl_type* release()
      {
        return std::exchange(mem, nullptr);
      }

      impl_type* get()
      {
        return mem;
      }

      decltype(a)& alloc;
      impl_type*   mem;
    } mem{a, traits::allocate(a, 1)};

    traits::construct(a, mem.get(), alloc, std::forward<F>(f));
    this->f = mem.release();
  }

  template <typename F>
  unique_function(F&& f)
    : unique_function(std::allocator_arg, std::allocator<void>(), std::forward<F>(f))
  {
  }

  ~unique_function()
  {
    if (f)
      f->destroy();
  }

  unique_function(unique_function&& rhs) noexcept
    : f(std::exchange(rhs.f, nullptr))
  {
  }

  unique_function& operator=(unique_function&& rhs) noexcept
  {
    if (f)
      f->destroy();
    f = std::exchange(rhs.f, nullptr);
    return *this;
  }

  R operator()(Args... args) const
  {
    if (!f)
      throw std::bad_function_call();
    return f->invoke(std::forward<Args>(args)...);
  }

  constexpr explicit operator bool() const
  {
    return static_cast<bool>(f);
  }

private:
  struct func_iface
  {
    virtual void destroy() noexcept   = 0;
    virtual R    invoke(Args... args) = 0;
  };

  // TODO: do something with SBO here instead
  func_iface* f = nullptr;

  template <typename F, typename Alloc>
  class func_impl final : public func_iface, private Alloc
  {
  public:
    func_impl(const Alloc& alloc, const F& f) noexcept(noexcept(F(f)))
      : Alloc(alloc)
      , f(f)
    {
    }

    func_impl(const Alloc& alloc, F&& f) noexcept(noexcept(F(std::move(f))))
      : Alloc(alloc)
      , f(std::move(f))
    {
    }

    void destroy() noexcept override
    {
      typename std::allocator_traits<Alloc>::template rebind_alloc<func_impl> a(*this);
      using traits = std::allocator_traits<decltype(a)>;

      // Use RAII to deallocate memory, even when the destructor throws
      struct mem_handle
      {
        ~mem_handle()
        {
          traits::deallocate(alloc, mem, 1);
        }

        decltype(a)& alloc;
        func_impl*   mem;
      } mem{a, this};

      traits::destroy(a, this);
    }

    R invoke(Args... args) override
    {
      return f(std::forward<Args>(args)...);
    }

  private:
    F f;
  };
};

namespace detail {
template <typename>
struct function_guide_helper
{
};

template <typename R, typename F, bool is_noexcept, typename... Args>
struct function_guide_helper<R (F::*)(Args...) noexcept(is_noexcept)>
{
  using type = R(Args...);
};

template <typename R, typename F, bool is_noexcept, typename... Args>
struct function_guide_helper<R (F::*)(Args...) & noexcept(is_noexcept)>
{
  using type = R(Args...);
};

template <typename R, typename F, bool is_noexcept, typename... Args>
struct function_guide_helper<R (F::*)(Args...) const noexcept(is_noexcept)>
{
  using type = R(Args...);
};

template <typename R, typename F, bool is_noexcept, typename... Args>
struct function_guide_helper<R (F::*)(Args...) const& noexcept(is_noexcept)>
{
  using type = R(Args...);
};
}

template <typename R, typename... Args>
unique_function(R (*)(Args...)) -> unique_function<R(Args...)>;

template <
  typename F,
  typename Signature = typename detail::function_guide_helper<decltype(&F::operator())>::type>
unique_function(F) -> unique_function<Signature>;

template <typename T>
class promise;
template <typename T>
class future;

class executor
{
public:
  // TODO: can we do this without having a virtual at all?
  virtual void queue(unique_function<void()> func) = 0;
};

executor& default_executor();

template <typename F>
auto async(F&& func, executor& exec = default_executor()) -> future<decltype(func())>;

// clang-format off
struct promise_exception : std::exception {};
struct broken_promise : std::exception {};
struct promise_already_satisfied : promise_exception {};
struct no_future_state : promise_exception {};
struct future_not_retrieved : promise_exception {};
struct future_already_retrieved : promise_exception {};
// clang-format on

namespace detail {
template <typename T>
struct future_factory;
}

template <typename T>
class future
{
  // clang-format off
  struct void_t {};
  // clang-format on
  using storage_t = std::conditional_t<std::is_void_v<T>, void_t, T>;

public:
  future() = default;
  future(future&& rhs) noexcept(std::is_nothrow_move_constructible_v<storage_t>)
  {
    while (true)
    {
      std::unique_lock lf(rhs.m);
      if (rhs.p)
      {
        std::unique_lock lp(rhs.p->m, std::try_to_lock);
        if (!lp)
        {
          // TODO: figure out if this attempt at avoiding dead lock risks live lock instead
          std::this_thread::yield();
          continue;
        }

        p = std::exchange(rhs.p, nullptr);
        assert(p->f == &rhs);
        p->f = this;
      }

      continuation = std::move(rhs.continuation);
      storage      = std::exchange(rhs.storage, {});
      retrieved    = std::exchange(rhs.retrieved, false);
      break;
    }
  }

  future& operator=(future&& rhs) noexcept(std::is_nothrow_move_constructible_v<storage_t>)
  {
    while (true)
    {
      std::unique_lock lf(m);
      if (p)
      {
        std::unique_lock lp(p->m, std::try_to_lock);
        if (!lp)
        {
          // TODO: figure out if this attempt at avoiding dead lock risks live lock instead
          std::this_thread::yield();
          continue;
        }

        p->f = nullptr;
        p    = nullptr;
      }
      break;
    }

    while (true)
    {
      std::unique_lock lf(rhs.m);
      if (rhs.p)
      {
        std::unique_lock lp(rhs.p->m, std::try_to_lock);
        if (!lp)
        {
          // TODO: figure out if this attempt at avoiding dead lock risks live lock instead
          std::this_thread::yield();
          continue;
        }

        p = std::exchange(rhs.p, nullptr);
        assert(p->f == &rhs);
        p->f = this;
      }

      continuation = std::move(rhs.continuation);
      storage      = std::exchange(rhs.storage, {});
      retrieved    = std::exchange(rhs.retrieved, false);
      break;
    }

    return *this;
  }

  ~future()
  {
    while (true)
    {
      std::unique_lock lf(m);
      if (p)
      {
        std::unique_lock lp(p->m, std::try_to_lock);
        if (!lp)
        {
          // TODO: figure out if this attempt at avoiding dead lock risks live lock instead
          std::this_thread::yield();
          continue;
        }

        p->f = nullptr;
        p    = nullptr;
      }
      break;
    }
  }

  T get() &&
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return !std::holds_alternative<std::monostate>(storage); });
    if (std::exchange(retrieved, true))
      throw no_future_state();
    if (auto eptr = std::get_if<std::exception_ptr>(&storage); eptr != nullptr)
      std::rethrow_exception(*eptr);
    if constexpr (std::is_void_v<T>)
      std::get<future<T>::void_t>(storage);
    else
      return std::move(std::get<T>(storage));
  }

  bool is_ready() const
  {
    std::unique_lock<std::mutex> lk(m);
    return !std::holds_alternative<std::monostate>(storage);
  }

  template <typename F>
  auto
  then(F&&       func,
       executor& exec = default_executor()) && -> future<decltype(func(std::declval<future<T>>()))>;

private:
  friend class promise<T>;
  friend struct detail::future_factory<T>;

  unique_function<void(future<T>&&)>                          continuation;
  std::variant<std::monostate, storage_t, std::exception_ptr> storage;
  bool                                                        retrieved = false;
  mutable std::mutex                                          m;
  mutable std::condition_variable                             cv;
  promise<T>*                                                 p = nullptr;
};

namespace detail {
template <typename T>
struct future_factory
{
  static future<T> make_ready(T value)
  {
    future<T> f;
    f.storage = std::forward<T>(value);
    return f;
  }

  static future<T> make_exceptional(std::exception_ptr eptr)
  {
    future<T> f;
    f.storage = std::move(eptr);
    return f;
  }
};

template <>
struct future_factory<void>
{
  static future<void> make_ready()
  {
    future<void> f;
    f.storage.emplace<future<void>::void_t>();
    return f;
  }

  static future<void> make_exceptional(std::exception_ptr eptr)
  {
    future<void> f;
    f.storage = std::move(eptr);
    return f;
  }
};
}

template <typename T>
future<T> make_ready_future(T value)
{
  return detail::future_factory<T>::make_ready(std::forward<T>(value));
}

inline future<void> make_ready_future()
{
  return detail::future_factory<void>::make_ready();
}

template <typename T>
future<T> make_exceptional_future(std::exception_ptr eptr)
{
  return detail::future_factory<T>::make_exceptional(std::move(eptr));
}

template <typename T>
class promise
{
public:
  future<T> get_future()
  {
    if (std::exchange(retrieved, true))
      throw future_already_retrieved();
    future<T> f;
    f.p     = this;
    this->f = &f;
    return f;
  }

  void set_value(T&& value)
  {
    if (!retrieved)
      throw future_not_retrieved();
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();

    std::unique_lock lp(m);
    if (f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p = nullptr;
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation([&value] {
          future<T> f;
          f.storage = std::move(value);
          return f;
        }());
      }
      else
      {
        f->storage = std::move(value);
        f->cv.notify_all();
      }
      f = nullptr;
    }
  }

  void set_value(const T& value)
  {
    if (!retrieved)
      throw future_not_retrieved();
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();

    std::unique_lock lp(m);
    if (f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p = nullptr;
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation([&value] {
          future<T> f;
          f.storage = value;
          return f;
        }());
      }
      else
      {
        f->storage = value;
        f->cv.notify_all();
      }
      f = nullptr;
    }
  }

  void set_error(std::exception_ptr eptr)
  {
    if (!retrieved)
      throw future_not_retrieved();
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();

    std::unique_lock lp(m);
    if (f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p = nullptr;
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation(make_exceptional_future<T>(std::move(eptr)));
      }
      else
      {
        f->storage = std::move(eptr);
        f->cv.notify_all();
      }
      f = nullptr;
    }
  }

  promise() = default;
  promise(promise&& rhs) noexcept
    : retrieved(std::exchange(rhs.retrieved, false))
    , satisfied(std::exchange(rhs.satisfied, false))
  {
    std::unique_lock lt(m);
    std::unique_lock lp(rhs.m);
    if (rhs.f)
    {
      std::unique_lock lf(rhs.f->m);
      f = std::exchange(rhs.f, nullptr);
      assert(f->p == &rhs);
      f->p = this;
    }
  }

  promise& operator=(promise&& rhs) noexcept
  {
    std::unique_lock lt(m);
    std::unique_lock lp(rhs.m);
    if (rhs.f)
    {
      std::unique_lock lf(rhs.f->m);
      f = std::exchange(rhs.f, nullptr);
      assert(f->p == &rhs);
      f->p = this;
    }
    retrieved = std::exchange(rhs.retrieved, false);
    satisfied = std::exchange(rhs.satisfied, false);
    return *this;
  }

  ~promise()
  {
    std::unique_lock lp(m);
    if (!satisfied && f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p      = nullptr;
      auto eptr = std::make_exception_ptr(broken_promise());
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation(make_exceptional_future<T>(std::move(eptr)));
      }
      else
      {
        f->storage = std::move(eptr);
        f->cv.notify_all();
      }
      f = nullptr;
    }
    assert(satisfied || !f);
  }

private:
  friend class future<T>;

  future<T>* f         = nullptr;
  bool       retrieved = false;
  bool       satisfied = false;
  std::mutex m;
};

template <>
class promise<void>
{
public:
  future<void> get_future()
  {
    if (std::exchange(retrieved, true))
      throw future_already_retrieved();
    future<void> f;
    f.p     = this;
    this->f = &f;
    return f;
  }

  void set_value()
  {
    if (!retrieved)
      throw future_not_retrieved();
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();

    std::unique_lock lp(m);
    if (f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p = nullptr;
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation([] {
          future<void> f;
          f.storage.emplace<future<void>::void_t>();
          return f;
        }());
      }
      else
      {
        f->storage.emplace<future<void>::void_t>();
        f->cv.notify_all();
      }
      f = nullptr;
    }
  }

  void set_error(std::exception_ptr eptr)
  {
    if (!retrieved)
      throw future_not_retrieved();
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();

    std::unique_lock lp(m);
    if (f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p = nullptr;
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation(make_exceptional_future<void>(std::move(eptr)));
      }
      else
      {
        f->storage = std::move(eptr);
        f->cv.notify_all();
      }
      f = nullptr;
    }
  }

  promise() = default;
  promise(promise&& rhs) noexcept
    : retrieved(std::exchange(rhs.retrieved, false))
    , satisfied(std::exchange(rhs.satisfied, false))
  {
    std::unique_lock lt(m);
    std::unique_lock lp(rhs.m);
    if (rhs.f)
    {
      std::unique_lock lf(rhs.f->m);
      f = std::exchange(rhs.f, nullptr);
      assert(f->p == &rhs);
      f->p = this;
    }
  }

  promise& operator=(promise&& rhs) noexcept
  {
    std::unique_lock lt(m);
    std::unique_lock lp(rhs.m);
    if (rhs.f)
    {
      std::unique_lock lf(rhs.f->m);
      f = std::exchange(rhs.f, nullptr);
      assert(f->p == &rhs);
      f->p = this;
    }
    retrieved = std::exchange(rhs.retrieved, false);
    satisfied = std::exchange(rhs.satisfied, false);
    return *this;
  }

  ~promise()
  {
    std::unique_lock lp(m);
    if (!satisfied && f)
    {
      std::unique_lock lf(f->m);
      assert(std::holds_alternative<std::monostate>(f->storage));
      f->p      = nullptr;
      auto eptr = std::make_exception_ptr(broken_promise());
      if (auto continuation = std::move(f->continuation); continuation)
      {
        f->retrieved = true;
        continuation(make_exceptional_future<void>(std::move(eptr)));
      }
      else
      {
        f->storage = std::move(eptr);
        f->cv.notify_all();
      }
      f = nullptr;
    }
    assert(satisfied || !f);
  }

private:
  friend class future<void>;
  future<void>* f         = nullptr;
  bool          retrieved = false;
  bool          satisfied = false;
  std::mutex    m;
};

template <typename T>
future<std::vector<future<T>>> when_all(std::vector<future<T>> futures,
                                        executor&              exec = default_executor())
{
  struct when_all_processor
  {
    std::vector<future<T>> futures;
    executor&              exec;
    size_t                 cur = 0;

    std::vector<future<T>> operator()(future<T> f)
    {
      futures[cur++] = std::move(f);
      return (*this)().get();
    }

    future<std::vector<future<T>>> operator()()
    {
      if (cur == futures.size())
        return make_ready_future(std::move(futures));

      return std::move(futures[cur]).then(std::move(*this), exec);
    }
  };

  return when_all_processor{std::move(futures), exec}();
}

template <typename T>
future<std::vector<future<T>>> when_any(std::vector<future<T>> futures,
                                        executor&              exec = default_executor())
{
  if (futures.empty())
    return make_ready_future(std::move(futures));

  struct when_any_holder
  {
    std::vector<future<T>>          futures;
    std::mutex                      m;
    promise<std::vector<future<T>>> prom;
    bool                            value_set = false;

    when_any_holder(std::vector<future<T>> futures) noexcept(
      std::is_nothrow_move_constructible_v<std::vector<future<T>>>)
      : futures(std::move(futures))
    {
    }
  };

  auto inst = std::make_shared<when_any_holder>(std::move(futures));
  auto rv   = inst->prom.get_future();
  for (auto& f : inst->futures)
  {
    f = std::move(f).then([inst, &exec](future<T> f) mutable {
      std::lock_guard lock(inst->m);
      if (!std::exchange(inst->value_set, true))
      {
        exec.queue([inst = std::move(inst)] {
          std::lock_guard lock(inst->m);
          inst->prom.set_value(std::move(inst->futures));
        });
      }
      return std::move(f).get();
    });
  }

  return rv;
}

template <typename T>
template <typename F>
auto future<T>::then(F&&       func,
                     executor& exec) && -> future<decltype(func(std::declval<future<T>>()))>
{
  using R = decltype(func(std::declval<future<T>>()));

  promise<R> value;
  auto       rv = value.get_future();

  std::unique_lock lk(m);
  if (retrieved || continuation)
    throw no_future_state();

  continuation =
    [&exec, value = std::move(value), func = std::forward<F>(func)](future<T>&& f) mutable {
      exec.queue([f = std::move(f), value = std::move(value), func = std::move(func)]() mutable {
        try
        {
          if constexpr (std::is_void_v<R>)
          {
            std::forward<F>(func)(std::move(f));
            value.set_value();
          }
          else
          {
            value.set_value(std::forward<F>(func)(std::move(f)));
          }
        }
        catch (...)
        {
          value.set_error(std::current_exception());
        }
      });
    };

  if (!std::holds_alternative<std::monostate>(storage))
  {
    lk.unlock();
    continuation(std::move(*this));
  }
  return rv;
}

template <typename F>
auto async(F&& func, executor& exec) -> future<decltype(func())>
{
  using R = decltype(func());

  promise<R> value;
  auto       rv = value.get_future();
  exec.queue([value = std::move(value), func = std::forward<F>(func)]() mutable {
    try
    {
      if constexpr (std::is_void_v<R>)
      {
        func();
        value.set_value();
      }
      else
      {
        value.set_value(func());
      }
    }
    catch (...)
    {
      value.set_error(std::current_exception());
    }
  });
  return rv;
}
}
