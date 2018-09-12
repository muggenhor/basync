#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
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

  unique_function operator=(unique_function&& rhs) noexcept
  {
    if (f)
      f->destroy();
    f = std::exchange(rhs.f, nullptr);
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
class shared_state;
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

executor* default_executor();

template <typename F>
auto async(F&& func, executor* exec = default_executor()) -> future<decltype(func())>;

// TODO: make this (functionally) unique_future
template <typename T>
class future
{
public:
  T    get() const;
  bool is_ready() const;
  template <typename F>
  auto then(F&& func, executor* exec = default_executor())
    -> future<decltype(func(std::declval<future<T>>()))>;
//private:
  future(std::shared_ptr<shared_state<T>> state);
  friend class promise<T>;
  std::shared_ptr<shared_state<T>> state;
};

// TODO: missing broken_promise here
struct promise_exception : public std::exception
{
};

struct promise_already_satisfied : promise_exception
{
};

template <typename T>
class shared_state
{
public:
  shared_state()
    : storage(NULL)
  {
  }
  ~shared_state()
  {
    delete storage;
  }
  void set(T&& value)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      throw promise_already_satisfied();
    storage   = new T(std::move(value));
    value_set = true;
    cv.notify_all();
    for (auto& p : cbs)
      async(std::move(p.second), p.first);
    cbs.clear();
  }
  void set(const T& value)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      throw promise_already_satisfied();
    storage   = new T(value);
    value_set = true;
    cv.notify_all();
    for (auto& p : cbs)
      async(std::move(p.second), p.first);
    cbs.clear();
  }
  void set_error(std::exception_ptr eptr)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      throw promise_already_satisfied();
    this->eptr = eptr;
    value_set  = true;
    cv.notify_all();
    for (auto& p : cbs)
      async(std::move(p.second), p.first);
    cbs.clear();
  }
  T get()
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return value_set; });
    if (eptr)
      std::rethrow_exception(eptr);
    return *storage;
  }
  void then(unique_function<void()> cb, executor* exec)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      async(std::move(cb), exec);
    else
      cbs.push_back(std::make_pair(exec, std::move(cb)));
  }
  std::vector<std::pair<executor*, unique_function<void()>>> cbs;
  T*                                                         storage;
  std::exception_ptr                                         eptr;
  mutable std::mutex                                         m;
  mutable std::condition_variable                            cv;
  bool                                                       value_set = false;
};

template <>
struct shared_state<void>
{
public:
  void set()
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      throw promise_already_satisfied();
    value_set = true;
    cv.notify_all();
    for (auto& p : cbs)
      async(std::move(p.second), p.first);
    cbs.clear();
  }
  void set_error(std::exception_ptr eptr)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      throw promise_already_satisfied();
    this->eptr = eptr;
    value_set  = true;
    cv.notify_all();
    for (auto& p : cbs)
      async(std::move(p.second), p.first);
    cbs.clear();
  }
  void get() const
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return value_set; });
    if (eptr)
      std::rethrow_exception(eptr);
  }
  void then(unique_function<void()> cb, executor* exec)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      async(std::move(cb), exec);
    else
      cbs.push_back(std::make_pair(exec, std::move(cb)));
  }
  std::vector<std::pair<executor*, unique_function<void()>>> cbs;
  std::exception_ptr                                         eptr;
  mutable std::mutex                                         m;
  mutable std::condition_variable                            cv;
  bool                                                       value_set = false;
};

template <typename T>
future<T> make_ready_future(T value)
{
  std::shared_ptr<shared_state<T>> state = std::make_shared<shared_state<T>>();
  future<T>                        f(state);
  state->set(std::move(value));
  return f;
}

template <typename T>
class promise;

template <typename T>
T future<T>::get() const
{
  return state->get();
}

template <typename T>
bool future<T>::is_ready() const
{
  return state->value_set;
}

template <typename T>
future<T>::future(std::shared_ptr<shared_state<T>> state)
  : state(state)
{
}

template <typename T>
class promise
{
public:
  // TODO:
  //   * store shared state by value inside the future
  //   * make having called get_future() a precondition for the set_XXX functions
  future<T> get_future()
  {
    return future<T>(state);
  }
  void set_value(T&& value)
  {
    state->set(std::forward<T>(value));
  }
  void set_value(const T& value)
  {
    state->set(value);
  }
  void set_error(std::exception_ptr eptr)
  {
    state->set_error(eptr);
  }
  std::shared_ptr<shared_state<T>> state = std::make_shared<shared_state<T>>();
};

template <>
class promise<void>
{
public:
  future<void> get_future()
  {
    return future<void>(state);
  }
  void set_value()
  {
    state->set();
  }
  void set_error(std::exception_ptr eptr)
  {
    state->set_error(eptr);
  }
  std::shared_ptr<shared_state<void>> state = std::make_shared<shared_state<void>>();
};

template <typename T>
struct all
{
  bool operator()(const std::vector<future<T>>& futures)
  {
    for (const auto& f : futures)
    {
      if (!f.is_ready())
        return false;
    }
    return true;
  }
};

template <typename T>
struct any
{
  bool operator()(const std::vector<future<T>>& futures)
  {
    for (const auto& f : futures)
    {
      if (f.is_ready())
        return true;
    }
    return false;
  }
};

template <template <typename> class P, typename T>
future<std::vector<future<T>>> when(const std::vector<future<T>>& futures)
{
  struct when_holder
  {
    std::vector<future<T>>          futures;
    P<T>                            pred;
    std::mutex                      m;
    promise<std::vector<future<T>>> prom;
    bool                            value_set;
    when_holder(std::vector<future<T>> v)
      : futures(v)
    {
      value_set = false;
    }
    void updatePromise()
    {
      std::lock_guard<std::mutex> lock(m);
      if (!value_set && pred(futures))
      {
        prom.set_value(std::move(futures));
        value_set = true;
      }
    }
  };
  std::shared_ptr<when_holder> inst = std::make_shared<when_holder>(futures);
  for (auto& f : inst->futures)
  {
    f.then([inst](const future<T>&) { inst->updatePromise(); });
  }
  future<std::vector<future<T>>> rv = inst->prom.get_future();
  inst->updatePromise();
  return rv;
}

namespace detail {
template <typename T, typename U>
struct then_impl
{
  template <typename F>
  static future<U> impl(future<T> t, F&& func, executor* exec)
  {
    promise<U> value;
    auto       rv = value.get_future();
    t.state->then(
      [t, value = std::move(value), func = std::forward<F>(func)]() mutable {
        try
        {
          value.set_value(func(t));
        }
        catch (...)
        {
          value.set_error(std::current_exception());
        }
      },
      exec);
    return rv;
  }
};

template <typename T>
struct then_impl<T, void>
{
  template <typename F>
  static future<void> impl(future<T> t, F&& func, executor* exec)
  {
    promise<void> value;
    auto          rv = value.get_future();
    t.state->then(
      [t, value = std::move(value), func = std::forward<F>(func)]() mutable {
        try
        {
          func(t);
          value.set_value();
        }
        catch (...)
        {
          value.set_error(std::current_exception());
        }
      },
      exec);
    return rv;
  }
};
}

template <typename T>
template <typename F>
auto future<T>::then(F&& func, executor* exec) -> future<decltype(func(std::declval<future<T>>()))>
{
  return detail::then_impl<T, decltype(func(std::declval<future<T>>()))>::impl(
    *this, std::forward<F>(func), exec);
}

namespace detail {
template <typename U>
struct async_impl
{
  template <typename F>
  static future<U> impl(executor* exec, F&& func)
  {
    promise<U> value;
    auto       rv = value.get_future();
    exec->queue([value = std::move(value), func = std::forward<F>(func)]() mutable {
      try
      {
        value.set_value(func());
      }
      catch (...)
      {
        value.set_error(std::current_exception());
      }
    });
    return rv;
  }
};

template <>
struct async_impl<void>
{
  template <typename F>
  static future<void> impl(executor* exec, F&& func)
  {
    promise<void> value;
    auto          rv = value.get_future();
    exec->queue([value = std::move(value), func = std::forward<F>(func)]() mutable {
      try
      {
        func();
        value.set_value();
      }
      catch (...)
      {
        value.set_error(std::current_exception());
      }
    });
    return rv;
  }
};
}

template <typename F>
auto async(F&& func, executor* exec) -> future<decltype(func())>
{
  return detail::async_impl<decltype(func())>::impl(exec, std::forward<F>(func));
}
}
