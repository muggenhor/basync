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

template <typename T>
class shared_state;
template <typename T>
class promise;
template <typename T>
class future;

class executor
{
public:
  // TODO: accept something equivalent to unique_function<void()> instead
  // NOTE: unique_function should not require stored functors to be copy constructible (only moveable)
  //       furthermore it should be implicitly constructible from std::function<void()>
  // TODO: can we do this without having a virtual at all?
  virtual void queue(std::function<void()> func) = 0;
  template <typename T>
  auto async(T func) -> future<decltype(func())>;
};

executor* default_executor();

template <typename T>
auto async(T func) -> future<decltype(func())>
{
  return default_executor()->async(func);
}

// TODO: make this (functionally) unique_future
template <typename T>
class future
{
public:
  T    get() const;
  bool is_ready() const;
  template <typename U>
  auto then(U func, executor* exec = default_executor()) -> future<decltype(func(*(future<T>*)0))>;
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
    for (const auto& p : cbs)
      p.first->async(p.second);
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
    for (const auto& p : cbs)
      p.first->async(p.second);
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
    for (const auto& p : cbs)
      p.first->async(p.second);
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
  void then(std::function<void()> cb, executor* exec)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      exec->async(cb);
    else
      cbs.push_back(std::make_pair(exec, cb));
  }
  std::vector<std::pair<executor*, std::function<void()>>> cbs;
  T*                                                       storage;
  std::exception_ptr                                       eptr;
  mutable std::mutex                                       m;
  mutable std::condition_variable                          cv;
  bool                                                     value_set = false;
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
    for (const auto& p : cbs)
      p.first->async(p.second);
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
    for (const auto& p : cbs)
      p.first->async(p.second);
    cbs.clear();
  }
  void get() const
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return value_set; });
    if (eptr)
      std::rethrow_exception(eptr);
  }
  void then(std::function<void()> cb, executor* exec)
  {
    std::unique_lock<std::mutex> lk(m);
    if (value_set)
      exec->async(cb);
    else
      cbs.push_back(std::make_pair(exec, cb));
  }
  std::vector<std::pair<executor*, std::function<void()>>> cbs;
  std::exception_ptr                                       eptr;
  mutable std::mutex                                       m;
  mutable std::condition_variable                          cv;
  bool                                                     value_set = false;
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

template <typename T, typename U>
struct then_impl
{
  static future<U> impl(future<T> t, std::function<U(const future<T>&)> func, executor* exec)
  {
    std::shared_ptr<promise<U>> value = std::make_shared<promise<U>>();
    future<U>                   rv    = value->get_future();
    t.state->then(
      [t, value, func] {
        try
        {
          value->set_value(func(t));
        }
        catch (...)
        {
          value->set_error(std::current_exception());
        }
      },
      exec);
    return std::move(rv);
  }
};

template <typename T>
struct then_impl<T, void>
{
  static future<void> impl(future<T> t, std::function<void(const future<T>&)> func, executor* exec)
  {
    std::shared_ptr<promise<void>> value = std::make_shared<promise<void>>();
    future<void>                   rv    = value->get_future();
    t.state->then(
      [t, value, func] {
        try
        {
          func(t);
          value->set_value();
        }
        catch (...)
        {
          value->set_error(std::current_exception());
        }
      },
      exec);
    return std::move(rv);
  }
};

template <typename T>
template <typename U>
auto future<T>::then(U func, executor* exec) -> future<decltype(func(*(future<T>*)0))>
{
  std::function<decltype(func(*(future<T>*)0))(const future<T>&)> f = func;
  return then_impl<T, decltype(func(*(future<T>*)0))>::impl(*this, f, exec);
}

template <typename U>
struct async_impl
{
  static future<U> impl(executor* exec, std::function<U()> func)
  {
    std::shared_ptr<promise<U>> value = std::make_shared<promise<U>>();
    future<U>                   rv    = value->get_future();
    exec->queue([value, func] {
      try
      {
        value->set_value(func());
      }
      catch (...)
      {
        value->set_error(std::current_exception());
      }
    });
    return std::move(rv);
  }
};

template <>
struct async_impl<void>
{
  static future<void> impl(executor* exec, std::function<void()> func)
  {
    std::shared_ptr<promise<void>> value = std::make_shared<promise<void>>();
    future<void>                   rv    = value->get_future();
    exec->queue([value, func] {
      try
      {
        func();
        value->set_value();
      }
      catch (...)
      {
        value->set_error(std::current_exception());
      }
    });
    return std::move(rv);
  }
};

template <typename T>
auto executor::async(T func) -> future<decltype(func())>
{
  std::function<decltype(func())()> f = func;
  return async_impl<decltype(func())>::impl(this, f);
}
}
