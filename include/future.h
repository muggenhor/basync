#pragma once

#include <cassert>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
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

executor& default_executor();

template <typename F>
auto async(F&& func, executor& exec = default_executor()) -> future<decltype(func())>;

namespace detail {
template <typename T>
struct future_factory;
}

template <typename T>
class future
{
public:
  future()         = default;
  future(future&&) = default;
  future& operator=(future&&) = default;

  T    get() &&;
  bool is_ready() const;
  template <typename F>
  auto
  then(F&&       func,
       executor& exec = default_executor()) && -> future<decltype(func(std::declval<future<T>>()))>;

private:
  future(std::shared_ptr<shared_state<T>> state);
  friend class promise<T>;
  friend struct detail::future_factory<T>;

  std::shared_ptr<shared_state<T>> state;
};

// clang-format off
struct promise_exception : std::exception {};
struct broken_promise : std::exception {};
struct promise_already_satisfied : promise_exception {};
struct no_future_state : promise_exception {};
struct future_not_retrieved : promise_exception {};
struct future_already_retrieved : promise_exception {};
// clang-format on

template <typename T>
class shared_state
{
public:
  void set(T&& value)
  {
    std::unique_lock<std::mutex> lk(m);
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();
    assert(std::holds_alternative<std::monostate>(storage));
    storage = std::move(value);
    cv.notify_all();
    if (auto cb = std::move(this->cb); cb)
      cb();
  }
  void set(const T& value)
  {
    std::unique_lock<std::mutex> lk(m);
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();
    assert(std::holds_alternative<std::monostate>(storage));
    storage = value;
    cv.notify_all();
    if (auto cb = std::move(this->cb); cb)
      cb();
  }
  void set_error(std::exception_ptr eptr)
  {
    std::unique_lock<std::mutex> lk(m);
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();
    assert(std::holds_alternative<std::monostate>(storage));
    storage = eptr;
    cv.notify_all();
    if (auto cb = std::move(this->cb); cb)
      cb();
  }
  T get() &&
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return !std::holds_alternative<std::monostate>(storage); });
    if (std::exchange(retrieved, true))
      throw no_future_state();
    if (auto eptr = std::get_if<std::exception_ptr>(&storage); eptr != nullptr)
      std::rethrow_exception(*eptr);
    return std::move(std::get<T>(storage));
  }
  template <typename F, typename = decltype(std::declval<F>()())>
  void then(F&& cb)
  {
    std::unique_lock<std::mutex> lk(m);
    if (retrieved || this->cb)
      throw no_future_state();
    visit(
      [&cb, this](auto&& arg) {
        using U = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<U, std::monostate>)
          this->cb = std::forward<F>(cb);
        else
          cb();
      },
      storage);
  }
  unique_function<void()>                             cb;
  std::variant<std::monostate, T, std::exception_ptr> storage;
  bool                                                retrieved = false;
  bool                                                satisfied = false;
  mutable std::mutex                                  m;
  mutable std::condition_variable                     cv;
};

template <>
struct shared_state<void>
{
public:
  void set()
  {
    std::unique_lock<std::mutex> lk(m);
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();
    assert(std::holds_alternative<std::monostate>(storage));
    storage.emplace<void_t>();
    cv.notify_all();
    if (auto cb = std::move(this->cb); cb)
      cb();
  }
  void set_error(std::exception_ptr eptr)
  {
    std::unique_lock<std::mutex> lk(m);
    if (std::exchange(satisfied, true))
      throw promise_already_satisfied();
    assert(std::holds_alternative<std::monostate>(storage));
    storage = eptr;
    cv.notify_all();
    if (auto cb = std::move(this->cb); cb)
      cb();
  }
  void get() &&
  {
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this] { return !std::holds_alternative<std::monostate>(storage); });
    if (std::exchange(retrieved, true))
      throw no_future_state();
    if (auto eptr = std::get_if<std::exception_ptr>(&storage); eptr != nullptr)
      std::rethrow_exception(*eptr);
    return static_cast<void>(std::move(std::get<void_t>(storage)));
  }
  template <typename F, typename = decltype(std::declval<F>()())>
  void then(F&& cb)
  {
    std::unique_lock<std::mutex> lk(m);
    if (retrieved || this->cb)
      throw no_future_state();
    visit(
      [&cb, this](auto&& arg) {
        using U = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<U, std::monostate>)
          this->cb = std::forward<F>(cb);
        else
          cb();
      },
      storage);
  }
  // clang-format off
  struct void_t {};
  // clang-format on
  unique_function<void()>                                  cb;
  std::variant<std::monostate, void_t, std::exception_ptr> storage;
  bool                                                     retrieved = false;
  bool                                                     satisfied = false;
  mutable std::mutex                                       m;
  mutable std::condition_variable                          cv;
};

namespace detail {
template <typename T>
struct future_factory
{
  static future<T> make_ready(T value)
  {
    auto state = std::make_shared<shared_state<T>>();
    state->set(std::forward<T>(value));
    return future<T>(std::move(state));
  }
};

template <>
struct future_factory<void>
{
  static future<void> make_ready()
  {
    auto state = std::make_shared<shared_state<void>>();
    state->set();
    return future<void>(std::move(state));
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
class promise;

template <typename T>
T future<T>::get() &&
{
  return std::move(*state).get();
}

template <typename T>
bool future<T>::is_ready() const
{
  return !std::holds_alternative<std::monostate>(state->storage);
}

template <typename T>
future<T>::future(std::shared_ptr<shared_state<T>> state)
  : state(std::move(state))
{
}

template <typename T>
class promise
{
public:
  // TODO:
  //   * store shared state by value inside the future
  future<T> get_future()
  {
    if (std::exchange(retrieved, true))
      throw future_already_retrieved();
    return future<T>(state);
  }
  void set_value(T&& value)
  {
    if (!retrieved)
      throw future_not_retrieved();
    state->set(std::forward<T>(value));
  }
  void set_value(const T& value)
  {
    if (!retrieved)
      throw future_not_retrieved();
    state->set(value);
  }
  void set_error(std::exception_ptr eptr)
  {
    if (!retrieved)
      throw future_not_retrieved();
    state->set_error(std::move(eptr));
  }
  promise()          = default;
  promise(promise&&) = default;
  promise& operator=(promise&&) = default;
  ~promise()
  {
    if (state && !state->satisfied)
      state->set_error(std::make_exception_ptr(broken_promise()));
  }

private:
  std::shared_ptr<shared_state<T>> state     = std::make_shared<shared_state<T>>();
  bool                             retrieved = false;
};

template <>
class promise<void>
{
public:
  future<void> get_future()
  {
    if (std::exchange(retrieved, true))
      throw future_already_retrieved();
    return future<void>(state);
  }
  void set_value()
  {
    if (!retrieved)
      throw future_not_retrieved();
    state->set();
  }
  void set_error(std::exception_ptr eptr)
  {
    if (!retrieved)
      throw future_not_retrieved();
    state->set_error(std::move(eptr));
  }
  promise()          = default;
  promise(promise&&) = default;
  promise& operator=(promise&&) = default;
  ~promise()
  {
    if (state && !state->satisfied)
      state->set_error(std::make_exception_ptr(broken_promise()));
  }

private:
  std::shared_ptr<shared_state<void>> state     = std::make_shared<shared_state<void>>();
  bool                                retrieved = false;
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
  auto       rv    = value.get_future();
  auto&      state = *this->state;
  state.then([&exec,
              t     = std::move(*this),
              value = std::move(value),
              func  = std::forward<F>(func)]() mutable {
    exec.queue([t = std::move(t), value = std::move(value), func = std::move(func)]() mutable {
      try
      {
        if constexpr (std::is_void_v<R>)
        {
          func(std::move(t));
          value.set_value();
        }
        else
        {
          value.set_value(func(std::move(t)));
        }
      }
      catch (...)
      {
        value.set_error(std::current_exception());
      }
    });
  });
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
