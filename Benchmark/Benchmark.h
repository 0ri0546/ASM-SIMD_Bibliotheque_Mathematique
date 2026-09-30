#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <print>
#include <string>
#include <type_traits>
#include <vector>

#include <intrin.h>

namespace bench_detail {
	inline const void* escape_ptr = nullptr;
	inline std::size_t benchmarks_count = 1;

	// noinline store: the compiler must assume someone reads *p
	__declspec(noinline) inline void Escape(const void* p) { escape_ptr = p; }
}


// Forces pending writes to memory to be considered observable
inline void ClobberMemory() { _ReadWriteBarrier(); }

// Forces `value` to exist in memory with its final value
template <typename T>
inline void DoNotOptimize(const T& value) {
	bench_detail::Escape(&value);
	ClobberMemory();
}

struct alignas(8) bench_data {
	const char* name = {};
	double op_s = {};
	double ns_op = {};
	double err_pct = {};
	std::uint64_t cyc_op = {};
	double total = {};
};

namespace bench_detail {
	template <typename Fn>
	inline void Run(Fn& fn) {
		if constexpr (std::is_void_v<std::invoke_result_t<Fn&>>) {
			fn();
			ClobberMemory();
		}
		else {
			auto r = fn();
			DoNotOptimize(r);   // the value, not its address
		}
	}
}

template <typename Fn>
void Benchmark(const std::string& name, Fn&& fn,
	std::size_t iterations = 1'000'000,
	std::size_t samples = 10)
{
	using namespace std::chrono;
	bench_data data{};
	data.name = name.c_str();

	const std::size_t per_sample = std::max<std::size_t>(1, iterations / samples);

	// time-based warmup (~20 ms) so the CPU reaches steady clocks before sampling
	{
		auto w0 = steady_clock::now();
		while (steady_clock::now() - w0 < milliseconds(20))
			for (int i = 0; i < 1000; ++i) bench_detail::Run(fn);
	}

	std::vector<double> ns_per_op;
	ns_per_op.reserve(samples);
	double total_ns = 0.0;
	std::uint64_t total_cycles = 0;

	for (std::size_t s = 0; s < samples; ++s) {
		auto c0 = __rdtsc();
		auto t0 = steady_clock::now();
		for (std::size_t i = 0; i < per_sample; ++i) bench_detail::Run(fn);
		auto t1 = steady_clock::now();
		auto c1 = __rdtsc();

		double ns = static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count());
		ns_per_op.push_back(ns / per_sample);
		total_ns += ns;
		total_cycles += (c1 - c0);
	}

	double mean = 0.0;
	for (double v : ns_per_op) mean += v;
	mean /= ns_per_op.size();

	double var = 0.0;
	for (double v : ns_per_op) var += (v - mean) * (v - mean);
	var /= (ns_per_op.size() > 1 ? ns_per_op.size() - 1 : 1);

	data.ns_op = mean;
	data.op_s = 1.0e9 / mean;
	data.err_pct = mean > 0 ? 100.0 * std::sqrt(var) / mean : 0.0;
	data.cyc_op = total_cycles / (per_sample * samples);
	data.total = total_ns * 1.0e-6; // ms

	std::println("{:<4} | {:<40} | {:>12.4e} | {:>12.4f} | {:>12.4f} | {:>12} | {:>12.4f}",
		bench_detail::benchmarks_count,
		data.name,
		data.op_s,
		data.ns_op,
		data.err_pct,
		data.cyc_op,
		data.total
	);
	std::println("--------------------------------------------------------------------------------------------------------------------------");
	bench_detail::benchmarks_count++;
}
