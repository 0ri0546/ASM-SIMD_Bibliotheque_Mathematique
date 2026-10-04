#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <format>
#include <functional>
#include <iostream>
#include <intrin.h>
#include <numeric>
#include <print>
#include <string>
#include <vector>

#pragma optimize("", off)
inline void DoNotOptimizeAwaySink(const void* value)
{
    (void)value;
}
#pragma optimize("", on)

template <typename T>
__forceinline inline void DoNotOptimizeAway(const T& value)
{
    DoNotOptimizeAwaySink(&value);
    _ReadWriteBarrier();
}

struct BenchmarkResult
{
    std::string name;

    std::size_t batchSize = 0;
    std::size_t iterations = 0;

    double operationsPerSecond = 0.0;
    double nsPerOperation = 0.0;
    double errorPercent = 0.0;
    double cyclesPerOperation = 0.0;
    double totalMs = 0.0;

    double speedup = 0.0;
};

inline std::size_t g_benchmarkIndex = 1;

inline std::vector<BenchmarkResult> g_referenceResults;
inline std::vector<BenchmarkResult> g_simdResults;

inline std::size_t g_batchSize = 1'000;
inline std::uint32_t g_seed = 0x12345678;

inline constexpr std::size_t BENCHMARK_ITERATIONS = 10;

inline constexpr std::size_t BENCHMARK_SAMPLES = 10;

inline constexpr std::size_t BENCHMARK_WARMUP = 3;

inline std::uint64_t ReadTSC()
{
    unsigned int aux = 0;
    return __rdtscp(&aux);
}

inline double Median(std::vector<double> values)
{
    if (values.empty())
        return 0.0;

    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 0)
    {
        return (values[middle - 1] + values[middle]) * 0.5;
    }

    return values[middle];
}

inline double IQR(std::vector<double> values)
{
    if (values.size() < 2)
        return 0.0;

    std::sort(values.begin(), values.end());

    const std::size_t q1 = values.size() / 4;
    const std::size_t q3 = (values.size() * 3) / 4;

    return values[q3] - values[q1];
}

inline double Mean(const std::vector<double>& values) {
    if (values.empty())
        return 0.0;

    return std::accumulate(
        values.begin(),
        values.end(),
        0.0
    ) / static_cast<double>(values.size());
}

inline double RelativeError(const std::vector<double>& values) {
    if (values.size() < 2)
        return 0.0;

    const double mean = Mean(values);

    if (mean <= 0.0)
        return 0.0;

    double variance = 0.0;

    for (const double value : values)
    {
        const double difference = value - mean;
        variance += difference * difference;
    }

    variance /= static_cast<double>(values.size() - 1);

    return 100.0 * std::sqrt(variance) / mean;
}

inline std::string GetCpuName() {
    int cpuInfo[4]{};
    char cpuName[49]{};

    __cpuid(cpuInfo, 0x80000000);

    const unsigned int maxId =
        static_cast<unsigned int>(cpuInfo[0]);

    if (maxId < 0x80000004)
        return "Unknown CPU";

    __cpuid(cpuInfo, 0x80000002);
    std::memcpy(cpuName, cpuInfo, sizeof(cpuInfo));

    __cpuid(cpuInfo, 0x80000003);
    std::memcpy(cpuName + 16, cpuInfo, sizeof(cpuInfo));

    __cpuid(cpuInfo, 0x80000004);
    std::memcpy(cpuName + 32, cpuInfo, sizeof(cpuInfo));

    return std::string(cpuName);
}

inline std::string GetISA()
{
#if defined(__AVX2__)
    return "AVX2";
#elif defined(__AVX__)
    return "AVX";
#elif defined(__SSE4_2__)
    return "SSE4.2";
#elif defined(__SSE4_1__)
    return "SSE4.1";
#elif defined(_M_X64)
    return "SSE2";
#else
    return "Unknown";
#endif
}

inline void BeginBenchmark(
    std::size_t batchSize,
    std::uint32_t seed)
{
    g_benchmarkIndex = 1;

    g_referenceResults.clear();
    g_simdResults.clear();

    g_batchSize = batchSize;
    g_seed = seed;

    std::ofstream file(
        "Benchmark.csv",
        std::ios::trunc
    );

    if (file.is_open())
    {
        file
            << "No.|Name|Batch|Iterations|op / s|ns / op|err %|cyc / op|total(ms)|Speedup\n";
    }

    std::println();
    std::println("==============================================================");
    std::println("                    SIMD BENCHMARK                            ");
    std::println("==============================================================");
    std::println("CPU        : {}", GetCpuName());
    std::println("ISA        : {}", GetISA());
    std::println("Batch      : {}", g_batchSize);
    std::println("Iterations : {}", BENCHMARK_ITERATIONS);
    std::println("Samples    : {}", BENCHMARK_SAMPLES);
    std::println("Warm-up    : {}", BENCHMARK_WARMUP);
    std::println("Seed       : {}", g_seed);
    std::println("Build      : Release x64");
    std::println("==============================================================");
    std::println();
}

inline void PrintBenchmarkHeader()
{
    std::println(
        "{:<4} | {:<34} | {:>10} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}",
        "No.",
        "Name",
        "Batch",
        "Iterations",
        "op / s",
        "ns / op",
        "err %",
        "cyc / op",
        "total(ms)"
    );

    std::println("{:-<165}", "");
}

inline void WriteBenchmarkCSV(
    const BenchmarkResult& result,
    std::size_t index)
{
    std::ofstream file(
        "Benchmark.csv",
        std::ios::app
    );

    if (!file.is_open())
        return;

    file << std::format(
        "{}|{}|{}|{}|{:.4e}|{:.4f}|{:.4f}|{:.0f}|{:.4f}|{:.3f}\n",
        index,
        result.name,
        result.batchSize,
        result.iterations,
        result.operationsPerSecond,
        result.nsPerOperation,
        result.errorPercent,
        result.cyclesPerOperation,
        result.totalMs,
        result.speedup
    );
}

inline void PrintBenchmarkResult(
    BenchmarkResult& result,
    std::vector<BenchmarkResult>& destination)
{
    const std::size_t index = g_benchmarkIndex++;

    const std::string speedup =
        result.speedup > 0.0
        ? std::format("{:.3f}x", result.speedup)
        : "-";

    std::println(
        "{:<4} | {:<34} | {:>10} | {:>12} | {:>12.4e} | {:>12.4f} | {:>12.4f} | {:>12.0f} | {:>12.4f}",
        index,
        result.name,
        result.batchSize,
        result.iterations,
        result.operationsPerSecond,
        result.nsPerOperation,
        result.errorPercent,
        result.cyclesPerOperation,
        result.totalMs
    );

    destination.push_back(result);

    WriteBenchmarkCSV(result, index);
}

template <typename Fn>
BenchmarkResult Benchmark(
    const std::string& name,
    Fn&& fn,
    std::size_t batchSize,
    std::size_t iterations = BENCHMARK_ITERATIONS,
    std::size_t samples = BENCHMARK_SAMPLES,
    std::size_t warmupIterations = BENCHMARK_WARMUP)
{
    BenchmarkResult result;

    result.name = name;
    result.batchSize = batchSize;
    result.iterations = iterations;

    for (std::size_t i = 0; i < warmupIterations; ++i) {
        fn();
    }

    std::vector<double> nsPerOperation;
    std::vector<double> cyclesPerOperation;

    nsPerOperation.reserve(samples);
    cyclesPerOperation.reserve(samples);

    double totalNanoseconds = 0.0;

    for (std::size_t sample = 0; sample < samples; ++sample)
    {
        const auto timeBegin =
            std::chrono::steady_clock::now();

        const std::uint64_t cycleBegin =
            ReadTSC();

        for (std::size_t i = 0; i < iterations; ++i) {
            fn();
        }

        const std::uint64_t cycleEnd =
            ReadTSC();

        const auto timeEnd =
            std::chrono::steady_clock::now();

        const double elapsedNanoseconds =
            static_cast<double>(
                std::chrono::duration_cast<
                std::chrono::nanoseconds
                >(timeEnd - timeBegin).count()
                );

        const double elapsedCycles =
            static_cast<double>(
                cycleEnd - cycleBegin
                );

        const double totalOperations =
            static_cast<double>(batchSize) *
            static_cast<double>(iterations);

        const double nsPerElement =
            elapsedNanoseconds /
            totalOperations;

        const double cyclesPerElement =
            elapsedCycles /
            totalOperations;

        nsPerOperation.push_back(nsPerElement);
        cyclesPerOperation.push_back(cyclesPerElement);

        totalNanoseconds += elapsedNanoseconds;
    }

    result.nsPerOperation =
        Median(nsPerOperation);

    result.operationsPerSecond =
        result.nsPerOperation > 0.0
        ? 1.0e9 / result.nsPerOperation
        : 0.0;

    result.errorPercent =
        RelativeError(nsPerOperation);

    result.cyclesPerOperation =
        Median(cyclesPerOperation);

    result.totalMs =
        totalNanoseconds * 1.0e-6;

    return result;
}

inline void CalculateSpeedups()
{
    for (auto& simd : g_simdResults)
    {
        simd.speedup = 0.0;

        const BenchmarkResult* reference = nullptr;

        for (const auto& ref : g_referenceResults)
        {
            if (simd.name == "VecSIMD<float, 4>::Normalize" &&
                ref.name == "Vec4f::Normalize")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<float, 4>::Dot" &&
                ref.name == "Vec4f::Dot")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<double, 2>::Normalize" &&
                ref.name == "Vec2d::Normalize")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<double, 2>::Dot" &&
                ref.name == "Vec2d::Dot")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<double, 4>::Normalize" &&
                ref.name == "Vec4d::Normalize")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<double, 4>::Dot" &&
                ref.name == "Vec4d::Dot")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<float, 8>::Normalize" &&
                ref.name == "Vec8f::Normalize")
            {
                reference = &ref;
                break;
            }

            if (simd.name == "VecSIMD<float, 8>::Dot" &&
                ref.name == "Vec8f::Dot")
            {
                reference = &ref;
                break;
            }
        }

        if (reference != nullptr &&
            simd.nsPerOperation > 0.0)
        {
            simd.speedup =
                reference->nsPerOperation /
                simd.nsPerOperation;
        }
    }
}

inline void PrintSpeedupTable()
{
    std::println();
    std::println("==============================================================");
    std::println("                         SPEEDUP");
    std::println("==============================================================");

    std::println(
        "{:<38} | {:>14}",
        "Benchmark",
        "Speedup"
    );

    std::println("{:-<58}", "");

    for (const auto& simd : g_simdResults)
    {
        if (simd.speedup <= 0.0)
            continue;

        std::println(
            "{:<38} | {:>13.3f}x",
            simd.name,
            simd.speedup
        );
    }
}

void BenchmarkNoSimd();
void BenchmarkSIMD();