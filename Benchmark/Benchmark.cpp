#include "Benchmark.h"

#include <filesystem>

void BenchmarkNoSimd()
{
    std::println();
    std::println("Benchmark without SIMD optimizations:");
    PrintBenchmarkHeader();
    BenchmarkVecNoSIMD();
    BenchmarkMatNoSIMD();
    BenchmarkQuaternionNoSIMD();
}

void BenchmarkSIMD()
{
    std::println();
    std::println("Benchmark with SIMD optimizations:");
    PrintBenchmarkHeader();
    BenchmarkVecSIMD();
    BenchmarkMatSIMD();
    BenchmarkQuaternionSIMD();
}

int main()
{
    if (std::filesystem::exists("Benchmark.csv"))
    {
        std::filesystem::remove("Benchmark.csv");
    }

    std::println("==============================================================");
    std::println("                  CONFIGURATION DU BENCHMARK");
    std::println("==============================================================");

    std::size_t batchSize = ChooseBatchSize();
    int benchmarkType = ChooseBenchmarkType();

    constexpr std::uint32_t seed = 0x12345678;

    BeginBenchmark(batchSize, seed);

    if (benchmarkType == 1 || benchmarkType == 3)
    {
        BenchmarkNoSimd();
    }

    if (benchmarkType == 2 || benchmarkType == 3)
    {
        BenchmarkSIMD();
    }

    if (benchmarkType == 3)
    {
        CalculateSpeedups();
        PrintSpeedupTable();
    }

    std::println();
    std::println("Resultats sauvegardes dans Benchmark.csv");

    return 0;
}