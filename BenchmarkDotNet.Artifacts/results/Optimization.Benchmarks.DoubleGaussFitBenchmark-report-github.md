```

BenchmarkDotNet v0.15.0, Linux Manjaro Linux
AMD Ryzen Threadripper 1950X 3.40GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.105
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                                   | Mean      | Error     | StdDev   | Ratio | Gen0   | Allocated | Alloc Ratio |
|----------------------------------------- |----------:|----------:|---------:|------:|-------:|----------:|------------:|
| OurNelderMead_DoubleGaussFit_Specialized | 926.90 μs | 10.485 μs | 9.295 μs |  1.00 |      - |     841 B |        1.00 |
| NLopt_NelderMead_DoubleGaussFit          |  16.06 μs |  0.315 μs | 0.509 μs |  0.02 | 0.0916 |     424 B |        0.50 |
