```

BenchmarkDotNet v0.15.0, Linux Manjaro Linux
AMD Ryzen Threadripper 1950X 3.40GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.105
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                       | Mean     | Error    | StdDev   | Gen0    | Allocated |
|----------------------------- |---------:|---------:|---------:|--------:|----------:|
| OurNelderMead_Rosenbrock     | 32.18 μs | 0.641 μs | 1.123 μs | 18.1274 |  74.09 KB |
| MathNetNelderMead_Rosenbrock | 64.29 μs | 0.545 μs | 0.455 μs | 42.7246 | 174.93 KB |
