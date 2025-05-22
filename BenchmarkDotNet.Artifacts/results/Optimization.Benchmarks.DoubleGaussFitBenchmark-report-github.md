```

BenchmarkDotNet v0.15.0, Linux Manjaro Linux
AMD Ryzen Threadripper 1950X 3.40GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.105
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                                   | Mean     | Error     | StdDev    | Ratio | RatioSD | Gen0     | Allocated | Alloc Ratio |
|----------------------------------------- |---------:|----------:|----------:|------:|--------:|---------:|----------:|------------:|
| OurNelderMead_DoubleGaussFit_Specialized | 4.962 ms | 0.0955 ms | 0.1208 ms |  1.00 |    0.03 |  31.2500 | 147.05 KB |        1.00 |
| MathNetNelderMead_DoubleGaussFit         | 1.131 ms | 0.0175 ms | 0.0163 ms |  0.23 |    0.01 | 156.2500 | 639.49 KB |        4.35 |
