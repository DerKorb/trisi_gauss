```

BenchmarkDotNet v0.15.0, Linux Manjaro Linux
AMD Ryzen Threadripper 1950X 3.40GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.105
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                           | Mean     | Error     | StdDev    | Ratio | RatioSD | Gen0     | Allocated | Alloc Ratio |
|--------------------------------- |---------:|----------:|----------:|------:|--------:|---------:|----------:|------------:|
| OurNelderMead_DoubleGaussFit     | 3.092 ms | 0.0618 ms | 0.0825 ms |  1.00 |    0.04 | 386.7188 | 1588.8 KB |        1.00 |
| MathNetNelderMead_DoubleGaussFit | 1.124 ms | 0.0145 ms | 0.0136 ms |  0.36 |    0.01 | 156.2500 | 639.49 KB |        0.40 |
