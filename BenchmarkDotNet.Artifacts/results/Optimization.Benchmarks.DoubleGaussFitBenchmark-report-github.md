```

BenchmarkDotNet v0.15.0, Linux Manjaro Linux
AMD Ryzen Threadripper 1950X 3.40GHz, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.105
  [Host]     : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2
  DefaultJob : .NET 9.0.4 (9.0.425.16305), X64 RyuJIT AVX2


```
| Method                                   | Mean     | Error     | StdDev    | Ratio | RatioSD | Gen0     | Allocated | Alloc Ratio |
|----------------------------------------- |---------:|----------:|----------:|------:|--------:|---------:|----------:|------------:|
| OurNelderMead_DoubleGaussFit_Specialized | 1.158 ms | 0.0230 ms | 0.0255 ms |  1.00 |    0.03 |        - |     841 B |        1.00 |
| MathNetNelderMead_DoubleGaussFit         | 1.120 ms | 0.0214 ms | 0.0263 ms |  0.97 |    0.03 | 156.2500 |  654841 B |      778.65 |
