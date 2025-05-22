# Optimization.Core Library

## Overview

`Optimization.Core` is a .NET library designed for numerical optimization tasks. It currently features a robust and efficient implementation of the Nelder-Mead simplex algorithm, specifically optimized for `double` precision floating-point numbers (`NelderMeadDouble`). The library also includes a model for Double Gaussian functions and demonstrates fitting this model to data.

## Features

*   **`NelderMeadDouble` Optimizer:** A specialized implementation of the Nelder-Mead algorithm.
    *   Optimized for `double` precision for high performance and low memory allocation.
    *   Supports bound constraints on parameters.
    *   Proven robust in challenging fitting scenarios with noisy data and poor initial guesses.
*   **`DoubleGaussModel`:** Provides methods to calculate double Gaussian functions and their sum of squared residuals, suitable for use as an objective function.
*   **Benchmarked Performance:** Internal benchmarks show performance competitive with established libraries for certain scenarios, and superior robustness in others.
*   **Unit Tested:** Core components are validated with a suite of unit tests.

## Getting Started

### Prerequisites

*   .NET SDK (compatible with .NET Standard 2.1 and .NET 9.0 for the library and examples/tests respectively).

### Usage Example

Below is an example of how to use `NelderMeadDouble` to fit a double Gaussian model to synthetic data.

```csharp
using System;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;

public class ExampleUsage
{
    public static void Main(string[] args)
    {
        // Example: Simple Rosenbrock function minimization
        ObjectiveFunctionDouble rosenbrockFunc = (p) =>
        {
            const double a = 1.0;
            const double b = 100.0;
            double x = p[0];
            double y = p[1];
            return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
        };

        double[] initialParameters = { -1.2, 1.0 };
        double[] lowerBounds = { -5.0, -5.0 };
        double[] upperBounds = { 5.0, 5.0 };
        double step = 0.5;
        int maxIterations = 10000;
        double tolerance = 1e-7;

        Console.WriteLine($"Initial Parameters: {string.Join(", ", initialParameters)}");

        ReadOnlySpan<double> bestParameters = NelderMeadDouble.Minimize(
            rosenbrockFunc,
            initialParameters,
            lowerBounds,
            upperBounds,
            step,
            maxIterations,
            tolerance,
            verbose: false);

        Console.WriteLine("Optimization Finished.");
        Console.WriteLine($"Found Parameters: {string.Join(", ", bestParameters.ToArray())}");
        Console.WriteLine($"Final Objective Value: {rosenbrockFunc(bestParameters):E3}");
    }
}
```

## Building the Code

To build the solution, navigate to the root directory (`tristan`) and run:

```bash
dotnet build
```

## Running Tests

Unit tests are located in the `Optimization.Core.Tests` project. To run them:

```bash
dotnet test Optimization.Core.Tests/Optimization.Core.Tests.csproj
```

## Running Benchmarks and Examples

The `Optimization.Benchmarks` project contains performance benchmarks and the main fitting example (`DoubleGaussFitExample.cs`). 

To run the current main example (fitting the Double Gaussian model):
```bash
dotnet run --project Optimization.Benchmarks -c Release
```
(The `Program.Main` in `Optimization.Benchmarks/NelderMeadBenchmark.cs` controls which example or benchmark is run. By default, it runs `DoubleGaussFitExample.RunExample(csvOutputOnly: false);`)

To run the performance benchmarks, modify `Program.Main` in `Optimization.Benchmarks/NelderMeadBenchmark.cs` to execute `BenchmarkRunner.Run<DoubleGaussFitBenchmark>()` and then run the command above.

## License

(To be determined - e.g., MIT License) 