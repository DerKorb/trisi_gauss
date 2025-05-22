using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Optimization.Core.Algorithms;
using MathNet.Numerics.LinearAlgebra;
using MathNetNumericsOptimization = MathNet.Numerics.Optimization;

namespace Optimization.Benchmarks
{
    [MemoryDiagnoser] // To diagnose memory allocations
    public class NelderMeadBenchmark
    {
        private const int N = 2; // Number of dimensions for Rosenbrock
        private double[] _initialParameters = new double[N];

        // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        // Standard values: a=1, b=100
        // Minimum is at (a, a^2) = (1,1) where f(x,y) = 0
        public static double Rosenbrock(Span<double> p)
        {
            if (p.Length != N) throw new ArgumentException($"Expected {N} parameters for Rosenbrock function.");
            const double a = 1.0;
            const double b = 100.0;
            double x = p[0];
            double y = p[1];
            return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
        }

        // Rosenbrock for MathNet.Numerics
        public static double MathNet_Rosenbrock(Vector<double> p)
        {
            if (p.Count != N) throw new ArgumentException($"Expected {N} parameters for Rosenbrock function.");
            const double a = 1.0;
            const double b = 100.0;
            double x = p[0];
            double y = p[1];
            return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
        }

        [GlobalSetup]
        public void Setup()
        {
            // Standard starting point for Rosenbrock 2D
            _initialParameters[0] = -1.2;
            _initialParameters[1] = 1.0;
        }

        [Benchmark]
        public Span<double> OurNelderMead_Rosenbrock()
        {
            return NelderMead<double>.Minimize(
                Rosenbrock, 
                _initialParameters, 
                step: 0.5, 
                maxIterations: 10000, // Rosenbrock can be tricky
                tolerance: 1e-7
            );
        }

        [Benchmark]
        public MathNetNumericsOptimization.MinimizationResult MathNetNelderMead_Rosenbrock()
        {
            var objective = MathNetNumericsOptimization.ObjectiveFunction.Value(MathNet_Rosenbrock);
            var initialGuess = Vector<double>.Build.Dense(_initialParameters);
            // MathNet's NelderMeadSimplex doesn't have an explicit step parameter like ours for simplex initialization.
            // It uses a default perturbation or requires initial simplex definition.
            // We'll use its default behavior.
            var solver = new MathNetNumericsOptimization.NelderMeadSimplex(convergenceTolerance: 1e-7, maximumIterations: 10000);
            var result = solver.FindMinimum(objective, initialGuess);
            return result;
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            var summary = BenchmarkRunner.Run<NelderMeadBenchmark>();
            Console.WriteLine(summary);
        }
    }
} 