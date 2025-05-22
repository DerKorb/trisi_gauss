using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;
using MathNet.Numerics.LinearAlgebra;
using MathNetNumericsOptimization = MathNet.Numerics.Optimization;

namespace Optimization.Benchmarks
{
    [MemoryDiagnoser]
    public class DoubleGaussFitBenchmark
    {
        private const int NumDataPoints = 100;
        private double[] _xData;
        private double[] _yData;
        private double[] _initialParameters;

        private ObjectiveFunctionDouble _ourObjectiveFunctionDouble;
        private Func<Vector<double>, double> _mathNetObjectiveFunction;

        // True parameters for data generation
        private readonly double[] _trueParameters = { 
            10.0,  // A1
            20.0,  // mu1
            3.0,   // sigma1
            15.0,  // A2
            40.0,  // mu2
            5.0    // sigma2
        };

        [GlobalSetup]
        public void Setup()
        {
            _xData = new double[NumDataPoints];
            _yData = new double[NumDataPoints];
            Random random = new Random(123); // For reproducibility
            double noisePercentage = 0.10; // 10% noise

            double xMin = 0;
            double xMax = 60;
            double maxAmplitude = Math.Max(_trueParameters[0], _trueParameters[3]);

            for (int i = 0; i < NumDataPoints; i++)
            {
                _xData[i] = xMin + (xMax - xMin) * i / (NumDataPoints - 1);
                double cleanY = DoubleGaussModel.Calculate(_xData[i], _trueParameters);
                double noise = (random.NextDouble() * 2 - 1) * maxAmplitude * noisePercentage;
                _yData[i] = cleanY + noise;
            }

            _initialParameters = new double[]{
                8.0,  // A1 guess
                15.0, // mu1 guess
                2.0,  // sigma1 guess
                12.0, // A2 guess
                35.0, // mu2 guess
                4.0   // sigma2 guess
            };

            // Objective function for our optimizer (specialized version)
            _ourObjectiveFunctionDouble = (pars) => 
                DoubleGaussModel.SumSquaredResiduals(pars, _xData, _yData);

            // Objective function for MathNet.Numerics optimizer
            _mathNetObjectiveFunction = (pars) =>
                DoubleGaussModel.SumSquaredResiduals(pars.AsArray(), _xData, _yData);
        }

        [Benchmark(Baseline = true)]
        public ReadOnlySpan<double> OurNelderMead_DoubleGaussFit_Specialized()
        {
            return NelderMeadDouble.Minimize(
                _ourObjectiveFunctionDouble,
                _initialParameters,
                step: 0.5,
                maxIterations: 10000, 
                tolerance: 1e-7
            );
        }

        [Benchmark]
        public MathNetNumericsOptimization.MinimizationResult MathNetNelderMead_DoubleGaussFit()
        {
            var objective = MathNetNumericsOptimization.ObjectiveFunction.Value(_mathNetObjectiveFunction);
            var initialGuess = Vector<double>.Build.Dense(_initialParameters);
            var solver = new MathNetNumericsOptimization.NelderMeadSimplex(convergenceTolerance: 1e-7, maximumIterations: 10000);
            var result = solver.FindMinimum(objective, initialGuess);
            return result;
        }
    }
} 