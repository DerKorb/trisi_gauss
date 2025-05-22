using System;
using BenchmarkDotNet.Attributes;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;
using Optimization.Core.Algorithms.External; // For NLoptWrapper
// using MathNet.Numerics.LinearAlgebra; // No longer needed for MathNet
// using MathNetNumericsOptimization = MathNet.Numerics.Optimization; // No longer needed for MathNet

namespace Optimization.Benchmarks
{
    [MemoryDiagnoser]
    public class DoubleGaussFitBenchmark
    {
        private const int NumDataPoints = 100;
        private double[] _xData;
        private double[] _yData;
        private double[] _initialParameters;
        private double[] _lowerBounds;
        private double[] _upperBounds;

        private ObjectiveFunctionDouble _ourObjectiveFunctionDouble;
        // private Func<Vector<double>, double> _mathNetObjectiveFunction; // Removed

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
            double noisePercentage = 0.15; // 15% noise for benchmark consistency

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

            // Define bounds consistent with the example, but benchmark will run our method unconstrained for now by passing empty bounds.
            // For NLopt, we WILL pass bounds.
            _lowerBounds = new double[] { 0.1,  xMin,  0.1,  0.1,  xMin,  0.1 }; 
            _upperBounds = new double[] { 50.0, xMax, 30.0, 50.0, xMax, 30.0 }; 

            // Objective function for our optimizer (specialized version)
            _ourObjectiveFunctionDouble = (pars) => 
                DoubleGaussModel.SumSquaredResiduals(pars, _xData, _yData);
        }

        [Benchmark(Baseline = true)]
        public ReadOnlySpan<double> OurNelderMead_DoubleGaussFit_Specialized()
        {
            return NelderMeadDouble.Minimize(
                _ourObjectiveFunctionDouble,
                _initialParameters,
                ReadOnlySpan<double>.Empty, // Run our specialized version unconstrained for this perf benchmark
                ReadOnlySpan<double>.Empty, 
                step: 0.5,
                maxIterations: 10000, 
                tolerance: 1e-7
            );
        }

        [Benchmark]
        public NLoptWrapper.NLoptResultData NLopt_NelderMead_DoubleGaussFit()
        {
            // NLopt requires double[] for parameters, not ReadOnlySpan for the initial call
            double[] initialParamsCopy = (double[])_initialParameters.Clone();
            
            return NLoptWrapper.OptimizeNelderMead(
                _ourObjectiveFunctionDouble, // Our C# objective function
                initialParamsCopy,           // Initial guess
                _lowerBounds,                // Lower bounds for NLopt
                _upperBounds,                // Upper bounds for NLopt
                null,                        // Pass null for initialStepArray, let NLopt use default
                ftol_rel: 1e-7,
                xtol_rel: 1e-7, 
                maxeval: 10000
            );
        }
    }
} 