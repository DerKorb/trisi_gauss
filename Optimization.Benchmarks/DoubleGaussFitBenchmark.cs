using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;
using Optimization.Core.Algorithms.External; // For our P/Invoke NLoptWrapper
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
        private double[] _nloptInitialSteps; 

        private ObjectiveFunctionDouble _objectiveFunction;
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
                _yData[i] = cleanY + (random.NextDouble() * 2 - 1) * maxAmplitude * noisePercentage;
            }

            // Use a good initial guess for benchmarks for fairer speed comparison
            _initialParameters = new double[]{ 9.0, 19.0, 2.5, 14.0, 39.0, 4.5 }; 
            
            _lowerBounds = new double[] { 0.1,  xMin,  0.1,  0.1,  xMin,  0.1 }; 
            _upperBounds = new double[] { 50.0, xMax, 30.0, 50.0, xMax, 30.0 }; 

            _nloptInitialSteps = new double[_initialParameters.Length];
            for(int i=0; i < _initialParameters.Length; ++i)
            {
                _nloptInitialSteps[i] = Math.Max(0.1, (_upperBounds[i] - _lowerBounds[i]) * 0.1); 
            }
            _objectiveFunction = (pars) => DoubleGaussModel.SumSquaredResiduals(pars, _xData, _yData);
        }

        [Benchmark(Baseline = true)]
        public ReadOnlySpan<double> OurNelderMead_DoubleGaussFit_Bounded()
        {
            return NelderMeadDouble.Minimize(_objectiveFunction, _initialParameters, _lowerBounds, 
                _upperBounds, step: 0.5, maxIterations: 10000, tolerance: 1e-7, verbose: false);
        }

        [Benchmark]
        public NLoptWrapper.NLoptResultData NLoptWrapper_NelderMead_Bounded()
        {
            return NLoptWrapper.OptimizeNelderMead(_objectiveFunction, (double[])_initialParameters.Clone(), 
                _lowerBounds, _upperBounds, _nloptInitialSteps, 
                ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-4, maxeval: 10000, verboseLogging: false);
        }

        [Benchmark]
        public NLoptWrapper.NLoptResultData NLoptWrapper_SBPLX_Bounded()
        {
           return NLoptWrapper.OptimizeSbplx(_objectiveFunction, (double[])_initialParameters.Clone(), 
                _lowerBounds, _upperBounds, _nloptInitialSteps, 
                ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-4, maxeval: 10000, verboseLogging: false);
        }

        [Benchmark]
        public NLoptWrapper.NLoptResultData NLoptWrapper_COBYLA_Bounded()
        {
            return NLoptWrapper.OptimizeCobyla(_objectiveFunction, (double[])_initialParameters.Clone(), 
                _lowerBounds, _upperBounds, _nloptInitialSteps, 
                ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-4, maxeval: 10000, verboseLogging: false);
        }
    }
} 