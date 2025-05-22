using System;
using Xunit;
using Optimization.Core.Algorithms;
using Optimization.Core.Models; // For DoubleGaussModel in fitting tests

namespace Optimization.Core.Tests
{
    public class NelderMeadDoubleTests
    {
        // Standard Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        // Minimum at (a, a^2), usually a=1, b=100. Min value = 0.
        private static double Rosenbrock(ReadOnlySpan<double> p)
        {
            const double a = 1.0;
            const double b = 100.0;
            double x = p[0];
            double y = p[1];
            return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
        }

        [Fact]
        public void Minimize_Rosenbrock_Unconstrained_Converges()
        {
            double[] initialParameters = { -1.2, 1.0 };
            var result = NelderMeadDouble.Minimize(
                Rosenbrock, 
                initialParameters, 
                ReadOnlySpan<double>.Empty, // No lower bounds
                ReadOnlySpan<double>.Empty, // No upper bounds
                step: 0.5, 
                maxIterations: 10000, 
                tolerance: 1e-7);

            Assert.Equal(1.0, result[0], 3); // x converges to a (1.0)
            Assert.Equal(1.0, result[1], 3); // y converges to a^2 (1.0)
            Assert.True(Rosenbrock(result) < 1e-6); // Function value at minimum is close to 0
        }

        [Fact]
        public void Minimize_Rosenbrock_WithBoundsIncludingMinimum_Converges()
        {
            double[] initialParameters = { -1.2, 1.0 };
            double[] lowerBounds = { -2.0, -2.0 };
            double[] upperBounds = { 2.0, 2.0 };

            var result = NelderMeadDouble.Minimize(
                Rosenbrock, 
                initialParameters, 
                lowerBounds,
                upperBounds,
                step: 0.5, 
                maxIterations: 10000, 
                tolerance: 1e-7);

            Assert.Equal(1.0, result[0], 3);
            Assert.Equal(1.0, result[1], 3);
            Assert.True(Rosenbrock(result) < 1e-6);
        }

        [Fact]
        public void Minimize_Rosenbrock_WithBoundsExcludingMinimum_StopsAtBoundary()
        {
            double[] initialParameters = { 0.0, 0.0 };
            // Bounds that exclude the true minimum of (1,1)
            double[] lowerBounds = { -0.5, -0.5 };
            double[] upperBounds = { 0.5, 0.5 }; 

            var result = NelderMeadDouble.Minimize(
                Rosenbrock, 
                initialParameters, 
                lowerBounds,
                upperBounds,
                step: 0.1, 
                maxIterations: 5000, 
                tolerance: 1e-6);
            
            // Expect it to converge to a point on or very near the boundary {0.5, 0.25} is close
            // Rosenbrock at (0.5, 0.25) = (1-0.5)^2 + 100*(0.25 - 0.5^2)^2 = 0.25 + 100*(0.25-0.25)^2 = 0.25
            // Due to nature of Nelder-Mead and bounds, it might be slightly off or at a corner.
            // We check if it respects the bounds.
            Assert.True(result[0] >= lowerBounds[0] - 1e-5 && result[0] <= upperBounds[0] + 1e-5);
            Assert.True(result[1] >= lowerBounds[1] - 1e-5 && result[1] <= upperBounds[1] + 1e-5);
            // And that it did better than the initial point
            Assert.True(Rosenbrock(result) < Rosenbrock(initialParameters));
        }

        [Fact]
        public void Minimize_DoubleGaussianFit_GoodInitial_ConvergesAccurately()
        {
            double[] trueParameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            double[] initialParameters = { 8.0, 18.0, 2.5, 12.0, 38.0, 4.5 }; // Good initial guess
            double[] lowerBounds = { 0.1, 0, 0.1, 0.1, 0, 0.1 };
            double[] upperBounds = { 50, 60, 30, 50, 60, 30 };
            int numDataPoints = 100;
            double[] xData = new double[numDataPoints];
            double[] yData = new double[numDataPoints];
            Random random = new Random(42);
            double noiseLevel = 0.05; // 5% noise
            double maxAmp = Math.Max(trueParameters[0], trueParameters[3]);

            for (int i = 0; i < numDataPoints; i++)
            {
                xData[i] = 0 + (60.0 - 0) * i / (numDataPoints - 1);
                double cleanY = DoubleGaussModel.Calculate(xData[i], trueParameters);
                yData[i] = cleanY + (random.NextDouble() * 2 - 1) * maxAmp * noiseLevel;
            }

            ObjectiveFunctionDouble objective = pars => DoubleGaussModel.SumSquaredResiduals(pars, xData, yData);

            var result = NelderMeadDouble.Minimize(objective, initialParameters, lowerBounds, upperBounds, 0.5, 20000, 1e-7);

            for (int i = 0; i < trueParameters.Length; i++)
            {
                Assert.True(Math.Abs(trueParameters[i] - result[i]) < 0.25, $"Parameter {i} ({result[i]:F3}) not close enough to true value {trueParameters[i]:F3}. Difference: {Math.Abs(trueParameters[i] - result[i]):F3}");
            }
            Assert.True(DoubleGaussModel.SumSquaredResiduals(result, xData, yData) < 100, "SSR too high for good initial guess.");
        }

        [Fact]
        public void Minimize_Throws_OnNullObjectiveFunction()
        {
            Assert.Throws<ArgumentNullException>(() => NelderMeadDouble.Minimize(null, new double[] {1,1}, null, null, 0.1, 100, 1e-6));
        }

        [Fact]
        public void Minimize_Throws_OnEmptyInitialParameters()
        {
             ObjectiveFunctionDouble rosenbrockFunc = Rosenbrock;
            Assert.Throws<ArgumentException>(() => NelderMeadDouble.Minimize(rosenbrockFunc, Array.Empty<double>(), null, null, 0.1, 100, 1e-6));
        }

        [Fact]
        public void Minimize_Throws_OnMismatchedBounds()
        {
            ObjectiveFunctionDouble rosenbrockFunc = Rosenbrock;
            double[] initial = {1,1};
            double[] wrongLower = {0};
            double[] correctUpper = {2,2};
            Assert.Throws<ArgumentException>(() => NelderMeadDouble.Minimize(rosenbrockFunc, initial, wrongLower, correctUpper, 0.1, 100, 1e-6));
            
            double[] correctLower = {0,0};
            double[] wrongUpper = {2};
            Assert.Throws<ArgumentException>(() => NelderMeadDouble.Minimize(rosenbrockFunc, initial, correctLower, wrongUpper, 0.1, 100, 1e-6));
        }

        [Fact]
        public void Minimize_Throws_OnInitialParametersViolatingBounds()
        {
            ObjectiveFunctionDouble rosenbrockFunc = Rosenbrock;
            double[] initialViolatingLower = { -1, 0 };
            double[] lower = {0,0};
            double[] upper = {2,2};
            Assert.Throws<ArgumentOutOfRangeException>(() => NelderMeadDouble.Minimize(rosenbrockFunc, initialViolatingLower, lower, upper, 0.1, 100, 1e-6));

            double[] initialViolatingUpper = { 3, 0 };
            Assert.Throws<ArgumentOutOfRangeException>(() => NelderMeadDouble.Minimize(rosenbrockFunc, initialViolatingUpper, lower, upper, 0.1, 100, 1e-6));
        }

        [Fact]
        public void Minimize_Rosenbrock_PerformanceSmokeTest()
        {
            double[] initialParameters = { -1.2, 1.0 };
            long maxAllowedMilliseconds = 200; // Generous time limit for a smoke test

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var result = NelderMeadDouble.Minimize(
                Rosenbrock, 
                initialParameters, 
                ReadOnlySpan<double>.Empty, 
                ReadOnlySpan<double>.Empty, 
                step: 0.5, 
                maxIterations: 10000, // Ensure it does enough work to be meaningful
                tolerance: 1e-7,
                verbose: false); // Keep verbose off for smoke test
            stopwatch.Stop();

            Assert.True(Rosenbrock(result) < 1e-6, "Rosenbrock did not converge to minimum.");
            Assert.True(stopwatch.ElapsedMilliseconds < maxAllowedMilliseconds, 
                $"Rosenbrock optimization took too long: {stopwatch.ElapsedMilliseconds} ms (expected < {maxAllowedMilliseconds} ms)");
        }
    }
} 