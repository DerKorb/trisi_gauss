using System;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;
using MathNet.Numerics.LinearAlgebra; // For MathNet comparison
using MathNetNumericsOptimization = MathNet.Numerics.Optimization; // For MathNet comparison

namespace Optimization.Benchmarks // Re-using namespace for simplicity, can be a separate project
{
    public static class DoubleGaussFitExample
    {
        public static void RunExample(double noisePercentage = 0.15, bool useBadInitialGuess = true)
        {
            Console.WriteLine($"Running Double Gaussian Fit Example (Noise: {noisePercentage:P0}, Bad Guess: {useBadInitialGuess})...");

            // 1. Define True Parameters
            double[] trueParameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            Console.WriteLine("\n--- True Parameters ---");
            PrintParameters("True", trueParameters);

            // 2. Generate Synthetic Data
            int numDataPoints = 100;
            double[] xData = new double[numDataPoints];
            double[] yData = new double[numDataPoints];
            Random random = new Random(123); 
            double xMin = 0, xMax = 60;
            double maxAmplitude = Math.Max(trueParameters[0], trueParameters[3]);

            for (int i = 0; i < numDataPoints; i++)
            {
                xData[i] = xMin + (xMax - xMin) * i / (numDataPoints - 1);
                double cleanY = DoubleGaussModel.Calculate(xData[i], trueParameters);
                double noise = (random.NextDouble() * 2 - 1) * maxAmplitude * noisePercentage;
                yData[i] = cleanY + noise;
            }

            // 3. Define Initial Guess
            double[] initialParameters;
            if (useBadInitialGuess)
            {
                initialParameters = new double[] { 1.0, 10.0, 1.0, 1.0, 50.0, 10.0 }; // Worse initial guess
            }
            else
            {
                initialParameters = new double[] { 8.0, 15.0, 2.0, 12.0, 35.0, 4.0 }; // Original good guess
            }
            Console.WriteLine("\n--- Initial Guess ---");
            PrintParameters("Initial", initialParameters);

            // Define bounds for parameters [A1, mu1, sigma1, A2, mu2, sigma2]
            double[] lowerBounds = { 0.1,  xMin,  0.1,  0.1,  xMin,  0.1 }; // Amplitudes and Sigmas > 0.1, Mus within data range
            double[] upperBounds = { 50.0, xMax, 30.0, 50.0, xMax, 30.0 }; // Max Amplitudes, Sigmas up to half data range

            // --- Our NelderMeadDouble Implementation ---
            Console.WriteLine("\n--- Running Our NelderMeadDouble (with Bounds) ---");
            ObjectiveFunctionDouble ourObjectiveFunc = (pars) => 
                DoubleGaussModel.SumSquaredResiduals(pars, xData, yData);
            
            var watchOur = System.Diagnostics.Stopwatch.StartNew();
            ReadOnlySpan<double> bestParametersOur = NelderMeadDouble.Minimize(
                ourObjectiveFunc, 
                initialParameters, 
                lowerBounds, // Pass lower bounds
                upperBounds, // Pass upper bounds
                step: 0.5, 
                maxIterations: 20000, 
                tolerance: 1e-8
            );
            watchOur.Stop();
            PrintParameters("Our Found", bestParametersOur);
            Console.WriteLine($"Our SSR: {DoubleGaussModel.SumSquaredResiduals(bestParametersOur, xData, yData):E2}");
            Console.WriteLine($"Our Time: {watchOur.ElapsedMilliseconds} ms");

            // --- MathNet.Numerics NelderMeadSimplex Implementation ---
            Console.WriteLine("\n--- Running MathNet.Numerics NelderMeadSimplex ---");
            Func<Vector<double>, double> mathNetObjectiveFunc = (parsVec) =>
                DoubleGaussModel.SumSquaredResiduals(parsVec.AsArray(), xData, yData);
            
            var initialGuessMathNet = Vector<double>.Build.Dense(initialParameters);
            var solver = new MathNetNumericsOptimization.NelderMeadSimplex(convergenceTolerance: 1e-8, maximumIterations: 20000);
            
            var watchMathNet = System.Diagnostics.Stopwatch.StartNew();
            // MathNet's basic NelderMeadSimplex does not directly take bounds array like ours.
            // To compare fairly with bounds, one would typically wrap the objective function 
            // with a penalty for MathNet, or use a MathNet solver that supports bounds.
            // For this run, MathNet will be unconstrained as before, to highlight the effect of bounds on our impl.
            var mathNetObjective = MathNetNumericsOptimization.ObjectiveFunction.Value(mathNetObjectiveFunc);
            var resultMathNet = solver.FindMinimum(mathNetObjective, initialGuessMathNet);
            watchMathNet.Stop();
            
            PrintParameters("MathNet Found", resultMathNet.MinimizingPoint.AsArray());
            Console.WriteLine($"MathNet SSR: {resultMathNet.FunctionInfoAtMinimum.Value:E2}");
            Console.WriteLine($"MathNet Iterations: {resultMathNet.Iterations}");
            Console.WriteLine($"MathNet ReasonForExit: {resultMathNet.ReasonForExit}");
            Console.WriteLine($"MathNet Time: {watchMathNet.ElapsedMilliseconds} ms");

            // --- Generate Data for Plotting ---
            Console.WriteLine("\n--- Data for Plotting (CSV format) ---");
            Console.WriteLine("X,Y_True,Y_Fitted_Our,Y_Fitted_MathNet,Y_NoisyData");
            
            // For smoother function plots, we can use a denser set of X values than just xData
            // Or, for simplicity, just use the xData points and connect them in the plot.
            // Let's use xData for direct comparison with noisy points.
            for(int i=0; i < xData.Length; ++i)
            {
                double currentX = xData[i];
                double yTrue = DoubleGaussModel.Calculate(currentX, trueParameters);
                double yFittedOur = DoubleGaussModel.Calculate(currentX, bestParametersOur);
                double yFittedMathNet = DoubleGaussModel.Calculate(currentX, resultMathNet.MinimizingPoint.AsArray());
                double yNoisy = yData[i];
                Console.WriteLine($"{currentX:F2},{yTrue:F4},{yFittedOur:F4},{yFittedMathNet:F4},{yNoisy:F4}");
            }

            // Comment out the old basic console visualization if we're outputting CSV data
            /*
            Console.WriteLine("\nBasic Console Visualization (Y vs X - True vs OurFitted vs NoisyData):");
            Console.WriteLine("X\tY_True\tY_OurFit\tY_Data(Noisy)");
            for(int i=0; i<numDataPoints; i+= numDataPoints/10) // Print a subset of points
            {
                double yTrueCalc = DoubleGaussModel.Calculate(xData[i], trueParameters);
                double yFittedOurCalc = DoubleGaussModel.Calculate(xData[i], bestParametersOur);
                Console.WriteLine($"{xData[i]:F1}\t{yTrueCalc:F2}\t{yFittedOurCalc:F2}\t{yData[i]:F2}");
            }
            */
        }

        private static void PrintParameters(string label, ReadOnlySpan<double> parameters)
        {
            if (parameters.Length == 6)
            {
                Console.WriteLine($"{label}: A1={parameters[0]:F2}, mu1={parameters[1]:F2}, sigma1={parameters[2]:F2}, " +
                                  $"A2={parameters[3]:F2}, mu2={parameters[4]:F2}, sigma2={parameters[5]:F2}");
            }
            else
            {
                Console.WriteLine($"{label}: Invalid parameter count {parameters.Length}");
            }
        }
    }
} 