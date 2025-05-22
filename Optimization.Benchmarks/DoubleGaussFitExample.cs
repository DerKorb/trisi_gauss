using System;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;
using Optimization.Core.Algorithms.External; // For NLoptWrapper
// using MathNet.Numerics.LinearAlgebra; // No longer needed for MathNet
// using MathNetNumericsOptimization = MathNet.Numerics.Optimization; // No longer needed for MathNet

namespace Optimization.Benchmarks // Re-using namespace for simplicity, can be a separate project
{
    public static class DoubleGaussFitExample
    {
        public static void RunExample(double noisePercentage = 0.15, bool useBadInitialGuess = true, bool csvOutputOnly = false)
        {
            if (!csvOutputOnly)
            {
                Console.WriteLine($"Running Double Gaussian Fit Example (Noise: {noisePercentage:P0}, Bad Guess: {useBadInitialGuess})...");
            }

            // 1. Define True Parameters
            double[] trueParameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            if (!csvOutputOnly) 
            {
                Console.WriteLine("\n--- True Parameters ---");
                PrintParameters("True", trueParameters, csvOutputOnly);
            }

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
            if (!csvOutputOnly) 
            {
                Console.WriteLine("\n--- Initial Guess ---");
                PrintParameters("Initial", initialParameters, csvOutputOnly);
            }

            // Define bounds for parameters [A1, mu1, sigma1, A2, mu2, sigma2]
            double[] lowerBounds = { 0.1,  xMin,  0.1,  0.1,  xMin,  0.1 }; // Amplitudes and Sigmas > 0.1, Mus within data range
            double[] upperBounds = { 50.0, xMax, 30.0, 50.0, xMax, 30.0 }; // Max Amplitudes, Sigmas up to half data range

            // --- Our NelderMeadDouble Implementation ---
            if (!csvOutputOnly) Console.WriteLine("\n--- Running Our NelderMeadDouble (with Bounds) ---");
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
            if (!csvOutputOnly) 
            {
                PrintParameters("Our Found", bestParametersOur, csvOutputOnly);
                Console.WriteLine($"Our SSR: {DoubleGaussModel.SumSquaredResiduals(bestParametersOur, xData, yData):E2}");
                Console.WriteLine($"Our Time: {watchOur.ElapsedMilliseconds} ms");
            }

            // --- NLopt Trials with specific initial step strategies ---
            double[] initialParametersNlopt = (double[])initialParameters.Clone();
            NLoptWrapper.NLoptResultData nloptNelderMeadDefaultStepsResult = null;
            NLoptWrapper.NLoptResultData nloptNelderMeadTinyStepsResult = null;
            NLoptWrapper.NLoptResultData nloptSbplxDefaultStepsResult = null;
            NLoptWrapper.NLoptResultData nloptSbplxTinyStepsResult = null;

            // Strategy 1: NLopt Default Steps (null for initialStepArray)
            if (!csvOutputOnly) Console.WriteLine("\n--- Running NLopt (NelderMead with Bounds, Default Steps) ---");
            var watchNloptNmDefault = System.Diagnostics.Stopwatch.StartNew();
            nloptNelderMeadDefaultStepsResult = NLoptWrapper.OptimizeNelderMead(
                ourObjectiveFunc, (double[])initialParameters.Clone(), lowerBounds, upperBounds,
                null, ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-8, maxeval: 20000);
            watchNloptNmDefault.Stop();
            if (!csvOutputOnly) PrintNLoptResult("NLopt NM DefaultSteps", nloptNelderMeadDefaultStepsResult, watchNloptNmDefault.ElapsedMilliseconds, csvOutputOnly);

            // Strategy 2: NLopt Tiny Explicit Steps
            if (!csvOutputOnly) Console.WriteLine("\n--- Running NLopt (NelderMead with Bounds, Tiny Steps 1e-5) ---");
            double[] tinySteps = new double[initialParameters.Length];
            Array.Fill(tinySteps, 1e-5);
            var watchNloptNmTiny = System.Diagnostics.Stopwatch.StartNew();
            nloptNelderMeadTinyStepsResult = NLoptWrapper.OptimizeNelderMead(
                ourObjectiveFunc, (double[])initialParameters.Clone(), lowerBounds, upperBounds,
                tinySteps, ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-8, maxeval: 20000);
            watchNloptNmTiny.Stop();
            if (!csvOutputOnly) PrintNLoptResult("NLopt NM TinySteps", nloptNelderMeadTinyStepsResult, watchNloptNmTiny.ElapsedMilliseconds, csvOutputOnly);

            // Strategy 3: NLopt SBPLX Default Steps
             if (!csvOutputOnly) Console.WriteLine("\n--- Running NLopt (SBPLX with Bounds, Default Steps) ---");
            var watchNloptSbplxDefault = System.Diagnostics.Stopwatch.StartNew();
            nloptSbplxDefaultStepsResult = NLoptWrapper.OptimizeSbplx(
                ourObjectiveFunc, (double[])initialParameters.Clone(), lowerBounds, upperBounds,
                null, ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-8, maxeval: 20000);
            watchNloptSbplxDefault.Stop();
            if (!csvOutputOnly) PrintNLoptResult("NLopt SBPLX DefaultSteps", nloptSbplxDefaultStepsResult, watchNloptSbplxDefault.ElapsedMilliseconds, csvOutputOnly);

            // Strategy 4: NLopt SBPLX Tiny Explicit Steps
            if (!csvOutputOnly) Console.WriteLine("\n--- Running NLopt (SBPLX with Bounds, Tiny Steps 1e-5) ---");
            var watchNloptSbplxTiny = System.Diagnostics.Stopwatch.StartNew();
            nloptSbplxTinyStepsResult = NLoptWrapper.OptimizeSbplx(
                ourObjectiveFunc, (double[])initialParameters.Clone(), lowerBounds, upperBounds,
                tinySteps, ftol_rel: 1e-7, xtol_rel: 1e-7, ftol_abs: 1e-8, xtol_abs_val: 1e-8, maxeval: 20000);
            watchNloptSbplxTiny.Stop();
            if (!csvOutputOnly) PrintNLoptResult("NLopt SBPLX TinySteps", nloptSbplxTinyStepsResult, watchNloptSbplxTiny.ElapsedMilliseconds, csvOutputOnly);
            
            // Determine best NLopt result for plotting (simple SSR comparison)
            NLoptWrapper.NLoptResultData bestNloptResultForPlot = nloptNelderMeadDefaultStepsResult; // Start with one
            if (nloptNelderMeadTinyStepsResult.OptimalValue < bestNloptResultForPlot.OptimalValue) bestNloptResultForPlot = nloptNelderMeadTinyStepsResult;
            if (nloptSbplxDefaultStepsResult.OptimalValue < bestNloptResultForPlot.OptimalValue) bestNloptResultForPlot = nloptSbplxDefaultStepsResult;
            if (nloptSbplxTinyStepsResult.OptimalValue < bestNloptResultForPlot.OptimalValue) bestNloptResultForPlot = nloptSbplxTinyStepsResult;
            
            if (!csvOutputOnly && bestNloptResultForPlot != null) {
                 Console.WriteLine("\n--- Best NLopt result overall (used for plot) ---");
                 PrintParameters("Best NLopt Combined", bestNloptResultForPlot.OptimalParameters, csvOutputOnly);
                 Console.WriteLine($"Best NLopt Combined SSR: {bestNloptResultForPlot.OptimalValue:E2}");
            }

            // --- Generate Data for Plotting (CSV or Full) ---
            if (csvOutputOnly) Console.WriteLine("X,Y_True,Y_Fitted_Our,Y_Fitted_NLopt,Y_NoisyData");
            else {
                Console.WriteLine("\n--- Data for Plotting (CSV format) ---");
                Console.WriteLine("X,Y_True,Y_Fitted_Our,Y_Fitted_NLopt,Y_NoisyData");
            }

            for(int i=0; i < xData.Length; ++i)
            {
                double currentX = xData[i];
                double yTrue = DoubleGaussModel.Calculate(currentX, trueParameters);
                double yFittedOur = DoubleGaussModel.Calculate(currentX, bestParametersOur);
                double yFittedNloptToPlot = (bestNloptResultForPlot != null && bestNloptResultForPlot.OptimalParameters != null) ? DoubleGaussModel.Calculate(currentX, bestNloptResultForPlot.OptimalParameters) : double.NaN;
                double yNoisy = yData[i];
                Console.WriteLine($"{currentX:F2},{yTrue:F4},{yFittedOur:F4},{yFittedNloptToPlot:F4},{yNoisy:F4}");
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

        private static void PrintNLoptResult(string strategyLabel, NLoptWrapper.NLoptResultData nloptResult, long timeMs, bool csvOutputOnly)
        {
            if (csvOutputOnly) return;
            PrintParameters($"NLopt ({strategyLabel}) Found", nloptResult.OptimalParameters, csvOutputOnly);
            Console.WriteLine($"NLopt ({strategyLabel}) SSR: {nloptResult.OptimalValue:E2}");
            Console.WriteLine($"NLopt ({strategyLabel}) Result: {nloptResult.ResultMessage} ({nloptResult.ResultCode})");
            Console.WriteLine($"NLopt ({strategyLabel}) Time: {timeMs} ms");
        }

        private static void PrintParameters(string label, ReadOnlySpan<double> parameters, bool csvOutputOnly = false)
        {
            if (csvOutputOnly && !(label.Contains("Best NLopt Combined") || label.Contains("True") || label.Contains("Initial"))) return;
            if (csvOutputOnly && (label.Contains("Best NLopt Combined") || label.Contains("True") || label.Contains("Initial"))) {
                 Console.WriteLine($"# {label}: A1={parameters[0]:F2}, mu1={parameters[1]:F2}, sigma1={parameters[2]:F2}, " +
                                  $"A2={parameters[3]:F2}, mu2={parameters[4]:F2}, sigma2={parameters[5]:F2}");
                 // For SSR in comment, would need xData, yData. Skip for now.
                 return;
            }

            if (parameters != null && parameters.Length == 6)
            {
                Console.WriteLine($"{label}: A1={parameters[0]:F2}, mu1={parameters[1]:F2}, sigma1={parameters[2]:F2}, " +
                                  $"A2={parameters[3]:F2}, mu2={parameters[4]:F2}, sigma2={parameters[5]:F2}");
            }
            else
            {
                Console.WriteLine($"{label}: Invalid or null parameters.");
            }
        }
    }
} 