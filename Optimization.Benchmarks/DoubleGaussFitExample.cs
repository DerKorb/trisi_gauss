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

            // --- NLopt Trials with different initial step strategies ---
            double[] initialParametersNlopt = (double[])initialParameters.Clone();
            var stepStrategies = new System.Collections.Generic.Dictionary<string, double[]>();

            // Strategy 1: Small percentage of initial guess (similar to previous failing attempt)
            double[] stepsPctInitial = new double[initialParametersNlopt.Length];
            for(int i=0; i < initialParametersNlopt.Length; ++i) stepsPctInitial[i] = Math.Max(0.01, Math.Abs(initialParametersNlopt[i] * 0.05)); // 5% or min 0.01
            stepStrategies.Add("5%_Initial_Min0.01", stepsPctInitial);

            // Strategy 2: Moderate fixed absolute step
            double[] stepsFixedModerate = new double[initialParametersNlopt.Length];
            for(int i=0; i < initialParametersNlopt.Length; ++i) stepsFixedModerate[i] = 0.5;
            stepStrategies.Add("Fixed_0.5", stepsFixedModerate);
            
            // Strategy 3: Larger fixed absolute step
            double[] stepsFixedLarge = new double[initialParametersNlopt.Length];
            for(int i=0; i < initialParametersNlopt.Length; ++i) stepsFixedLarge[i] = 2.0;
            stepStrategies.Add("Fixed_2.0", stepsFixedLarge);

            // Strategy 4: Steps as a fraction of bounds range
            double[] stepsFractionOfRange = new double[initialParametersNlopt.Length];
            for(int i=0; i < initialParametersNlopt.Length; ++i) stepsFractionOfRange[i] = Math.Max(0.1, (upperBounds[i] - lowerBounds[i]) * 0.1); // 10% of range or min 0.1
            stepStrategies.Add("10%_BoundsRange_Min0.1", stepsFractionOfRange);
            
            // Strategy 5: NLopt default (pass null for initialStepArray)
            stepStrategies.Add("NLopt_Default_Steps", null);


            NLoptWrapper.NLoptResultData resultNLopt = null; // To store the best NLopt result for CSV

            foreach(var strategy in stepStrategies)
            {
                if (!csvOutputOnly) Console.WriteLine($"\n--- Running NLopt (NelderMead with Bounds, Steps: {strategy.Key}) ---");
                double[] currentInitialParamsNlopt = (double[])initialParameters.Clone(); // Reset for each strategy

                var watchNLopt = System.Diagnostics.Stopwatch.StartNew();
                var currentNLoptResult = NLoptWrapper.OptimizeNelderMead(
                    ourObjectiveFunc, currentInitialParamsNlopt, lowerBounds, upperBounds,
                    strategy.Value, // Pass the step array for current strategy
                    ftol_rel: 1e-7, xtol_rel: 1e-7, 
                    ftol_abs: 1e-8, xtol_abs_val: 1e-8, maxeval: 20000);
                watchNLopt.Stop();
                
                if (!csvOutputOnly)
                {
                    PrintParameters($"NLopt ({strategy.Key}) Found", currentNLoptResult.OptimalParameters, csvOutputOnly);
                    Console.WriteLine($"NLopt ({strategy.Key}) SSR: {currentNLoptResult.OptimalValue:E2}");
                    Console.WriteLine($"NLopt ({strategy.Key}) Result: {currentNLoptResult.ResultMessage} ({currentNLoptResult.ResultCode})");
                    Console.WriteLine($"NLopt ({strategy.Key}) Time: {watchNLopt.ElapsedMilliseconds} ms");
                }
                if (resultNLopt == null || (currentNLoptResult.OptimalParameters != null && currentNLoptResult.OptimalValue < resultNLopt.OptimalValue))
                {
                    resultNLopt = currentNLoptResult;
                }
            }
            
            if (!csvOutputOnly && resultNLopt != null) {
                 Console.WriteLine("\n--- Best NLopt NelderMead result (used for plot if SBPLX fails) ---");
                 PrintParameters("Best NLopt NM Plot", resultNLopt.OptimalParameters, csvOutputOnly);
                 Console.WriteLine($"Best NLopt NM Plot SSR: {resultNLopt.OptimalValue:E2}");
            }

            // --- Running NLopt (SBPLX with Bounds) ---
            if (!csvOutputOnly) Console.WriteLine("\n--- Running NLopt (SBPLX with Bounds) ---");
            double[] initialParametersSbplx = (double[])initialParameters.Clone();
            double[] sbplxInitialSteps = new double[initialParametersSbplx.Length];
            for(int i=0; i < initialParametersSbplx.Length; ++i) sbplxInitialSteps[i] = 1.0; // Using fixed step of 1.0

            var watchSbplx = System.Diagnostics.Stopwatch.StartNew();
            NLoptWrapper.NLoptResultData resultSbplx = NLoptWrapper.OptimizeSbplx(
                ourObjectiveFunc, initialParametersSbplx, lowerBounds, upperBounds,
                sbplxInitialSteps, 
                ftol_rel: 1e-7, xtol_rel: 1e-7, 
                ftol_abs: 1e-8, xtol_abs_val: 1e-8, maxeval: 20000);
            watchSbplx.Stop();
            
            NLoptWrapper.NLoptResultData finalNloptResultForPlot = resultNLopt; // Default to NelderMead best
            if (!csvOutputOnly) 
            {
                PrintParameters("NLopt SBPLX Found", resultSbplx.OptimalParameters, csvOutputOnly);
                Console.WriteLine($"NLopt SBPLX SSR: {resultSbplx.OptimalValue:E2}");
                Console.WriteLine($"NLopt SBPLX Result: {resultSbplx.ResultMessage} ({resultSbplx.ResultCode})");
                Console.WriteLine($"NLopt SBPLX Time: {watchSbplx.ElapsedMilliseconds} ms");
            }
            // If SBPLX was successful and better than NelderMead, use it for the plot
            if (resultSbplx.OptimalParameters != null && resultSbplx.ResultCode > 0 && 
                (resultNLopt == null || resultSbplx.OptimalValue < resultNLopt.OptimalValue)) 
            {
                finalNloptResultForPlot = resultSbplx;
                if(!csvOutputOnly) Console.WriteLine("(SBPLX result chosen for plot)");
            }
            else if (!csvOutputOnly && resultNLopt != null)
            {
                Console.WriteLine("(Best NelderMead result chosen for plot)");
            }


            if (csvOutputOnly) Console.WriteLine("X,Y_True,Y_Fitted_Our,Y_Fitted_NLopt,Y_NoisyData");
            else Console.WriteLine("\n--- Data for Plotting (CSV format) ---");
            if (!csvOutputOnly) Console.WriteLine("X,Y_True,Y_Fitted_Our,Y_Fitted_NLopt,Y_NoisyData");

            for(int i=0; i < xData.Length; ++i)
            {
                double currentX = xData[i];
                double yTrue = DoubleGaussModel.Calculate(currentX, trueParameters);
                double yFittedOur = DoubleGaussModel.Calculate(currentX, bestParametersOur);
                double yFittedNloptToPlot = (finalNloptResultForPlot != null && finalNloptResultForPlot.OptimalParameters != null) ? DoubleGaussModel.Calculate(currentX, finalNloptResultForPlot.OptimalParameters) : double.NaN;
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

        private static void PrintParameters(string label, ReadOnlySpan<double> parameters, bool csvOutputOnly = false)
        {
            if (csvOutputOnly && (!label.Contains("Best NLopt Plot") && !label.Contains("NLopt SBPLX Found"))) return; 
            if (csvOutputOnly && (label.Contains("Best NLopt Plot") || label.Contains("NLopt SBPLX Found"))) {
                 // Attempt to calculate SSR for comment - CAUTION: xData, yData not available here
                 // For simplicity, just print parameters for CSV comment
                 Console.WriteLine($"# {label}: A1={parameters[0]:F2}, mu1={parameters[1]:F2}, sigma1={parameters[2]:F2}, " +
                                  $"A2={parameters[3]:F2}, mu2={parameters[4]:F2}, sigma2={parameters[5]:F2}");
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