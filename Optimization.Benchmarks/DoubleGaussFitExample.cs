using System;
using System.Linq;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;
using Optimization.Core.Algorithms.External; // Using our P/Invoke NLoptWrapper

namespace Optimization.Benchmarks
{
    internal struct TempNLoptResult 
    { 
        public NLoptWrapper.nlopt_algorithm Algorithm {get; set;} 
        public string Context { get; set; } // e.g., "Bounded" or "Unconstrained"
        public double[] Parameters {get; set;} 
        public double OptimalValue {get; set;}
        public NLoptWrapper.nlopt_result ResultCode {get; set;} 
        public string ResultMessage { get; set; }
        public long TimeMs { get; set; }
        public int Evaluations { get; set; }
    }

    public static class DoubleGaussFitExample
    {
        private static int _nloptEvalCount = 0; 

        public static void RunExample(double noisePercentage = 0.15, bool useBadInitialGuess = true, bool csvOutputOnly = false)
        {
            if (!csvOutputOnly) Console.WriteLine($"Running Double Gaussian Fit Example (Noise: {noisePercentage:P0}, Bad Guess: {useBadInitialGuess})...");

            double[] trueParameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            if (!csvOutputOnly) PrintParameters("True Parameters", trueParameters);
            
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
                yData[i] = cleanY + (random.NextDouble() * 2 - 1) * maxAmplitude * noisePercentage;
            }
            double[] initialParameters = useBadInitialGuess ? 
                new double[] { 1.0, 10.0, 1.0, 1.0, 50.0, 10.0 } :
                new double[] { 9.0, 19.0, 2.5, 14.0, 39.0, 4.5 };
            if (!csvOutputOnly) PrintParameters("Initial Guess", initialParameters);
            
            double[] lowerBounds = { 0.1, xMin, 0.1, 0.1, xMin, 0.1 };
            double[] upperBounds = { 50.0, xMax, 30.0, 50.0, xMax, 30.0 };
            
            ObjectiveFunctionDouble objectiveFunction = (parsSpan) => 
                DoubleGaussModel.SumSquaredResiduals(parsSpan, xData, yData);

            bool runOurVerbose = !csvOutputOnly && useBadInitialGuess; 
            if (!csvOutputOnly) Console.WriteLine("\n--- Running Our NelderMeadDouble (with Bounds) ---");
            var watchOur = System.Diagnostics.Stopwatch.StartNew();
            ReadOnlySpan<double> bestParametersOur = NelderMeadDouble.Minimize(
                objectiveFunction, initialParameters, lowerBounds, upperBounds, 
                step: 0.5, maxIterations: 50000, tolerance: 1e-7, verbose: runOurVerbose);
            watchOur.Stop();
            double ssrOur = bestParametersOur.IsEmpty ? double.MaxValue : DoubleGaussModel.SumSquaredResiduals(bestParametersOur, xData, yData);
            if (!csvOutputOnly) 
            {
                PrintParameters("Our Found", bestParametersOur);
                Console.WriteLine($"Our SSR: {ssrOur:E2}");
                Console.WriteLine($"Our Time: {watchOur.ElapsedMilliseconds} ms");
            }
            
            var nloptAlgosToTest = new[] { 
                NLoptWrapper.nlopt_algorithm.NLOPT_LN_COBYLA,
                NLoptWrapper.nlopt_algorithm.NLOPT_LN_NELDERMEAD, 
                NLoptWrapper.nlopt_algorithm.NLOPT_LN_SBPLX 
            };
            TempNLoptResult? overallBestNloptResult = null; 

            double[] nloptInitialSteps = new double[initialParameters.Length];
            for(int i=0; i < initialParameters.Length; ++i) nloptInitialSteps[i] = Math.Max(0.1, (upperBounds[i] - lowerBounds[i]) * 0.1);

            bool nloptVerboseLogging = !csvOutputOnly && useBadInitialGuess;

            foreach (bool useBoundsForNloptRun in new[] { true, false })
            {
                string context = useBoundsForNloptRun ? "With Bounds" : "Unconstrained";
                if (!csvOutputOnly) Console.WriteLine($"\n=== NLOPT RUNS ({context}) ===");

                foreach (var algo in nloptAlgosToTest)
                {
                    if (!csvOutputOnly) Console.WriteLine($"\n--- Running NLoptWrapper ({algo}, {context}, InitialSteps: ScaledToBoundsRange) ---");
                    
                    double[] currentInitialParamsNlopt = (double[])initialParameters.Clone();
                    var watchNlopt = System.Diagnostics.Stopwatch.StartNew();
                    NLoptWrapper.NLoptResultData currentNLoptRunResult = null;
                    int maxEvalsForNlopt = 50000;

                    Func<NLoptWrapper.NLoptResultData> optimizationCall = algo switch
                    {
                        NLoptWrapper.nlopt_algorithm.NLOPT_LN_NELDERMEAD => () => NLoptWrapper.OptimizeNelderMead(objectiveFunction, currentInitialParamsNlopt, useBoundsForNloptRun ? lowerBounds : null, useBoundsForNloptRun ? upperBounds : null, nloptInitialSteps, verboseLogging: nloptVerboseLogging, maxeval:maxEvalsForNlopt, ftol_rel:1e-7, xtol_rel:1e-7, ftol_abs:1e-8, xtol_abs_val:1e-4),
                        NLoptWrapper.nlopt_algorithm.NLOPT_LN_SBPLX    => () => NLoptWrapper.OptimizeSbplx(objectiveFunction, currentInitialParamsNlopt, useBoundsForNloptRun ? lowerBounds : null, useBoundsForNloptRun ? upperBounds : null, nloptInitialSteps, verboseLogging: nloptVerboseLogging, maxeval:maxEvalsForNlopt, ftol_rel:1e-7, xtol_rel:1e-7, ftol_abs:1e-8, xtol_abs_val:1e-4),
                        NLoptWrapper.nlopt_algorithm.NLOPT_LN_COBYLA   => () => NLoptWrapper.OptimizeCobyla(objectiveFunction, currentInitialParamsNlopt, useBoundsForNloptRun ? lowerBounds : null, useBoundsForNloptRun ? upperBounds : null, nloptInitialSteps, verboseLogging: nloptVerboseLogging, maxeval:maxEvalsForNlopt, ftol_rel:1e-7, xtol_rel:1e-7, ftol_abs:1e-8, xtol_abs_val:1e-4),
                        _ => throw new ArgumentOutOfRangeException(nameof(algo), $"Unsupported NLopt algorithm: {algo}")
                    };
                    currentNLoptRunResult = optimizationCall();
                    watchNlopt.Stop();

                    if (!csvOutputOnly && currentNLoptRunResult != null)
                    {
                        Console.WriteLine($"NLoptWrapper ({algo} - {context}) Total Evals: {currentNLoptRunResult.Evaluations}");
                        PrintParameters($"NLoptWrapper ({algo} - {context}) Found", currentNLoptRunResult.OptimalParameters);
                        Console.WriteLine($"NLoptWrapper ({algo} - {context}) SSR: {currentNLoptRunResult.OptimalValue:E2}");
                        Console.WriteLine($"NLoptWrapper ({algo} - {context}) Result: {currentNLoptRunResult.ResultMessage} ({currentNLoptRunResult.ResultCode})");
                        Console.WriteLine($"NLoptWrapper ({algo} - {context}) Time: {watchNlopt.ElapsedMilliseconds} ms");
                    }
                    if (currentNLoptRunResult != null && currentNLoptRunResult.ResultCode > 0 && 
                        (!overallBestNloptResult.HasValue || currentNLoptRunResult.OptimalValue < overallBestNloptResult.Value.OptimalValue))
                    {
                        overallBestNloptResult = new TempNLoptResult { 
                            Algorithm = algo, Parameters = currentNLoptRunResult.OptimalParameters, 
                            OptimalValue = currentNLoptRunResult.OptimalValue, ResultCode = currentNLoptRunResult.ResultCode,
                            Context = context, TimeMs = watchNlopt.ElapsedMilliseconds, Evaluations = currentNLoptRunResult.Evaluations
                        };
                        if (!csvOutputOnly) Console.WriteLine($"(This NLopt result is currently best for plot - from {algo} {context})");
                    }
                }
            }
            
            if (!csvOutputOnly && overallBestNloptResult.HasValue) {
                 Console.WriteLine($"\n--- Best NLopt result overall for plot (from {overallBestNloptResult.Value.Algorithm} - {overallBestNloptResult.Value.Context}) ---");
                 PrintParameters("Best NLopt Plot", overallBestNloptResult.Value.Parameters);
                 Console.WriteLine($"Best NLopt SSR: {overallBestNloptResult.Value.OptimalValue:E2}");
            }
            
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
                double yFittedNloptToPlot = (overallBestNloptResult.HasValue && overallBestNloptResult.Value.Parameters != null) ? DoubleGaussModel.Calculate(currentX, overallBestNloptResult.Value.Parameters) : double.NaN;
                double yNoisy = yData[i];
                Console.WriteLine($"{currentX:F2},{yTrue:F4},{yFittedOur:F4},{yFittedNloptToPlot:F4},{yNoisy:F4}");
            }
        }

        private static void PrintParameters(string label, ReadOnlySpan<double> parameters, bool csvOutputOnly = false) 
        {
            bool isPlotCriticalLabel = label.Contains("Best NLopt Plot") || label.Contains("True") || label.Contains("Initial") || label.Contains("Our Found") || label.StartsWith("NLoptWrapper");
            if (csvOutputOnly && !isPlotCriticalLabel) return;
            
            if (parameters == null || parameters.IsEmpty) {
                 if (!csvOutputOnly) Console.WriteLine($"{label}: Parameters are null or empty.");
                 else Console.WriteLine($"# {label}: Parameters are null or empty.");
                return;
            }

            if (csvOutputOnly && isPlotCriticalLabel) { 
                 Console.WriteLine($"# {label}: A1={parameters[0]:F2}, mu1={parameters[1]:F2}, sigma1={parameters[2]:F2}, " +
                                  $"A2={parameters[3]:F2}, mu2={parameters[4]:F2}, sigma2={parameters[5]:F2}");
                 return;
            }
            
            if (parameters.Length == 6)
            {
                Console.WriteLine($"{label}: A1={parameters[0]:F2}, mu1={parameters[1]:F2}, sigma1={parameters[2]:F2}, " +
                                  $"A2={parameters[3]:F2}, mu2={parameters[4]:F2}, sigma2={parameters[5]:F2}");
            }
            else
            {
                if(!csvOutputOnly) Console.WriteLine($"{label}: Invalid parameter count {parameters.Length} for {label}.");
                 else Console.WriteLine($"# {label}: Invalid parameter count {parameters.Length}.");
            }
        }
    }
}