using System;
using Optimization.Core.Algorithms;
using Optimization.Core.Models;

namespace Optimization.Benchmarks // Re-using namespace for simplicity, can be a separate project
{
    public static class DoubleGaussFitExample
    {
        public static void RunExample()
        {
            Console.WriteLine("Running Double Gaussian Fit Example...");

            // 1. Define True Parameters for data generation
            double[] trueParameters = { 
                10.0,  // A1
                20.0,  // mu1
                3.0,   // sigma1
                15.0,  // A2
                40.0,  // mu2
                5.0    // sigma2
            };
            Console.WriteLine($"True Parameters: A1={trueParameters[0]:F2}, mu1={trueParameters[1]:F2}, sigma1={trueParameters[2]:F2}, " +
                              $"A2={trueParameters[3]:F2}, mu2={trueParameters[4]:F2}, sigma2={trueParameters[5]:F2}");

            // 2. Generate Synthetic Data
            int numDataPoints = 100;
            double[] xData = new double[numDataPoints];
            double[] yData = new double[numDataPoints];
            Random random = new Random(123); // For reproducibility
            double noisePercentage = 0.10; // 10% noise relative to max amplitude

            double xMin = 0;
            double xMax = 60;
            double maxAmplitude = Math.Max(trueParameters[0], trueParameters[3]);

            for (int i = 0; i < numDataPoints; i++)
            {
                xData[i] = xMin + (xMax - xMin) * i / (numDataPoints - 1);
                double cleanY = DoubleGaussModel.Calculate(xData[i], trueParameters);
                double noise = (random.NextDouble() * 2 - 1) * maxAmplitude * noisePercentage; // Noise centered around 0
                yData[i] = cleanY + noise;
            }

            // 3. Define an Objective Function using SumSquaredResiduals
            ObjectiveFunction<double> objectiveFunction = (pars) => 
                DoubleGaussModel.SumSquaredResiduals(pars, xData, yData);

            // 4. Define Initial Guess for Parameters
            // Let's start with parameters somewhat offset from the true ones
            double[] initialParameters = {
                8.0,  // A1 guess
                15.0, // mu1 guess
                2.0,  // sigma1 guess
                12.0, // A2 guess
                35.0, // mu2 guess
                4.0   // sigma2 guess
            };
             Console.WriteLine($"Initial Guess: A1={initialParameters[0]:F2}, mu1={initialParameters[1]:F2}, sigma1={initialParameters[2]:F2}, " +
                              $"A2={initialParameters[3]:F2}, mu2={initialParameters[4]:F2}, sigma2={initialParameters[5]:F2}");


            // 5. Run the Optimizer
            Console.WriteLine("Starting optimization...");
            Span<double> bestParameters = NelderMead<double>.Minimize(
                objectiveFunction, 
                initialParameters, 
                step: 0.5,        // Initial step for simplex construction
                maxIterations: 20000, // Increased iterations for potentially noisy data
                tolerance: 1e-8
            );

            // 6. Display Results
            Console.WriteLine("Optimization finished.");
            Console.WriteLine($"Found Parameters: A1={bestParameters[0]:F2}, mu1={bestParameters[1]:F2}, sigma1={bestParameters[2]:F2}, " +
                              $"A2={bestParameters[3]:F2}, mu2={bestParameters[4]:F2}, sigma2={bestParameters[5]:F2}");

            double finalSSR = DoubleGaussModel.SumSquaredResiduals(bestParameters, xData, yData);
            Console.WriteLine($"Final Sum of Squared Residuals: {finalSSR:E2}");

            // Optional: Visualize (basic console plot)
            Console.WriteLine("\nBasic Console Visualization (Y vs X - Original vs Fitted):");
            Console.WriteLine("X\tY_Orig\tY_Fit\tY_Data(Noisy)");
            for(int i=0; i<numDataPoints; i+= numDataPoints/20) // Print a subset of points
            {
                double yOriginal = DoubleGaussModel.Calculate(xData[i], trueParameters);
                double yFitted = DoubleGaussModel.Calculate(xData[i], bestParameters);
                Console.WriteLine($"{xData[i]:F1}\t{yOriginal:F2}\t{yFitted:F2}\t{yData[i]:F2}");
            }
        }
    }
} 