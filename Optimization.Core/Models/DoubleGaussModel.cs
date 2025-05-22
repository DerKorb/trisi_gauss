using System;
using System.Numerics;

namespace Optimization.Core.Models
{
    public static class DoubleGaussModel
    {
        /// <summary>
        /// Calculates the value of a double Gaussian function at a given point x.
        /// Parameters are: [A1, mu1, sigma1, A2, mu2, sigma2]
        /// y(x) = A1 * exp(- (x - mu1)^2 / (2 * sigma1^2)) + A2 * exp(- (x - mu2)^2 / (2 * sigma2^2))
        /// </summary>
        /// <param name="x">The x-coordinate.</param>
        /// <param name="parameters">The 6 parameters of the double Gaussian model:
        /// A1: Amplitude of the first Gaussian.
        /// mu1: Mean (center) of the first Gaussian.
        /// sigma1: Standard deviation of the first Gaussian.
        /// A2: Amplitude of the second Gaussian.
        /// mu2: Mean (center) of the second Gaussian.
        /// sigma2: Standard deviation of the second Gaussian.
        /// </param>
        /// <returns>The value of the double Gaussian function at x.</returns>
        public static double Calculate(double x, Span<double> parameters)
        {
            if (parameters.Length != 6)
            {
                throw new ArgumentException("Double Gaussian model requires 6 parameters: [A1, mu1, sigma1, A2, mu2, sigma2].", nameof(parameters));
            }

            double a1 = parameters[0];
            double mu1 = parameters[1];
            double sigma1 = parameters[2];
            double a2 = parameters[3];
            double mu2 = parameters[4];
            double sigma2 = parameters[5];

            // Handle corner case: sigma <= 0
            if (sigma1 <= 0 || sigma2 <= 0)
            {
                // This scenario should ideally be penalized heavily by SumSquaredResiduals
                // For a direct Calculate call, behavior might depend on context.
                // Here, we might return NaN or throw, but SumSquaredResiduals will handle it by returning double.MaxValue.
                return double.NaN; // Or throw new ArgumentOutOfRangeException(nameof(parameters), "Sigma values must be positive.");
            }

            double term1 = (x - mu1) / sigma1;
            double exp1 = Math.Exp(-0.5 * term1 * term1);
            double gauss1 = a1 * exp1;

            double term2 = (x - mu2) / sigma2;
            double exp2 = Math.Exp(-0.5 * term2 * term2);
            double gauss2 = a2 * exp2;

            return gauss1 + gauss2;
        }

        /// <summary>
        /// Calculates the sum of squared residuals between observed y-data and the double Gaussian model.
        /// This is typically used as the objective function for optimization.
        /// </summary>
        /// <param name="parameters">The 6 parameters of the double Gaussian model: [A1, mu1, sigma1, A2, mu2, sigma2].</param>
        /// <param name="xData">The observed x-coordinates.</param>
        /// <param name="yData">The observed y-coordinates.</param>
        /// <returns>The sum of squared residuals. Returns double.MaxValue if sigma1 or sigma2 is non-positive.</returns>
        public static double SumSquaredResiduals(Span<double> parameters, Span<double> xData, Span<double> yData)
        {
            if (parameters.Length != 6)
            {
                throw new ArgumentException("Double Gaussian model requires 6 parameters: [A1, mu1, sigma1, A2, mu2, sigma2].", nameof(parameters));
            }
            if (xData.Length != yData.Length)
            {
                throw new ArgumentException("xData and yData must have the same length.", nameof(xData));
            }

            double sigma1 = parameters[2];
            double sigma2 = parameters[5];

            // Penalize non-positive sigma values heavily.
            if (sigma1 <= 1e-9 || sigma2 <= 1e-9) // Using a small epsilon to avoid issues with exactly zero
            {
                return double.MaxValue;
            }

            double sumSqRes = 0.0;

            // SIMD optimization for SumSquaredResiduals
            if (Vector.IsHardwareAccelerated && xData.Length >= Vector<double>.Count)
            {
                int vectorSize = Vector<double>.Count;
                int i = 0;
                // Process chunks of vectorSize
                for (i = 0; i <= xData.Length - vectorSize; i += vectorSize)
                {
                    // Prepare vectors for xData and yData
                    Span<double> xSlice = xData.Slice(i, vectorSize);
                    Span<double> ySlice = yData.Slice(i, vectorSize);
                    Vector<double> xVec = new Vector<double>(xSlice);
                    Vector<double> yVec = new Vector<double>(ySlice);

                    // Calculate modelY for each element in xVec (scalar calls, then construct vector)
                    // This is the part not fully vectorized due to Math.Exp in Calculate
                    double[] modelY_chunk_array = new double[vectorSize];
                    for(int j=0; j < vectorSize; ++j)
                    {
                        modelY_chunk_array[j] = Calculate(xSlice[j], parameters); 
                    }
                    Vector<double> modelYVec = new Vector<double>(modelY_chunk_array);
                    
                    Vector<double> residualVec = yVec - modelYVec;
                    Vector<double> squaredResidualVec = residualVec * residualVec;
                    sumSqRes += Vector.Dot(squaredResidualVec, Vector<double>.One);
                }

                // Process remaining elements scalar way
                for (; i < xData.Length; i++)
                {
                    double modelY = Calculate(xData[i], parameters);
                    double residual = yData[i] - modelY;
                    sumSqRes += residual * residual;
                }
            }
            else
            {
                // Standard loop-based calculation (fallback if SIMD not accelerated or too few elements)
                for (int i = 0; i < xData.Length; i++)
                {
                    double modelY = Calculate(xData[i], parameters);
                    double residual = yData[i] - modelY;
                    sumSqRes += residual * residual;
                }
            }

            return sumSqRes;
        }
    }
} 