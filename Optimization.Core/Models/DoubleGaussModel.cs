using System;

namespace Optimization.Core.Models
{
    /// <summary>
    /// Provides methods for calculating a double Gaussian function and its sum of squared residuals.
    /// The model is y(x) = A1 * exp(- (x - mu1)^2 / (2 * sigma1^2)) + A2 * exp(- (x - mu2)^2 / (2 * sigma2^2)).
    /// </summary>
    public static class DoubleGaussModel
    {
        /// <summary>
        /// Calculates the value of a double Gaussian function at a given point x.
        /// Parameters are: [A1, mu1, sigma1, A2, mu2, sigma2]
        /// </summary>
        /// <param name="x">The x-coordinate at which to evaluate the function.</param>
        /// <param name="parameters">A ReadOnlySpan of 6 doubles representing the parameters:
        /// <list type="bullet">
        /// <item><description>parameters[0] (A1): Amplitude of the first Gaussian.</description></item>
        /// <item><description>parameters[1] (mu1): Mean (center) of the first Gaussian.</description></item>
        /// <item><description>parameters[2] (sigma1): Standard deviation of the first Gaussian (must be positive).</description></item>
        /// <item><description>parameters[3] (A2): Amplitude of the second Gaussian.</description></item>
        /// <item><description>parameters[4] (mu2): Mean (center) of the second Gaussian.</description></item>
        /// <item><description>parameters[5] (sigma2): Standard deviation of the second Gaussian (must be positive).</description></item>
        /// </list>
        /// </param>
        /// <returns>The value of the double Gaussian function at x. Returns double.NaN if sigma1 or sigma2 is non-positive.</returns>
        /// <exception cref="ArgumentException">Thrown if the length of <paramref name="parameters"/> is not 6.</exception>
        public static double Calculate(double x, ReadOnlySpan<double> parameters)
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
        /// Calculates the sum of squared residuals (SSR) between observed y-data and the double Gaussian model.
        /// This is typically used as the objective function for optimization algorithms.
        /// Non-positive sigma values in <paramref name="parameters"/> will result in double.MaxValue being returned to penalize such parameters.
        /// </summary>
        /// <param name="parameters">A ReadOnlySpan of 6 doubles representing the parameters of the double Gaussian model (A1, mu1, sigma1, A2, mu2, sigma2).</param>
        /// <param name="xData">A ReadOnlySpan of doubles representing the observed x-coordinates.</param>
        /// <param name="yData">A ReadOnlySpan of doubles representing the observed y-coordinates.</param>
        /// <returns>The sum of squared residuals. Returns double.MaxValue if sigma1 or sigma2 is non-positive (<= 1e-9).</returns>
        /// <exception cref="ArgumentException">Thrown if <paramref name="parameters"/> length is not 6, or if <paramref name="xData"/> and <paramref name="yData"/> have different lengths.</exception>
        public static double SumSquaredResiduals(ReadOnlySpan<double> parameters, ReadOnlySpan<double> xData, ReadOnlySpan<double> yData)
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
            int i = 0;
            int n = xData.Length;

            for (; i < n; i++)
            {
                double modelY = Calculate(xData[i], parameters);
                double residual = yData[i] - modelY;
                sumSqRes += residual * residual;
            }

            return sumSqRes;
        }
    }
} 