using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Optimization.Core.Models
{
    public static unsafe class DoubleGaussModel
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
        /// Calculates the sum of squared residuals between observed y-data and the double Gaussian model.
        /// This is typically used as the objective function for optimization.
        /// </summary>
        /// <param name="parameters">The 6 parameters of the double Gaussian model: [A1, mu1, sigma1, A2, mu2, sigma2].</param>
        /// <param name="xData">The observed x-coordinates.</param>
        /// <param name="yData">The observed y-coordinates.</param>
        /// <returns>The sum of squared residuals. Returns double.MaxValue if sigma1 or sigma2 is non-positive.</returns>
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

            if (Avx.IsSupported && n >= Vector256<double>.Count) // Check for AVX support
            {
                int vectorSize = Vector256<double>.Count;
                Vector256<double> accSquaredResiduals = Vector256<double>.Zero;
                
                // Temporary array on stack for modelY values for one vector chunk
                Span<double> modelY_chunk_span = stackalloc double[vectorSize]; 

                fixed (double* pxData = &MemoryMarshal.GetReference(xData))
                fixed (double* pyData = &MemoryMarshal.GetReference(yData))
                {
                    for (i = 0; i <= n - vectorSize; i += vectorSize)
                    {
                        Vector256<double> xVec = Avx.LoadVector256(pxData + i);
                        Vector256<double> yVec = Avx.LoadVector256(pyData + i);
                        
                        // Scalar Calculate calls, then load into vector
                        // This is the bottleneck for full vectorization of this function.
                        // We are creating a Span from the xVec for Calculate, which is not ideal.
                        // A better way would be to extract scalars, but let's test this first.
                        // For a direct AVX approach, we'd need Calculate to be AVX-aware.
                        for(int j=0; j < vectorSize; ++j)
                        {
                             // Get scalar from xVec - this is inefficient, direct pointer access is better
                            modelY_chunk_span[j] = Calculate(*(pxData + i + j), parameters);
                        }
                        Vector256<double> modelYVec = Avx.LoadVector256((double*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(modelY_chunk_span)));

                        Vector256<double> residualVec = Avx.Subtract(yVec, modelYVec);
                        // FMA could be used if we had a multiply-add pattern, here it's just multiply
                        accSquaredResiduals = Avx.Add(accSquaredResiduals, Avx.Multiply(residualVec, residualVec));
                    }
                }
                // Horizontal sum of the accumulator
                for(int k=0; k<vectorSize; ++k)
                {
                    sumSqRes += accSquaredResiduals.GetElement(k);
                }
            }
            
            // Process remaining elements scalar way
            for (; i < n; i++)
            {
                double modelY = Calculate(xData[i], parameters); // xData[i] is fine here
                double residual = yData[i] - modelY;             // yData[i] is fine here
                sumSqRes += residual * residual;
            }

            return sumSqRes;
        }
    }
} 