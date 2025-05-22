using System;
using Xunit;
using Optimization.Core.Models;

namespace Optimization.Core.Tests
{
    public class DoubleGaussModelTests
    {
        [Fact]
        public void Calculate_ValidParameters_ReturnsCorrectValue()
        {
            // A1=10, mu1=20, sigma1=3, A2=15, mu2=40, sigma2=5
            double[] parameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            double x1 = 20.0; // At first peak
            double y1_expected = 10.0 * Math.Exp(0) + 15.0 * Math.Exp(-0.5 * Math.Pow((20.0 - 40.0) / 5.0, 2));
            // y1_expected = 10.0 + 15.0 * Math.Exp(-0.5 * (-4.0)^2) = 10.0 + 15.0 * Math.Exp(-8)
            // y1_expected approx 10.0 + 15.0 * 0.000335 = 10.005025

            double x2 = 40.0; // At second peak
            double y2_expected = 10.0 * Math.Exp(-0.5 * Math.Pow((40.0 - 20.0) / 3.0, 2)) + 15.0 * Math.Exp(0);
            // y2_expected = 10.0 * Math.Exp(-0.5 * (20/3)^2) + 15.0 = 10.0 * Math.Exp(-0.5 * 44.44) + 15.0
            // y2_expected approx 10.0 * very_small_num + 15.0 = 15.0
            
            Assert.Equal(y1_expected, DoubleGaussModel.Calculate(x1, parameters), 5); // Precision to 5 decimal places
            Assert.Equal(y2_expected, DoubleGaussModel.Calculate(x2, parameters), 5);
        }

        [Fact]
        public void Calculate_InvalidSigma_ReturnsNaN()
        {
            double[] paramsSigma1Zero = { 10.0, 20.0, 0.0, 15.0, 40.0, 5.0 };
            Assert.True(double.IsNaN(DoubleGaussModel.Calculate(20.0, paramsSigma1Zero)));

            double[] paramsSigma2Negative = { 10.0, 20.0, 3.0, 15.0, 40.0, -5.0 };
            Assert.True(double.IsNaN(DoubleGaussModel.Calculate(40.0, paramsSigma2Negative)));
        }

        [Fact]
        public void Calculate_IncorrectParameterCount_ThrowsArgumentException()
        {
            double[] tooFewParams = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            Assert.Throws<ArgumentException>(() => DoubleGaussModel.Calculate(10.0, tooFewParams));

            double[] tooManyParams = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
            Assert.Throws<ArgumentException>(() => DoubleGaussModel.Calculate(10.0, tooManyParams));
        }

        [Fact]
        public void SumSquaredResiduals_PerfectFit_ReturnsZero()
        {
            double[] parameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            double[] xData = { 15, 20, 25, 35, 40, 45 };
            double[] yData = new double[xData.Length];
            for(int i=0; i<xData.Length; ++i) yData[i] = DoubleGaussModel.Calculate(xData[i], parameters);

            Assert.Equal(0.0, DoubleGaussModel.SumSquaredResiduals(parameters, xData, yData), 8);
        }

        [Fact]
        public void SumSquaredResiduals_KnownResiduals_ReturnsCorrectSum()
        {
            double[] parameters = { 10.0, 20.0, 3.0, 0.0, 0.0, 1.0 }; // Second peak amplitude is 0
            double[] xData = { 20.0 }; // Single point at peak of first Gaussian
            double[] yData = { 11.0 }; // True value is 10.0, so residual is 1.0
            // Expected SSR = (11.0 - 10.0)^2 = 1.0^2 = 1.0
            Assert.Equal(1.0, DoubleGaussModel.SumSquaredResiduals(parameters, xData, yData), 5);
        }

        [Fact]
        public void SumSquaredResiduals_InvalidSigma_ReturnsMaxValue()
        {
            double[] paramsSigma1Zero = { 10.0, 20.0, 0.0, 15.0, 40.0, 5.0 };
            double[] xData = { 20.0 };
            double[] yData = { 10.0 };
            Assert.Equal(double.MaxValue, DoubleGaussModel.SumSquaredResiduals(paramsSigma1Zero, xData, yData));

            double[] paramsSigma2Negative = { 10.0, 20.0, 3.0, 15.0, 40.0, -5.0 };
            Assert.Equal(double.MaxValue, DoubleGaussModel.SumSquaredResiduals(paramsSigma2Negative, xData, yData));
        }

        [Fact]
        public void SumSquaredResiduals_MismatchedDataLengths_ThrowsArgumentException()
        {
            double[] parameters = { 10.0, 20.0, 3.0, 15.0, 40.0, 5.0 };
            double[] xData = { 1.0, 2.0, 3.0 };
            double[] yData = { 1.0, 2.0 };
            Assert.Throws<ArgumentException>(() => DoubleGaussModel.SumSquaredResiduals(parameters, xData, yData));
        }
         [Fact]
        public void SumSquaredResiduals_IncorrectParameterCount_ThrowsArgumentException()
        {
            double[] tooFewParams = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            double[] xData = { 1.0 };
            double[] yData = { 1.0 };
            Assert.Throws<ArgumentException>(() => DoubleGaussModel.SumSquaredResiduals(tooFewParams, xData, yData));
        }
    }
} 