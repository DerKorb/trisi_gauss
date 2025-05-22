using System;
using System.Runtime.InteropServices; // For MemoryMarshal if using Spans with flat array
using System.Runtime.CompilerServices; // For MethodImplOptions

namespace Optimization.Core.Algorithms
{
    // Delegate for the objective function, now specifically for double
    public delegate double ObjectiveFunctionDouble(ReadOnlySpan<double> parameters);

    public static class NelderMeadDouble
    {
        // Nelder-Mead algorithm parameters (common default values)
        private const double Alpha = 1.0; // Reflection coefficient
        private const double Gamma = 2.0; // Expansion coefficient
        private const double Rho = 0.5;   // Contraction coefficient
        private const double Sigma = 0.5;   // Shrink coefficient

        public static ReadOnlySpan<double> Minimize(
            ObjectiveFunctionDouble objectiveFunction,
            ReadOnlySpan<double> initialParameters,
            double step,
            int maxIterations,
            double tolerance)
        {
            if (objectiveFunction == null)
                throw new ArgumentNullException(nameof(objectiveFunction));
            if (initialParameters.IsEmpty)
                throw new ArgumentException("Initial parameters cannot be empty.", nameof(initialParameters));

            int dimensions = initialParameters.Length;
            int numVertices = dimensions + 1;

            // Simplex: (N+1) vertices, each with N dimensions, stored in a flat array.
            // Vertex j (0 to N) starts at index j * dimensions.
            double[] simplex = new double[numVertices * dimensions];
            double[] fValues = new double[numVertices]; // Function value at each vertex
            int[] order = new int[numVertices];     // To store sorted indices of simplex vertices

            InitializeSimplex(objectiveFunction, initialParameters, step, simplex, fValues, dimensions);

            // Pre-allocate spans for temporary points outside the loop to potentially reduce stack pressure/overhead
            Span<double> centroid = stackalloc double[dimensions];
            Span<double> reflectedPoint = stackalloc double[dimensions];
            Span<double> expandedPoint = stackalloc double[dimensions];
            Span<double> contractedPoint = stackalloc double[dimensions];

            // Initial sort to establish order for the first iteration and identify initial best/worst
            OrderSimplex(fValues, order); 

            int iBest = 0, iWorst = 0, iNextWorst = 0; // Will be updated by FindMinMaxNextMax

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                // Find indices of best, worst, and second worst points for current iteration
                FindMinMaxNextMax(fValues, out iBest, out iWorst, out iNextWorst);

                double fBestVal = fValues[iBest];
                double fWorstVal = fValues[iWorst]; 
                if (Math.Abs(fWorstVal - fBestVal) < tolerance)
                {
                    break; 
                }

                // Centroid is calculated based on all points except the current iWorst
                CalculateCentroidExcludingOne(simplex, iWorst, dimensions, numVertices, centroid);

                int worstPointSimplexStartIndex = iWorst * dimensions;

                // Reflection: P_r = P_c + Alpha * (P_c - P_w)
                for (int j = 0; j < dimensions; ++j)
                {
                    reflectedPoint[j] = centroid[j] + Alpha * (centroid[j] - simplex[worstPointSimplexStartIndex + j]);
                }
                double fReflected = objectiveFunction(reflectedPoint);

                if (fReflected < fValues[iBest]) 
                {
                    // Expansion: P_e = P_c + Gamma * (P_r - P_c)
                    for (int j = 0; j < dimensions; ++j)
                    {
                        expandedPoint[j] = centroid[j] + Gamma * (reflectedPoint[j] - centroid[j]);
                    }
                    double fExpanded = objectiveFunction(expandedPoint);

                    if (fExpanded < fReflected)
                    {
                        expandedPoint.CopyTo(simplex.AsSpan().Slice(worstPointSimplexStartIndex, dimensions));
                        fValues[iWorst] = fExpanded;
                    }
                    else
                    {
                        reflectedPoint.CopyTo(simplex.AsSpan().Slice(worstPointSimplexStartIndex, dimensions));
                        fValues[iWorst] = fReflected;
                    }
                }
                else if (fReflected < fValues[iNextWorst]) // If reflected point is better than second worst
                {
                    reflectedPoint.CopyTo(simplex.AsSpan().Slice(worstPointSimplexStartIndex, dimensions));
                    fValues[iWorst] = fReflected;
                }
                else 
                {
                    bool performInsideContraction = fReflected >= fValues[iWorst];
                    
                    if (performInsideContraction)
                    {
                        // Inside Contraction: P_ic = P_c + Rho * (P_w - P_c)
                        for (int j = 0; j < dimensions; ++j)
                        {
                            contractedPoint[j] = centroid[j] + Rho * (simplex[worstPointSimplexStartIndex + j] - centroid[j]);
                        }
                    }
                    else
                    {
                        // Outside Contraction: P_oc = P_c + Rho * (P_r - P_c)
                         for (int j = 0; j < dimensions; ++j)
                        {
                            contractedPoint[j] = centroid[j] + Rho * (reflectedPoint[j] - centroid[j]);
                        }
                    }
                    double fContracted = objectiveFunction(contractedPoint);

                    if (fContracted < (performInsideContraction ? fValues[iWorst] : fReflected) )
                    {
                        contractedPoint.CopyTo(simplex.AsSpan().Slice(worstPointSimplexStartIndex, dimensions));
                        fValues[iWorst] = fContracted;
                    }
                    else
                    {
                        // Shrink operation needs the best point's actual index (iBest from FindMinMaxNextMax)
                        ShrinkSimplexTowardsBest(simplex, iBest, objectiveFunction, fValues, dimensions, Sigma);
                    }
                }
            }

            // Final sort to ensure the absolute best is returned (in case FindMinMaxNextMax logic isn't perfect or for consistency)
            OrderSimplex(fValues, order); 
            return simplex.AsSpan().Slice(order[0] * dimensions, dimensions).ToArray(); 
        }

        private static void InitializeSimplex(
            ObjectiveFunctionDouble objectiveFunction,
            ReadOnlySpan<double> initialParameters,
            double step,
            Span<double> simplex, // Flat array: numVertices * dimensions
            Span<double> fValues,
            int dimensions)
        {
            int numVertices = dimensions + 1;

            // First vertex is the initial parameters
            initialParameters.CopyTo(simplex.Slice(0, dimensions));
            fValues[0] = objectiveFunction(simplex.Slice(0, dimensions));

            // Generate N other vertices by perturbing each dimension
            for (int i = 0; i < dimensions; i++)
            {
                int vertexStartIndex = (i + 1) * dimensions;
                initialParameters.CopyTo(simplex.Slice(vertexStartIndex, dimensions));
                simplex[vertexStartIndex + i] += step;
                fValues[i + 1] = objectiveFunction(simplex.Slice(vertexStartIndex, dimensions));
            }
        }

        private static void OrderSimplex(
            ReadOnlySpan<double> fValues,
            Span<int> order) // Output: indices sorted from best (lowest fValue) to worst
        {
            var indexedValues = new (double Value, int Index)[fValues.Length];
            for (int i = 0; i < fValues.Length; i++)
            {
                indexedValues[i] = (fValues[i], i);
            }
            Array.Sort(indexedValues, (a, b) => a.Value.CompareTo(b.Value));
            for (int i = 0; i < indexedValues.Length; i++)
            {
                order[i] = indexedValues[i].Index;
            }
        }

        // New method to find best, worst, and second worst indices
        private static void FindMinMaxNextMax(ReadOnlySpan<double> fValues, out int iMin, out int iMax, out int iNextMax)
        {
            iMin = 0;
            iMax = 0;
            iNextMax = 0; // Placeholder, will be properly assigned

            double fMinVal = fValues[0];
            double fMaxVal = fValues[0];
            
            if (fValues.Length == 0) return; // Should not happen with N+1 vertices
            if (fValues.Length == 1) { iNextMax = 0; return; }

            // Find min and max
            for (int i = 1; i < fValues.Length; i++)
            {
                if (fValues[i] < fMinVal)
                {
                    fMinVal = fValues[i];
                    iMin = i;
                }
                if (fValues[i] > fMaxVal)
                {
                    fMaxVal = fValues[i];
                    iMax = i;
                }
            }

            // Find second max (iNextMax)
            if (iMax == 0) // if max is the first element
            {
                iNextMax = 1;
                double fNextMaxVal = fValues[1];
                for(int i = 2; i < fValues.Length; ++i)
                {
                    if(fValues[i] > fNextMaxVal) { fNextMaxVal = fValues[i]; iNextMax = i;}
                }
            }
            else // max is not the first element, so fValues[0] is a candidate for second max
            {
                iNextMax = 0;
                double fNextMaxVal = fValues[0];
                 for(int i = 1; i < fValues.Length; ++i)
                {
                    if(i == iMax) continue;
                    if(fValues[i] > fNextMaxVal) { fNextMaxVal = fValues[i]; iNextMax = i;}
                }
            }
        }

        // Modified CalculateCentroid to exclude a specific vertex by its actual index
        private static void CalculateCentroidExcludingOne(
            ReadOnlySpan<double> simplex, 
            int excludedVertexActualIndex, 
            int dimensions,
            int numTotalVertices,
            Span<double> centroidOutput)
        {
            centroidOutput.Fill(0.0);
            int pointsSummed = 0;
            for (int i = 0; i < numTotalVertices; i++) 
            {
                if (i == excludedVertexActualIndex) continue;
                
                int vertexStartIndexInSimplex = i * dimensions;
                for (int j = 0; j < dimensions; j++)
                {
                    centroidOutput[j] += simplex[vertexStartIndexInSimplex + j];
                }
                pointsSummed++;
            }
            if (pointsSummed > 0)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    centroidOutput[j] /= pointsSummed;
                }
            }
        }
        
        // Modified Shrink to operate with the best point's actual index
        private static void ShrinkSimplexTowardsBest(
            Span<double> simplex,       
            int bestPointActualIndex, 
            ObjectiveFunctionDouble objectiveFunction,
            Span<double> fValues,
            int dimensions,
            double sigmaCoefficient)
        {
            int bestPointStartIndexInSimplex = bestPointActualIndex * dimensions;

            for (int i = 0; i < fValues.Length; i++) 
            {
                if (i == bestPointActualIndex) continue; // Don't shrink the best point

                int vertexToShrinkStartIndex = i * dimensions;
                for (int j = 0; j < dimensions; j++)
                {
                    simplex[vertexToShrinkStartIndex + j] = simplex[bestPointStartIndexInSimplex + j] +
                                                          sigmaCoefficient * (simplex[vertexToShrinkStartIndex + j] - simplex[bestPointStartIndexInSimplex + j]);
                }
                fValues[i] = objectiveFunction(simplex.Slice(vertexToShrinkStartIndex, dimensions));
            }
        }
    }
} 