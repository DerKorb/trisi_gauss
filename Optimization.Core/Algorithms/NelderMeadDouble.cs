using System;
using System.Runtime.InteropServices; // For MemoryMarshal if using Spans with flat array

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

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                OrderSimplex(fValues, order); // Sorts based on fValues, populates order

                // Check for convergence (tolerance on function values)
                double fBest = fValues[order[0]];
                double fWorst = fValues[order[dimensions]]; // order[dimensions] is the worst index after sorting
                if (Math.Abs(fWorst - fBest) < tolerance)
                {
                    break; // Converged
                }

                // Calculate centroid of all points except the worst one
                // Centroid is calculated based on the N best points.
                Span<double> centroid = stackalloc double[dimensions];
                CalculateCentroid(simplex, order, dimensions, numVertices -1, centroid);

                int worstVertexIndexInSimplexArray = order[dimensions] * dimensions;
                ReadOnlySpan<double> worstPoint = simplex.AsSpan().Slice(worstVertexIndexInSimplexArray, dimensions);

                // Reflection
                Span<double> reflectedPoint = stackalloc double[dimensions];
                TransformPoint(centroid, worstPoint, Alpha, reflectedPoint, dimensions);
                double fReflected = objectiveFunction(reflectedPoint);

                if (fReflected < fValues[order[0]]) // Reflected point is better than the current best
                {
                    // Expansion
                    Span<double> expandedPoint = stackalloc double[dimensions];
                    TransformPoint(reflectedPoint, centroid, Gamma, expandedPoint, dimensions); // p_e = c + gamma * (p_r - c)
                    double fExpanded = objectiveFunction(expandedPoint);

                    if (fExpanded < fReflected)
                    {
                        expandedPoint.CopyTo(simplex.AsSpan().Slice(worstVertexIndexInSimplexArray, dimensions));
                        fValues[order[dimensions]] = fExpanded;
                    }
                    else
                    {
                        reflectedPoint.CopyTo(simplex.AsSpan().Slice(worstVertexIndexInSimplexArray, dimensions));
                        fValues[order[dimensions]] = fReflected;
                    }
                }
                else if (fReflected < fValues[order[dimensions - 1]]) // Reflected point is not better than best, but better than second worst
                {
                    reflectedPoint.CopyTo(simplex.AsSpan().Slice(worstVertexIndexInSimplexArray, dimensions));
                    fValues[order[dimensions]] = fReflected;
                }
                else // Reflected point is not better than second worst
                {
                    Span<double> contractedPoint = stackalloc double[dimensions];
                    if (fReflected < fValues[order[dimensions]]) // P_r is better than P_worst (f_r < f_worst)
                    {
                        // Outside contraction: c_ = c + rho * (p_r - c)
                        TransformPoint(reflectedPoint, centroid, Rho, contractedPoint, dimensions);
                    }
                    else // P_r is not better than P_worst (f_r >= f_worst)
                    {
                        // Inside contraction: c_ = c + rho * (p_worst - c)  (or c - rho * (c - p_worst))
                        TransformPoint(worstPoint, centroid, Rho, contractedPoint, dimensions);
                    }
                    double fContracted = objectiveFunction(contractedPoint);

                    if (fContracted < fValues[order[dimensions]]) // Contracted point is better than worst
                    {
                        contractedPoint.CopyTo(simplex.AsSpan().Slice(worstVertexIndexInSimplexArray, dimensions));
                        fValues[order[dimensions]] = fContracted;
                    }
                    else
                    {
                        // Shrink
                        Shrink(simplex, order, objectiveFunction, fValues, dimensions, Sigma);
                    }
                }
            }

            OrderSimplex(fValues, order);
            return simplex.AsSpan().Slice(order[0] * dimensions, dimensions).ToArray(); // Return a copy of the best parameters
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

        // Calculates centroid of the N_best_points (all but the worst one implicitly via `order`)
        private static void CalculateCentroid(
            ReadOnlySpan<double> simplex, // Flat array
            ReadOnlySpan<int> order,      // order[numBestPoints] is the worst point to exclude
            int dimensions,
            int numBestPoints,          // dimensions (N) for standard Nelder-Mead (N+1 total points, N best)
            Span<double> centroidOutput) // Output span for the centroid
        {
            centroidOutput.Fill(0.0);
            for (int i = 0; i < numBestPoints; i++) // Summing the N best points
            {
                int vertexActualIndex = order[i]; // Get the true index of the i-th best point
                int vertexStartIndexInSimplex = vertexActualIndex * dimensions;
                for (int j = 0; j < dimensions; j++)
                {
                    centroidOutput[j] += simplex[vertexStartIndexInSimplex + j];
                }
            }
            for (int j = 0; j < dimensions; j++)
            {
                centroidOutput[j] /= numBestPoints;
            }
        }
        
        // P_transformed = p_anchor + factor * (p_dynamic - p_anchor)
        // Example: Reflected = Centroid + Alpha * (ReflectedAnchorPoint (e.g. Centroid or ReflectedPoint) - WorstPoint)
        // Reflected: P_new = P_centroid + Alpha * (P_centroid - P_worst) -> transform(centroid, worst, Alpha, out)
        // Expanded:  P_new = P_centroid + Gamma * (P_reflected - P_centroid) -> transform(centroid, reflected, Gamma, out) WRONG: should be reflected + gamma * (reflected - centroid)
        // Corrected Expanded: P_expanded = P_reflected + Gamma * (P_reflected - P_centroid)
        // Contracted (outside): P_c = P_centroid + Rho * (P_reflected - P_centroid) -> transform(centroid, reflected, Rho, out)
        // Contracted (inside):  P_c = P_centroid + Rho * (P_worst - P_centroid) -> transform(centroid, worst, Rho, out)
        private static void TransformPoint(
            ReadOnlySpan<double> pAnchor,    // The point from which the transformation originates (e.g., centroid)
            ReadOnlySpan<double> pDynamic,   // The point that defines the direction vector (e.g., worst point or reflected point)
            double factor,                 // The coefficient (Alpha, Gamma, Rho)
            Span<double> resultPoint,      // Output: the transformed point
            int dimensions)
        {
            for (int i = 0; i < dimensions; i++)
            {
                resultPoint[i] = pAnchor[i] + factor * (pDynamic[i] - pAnchor[i]);
            }
        }

        // Reflect, Expand, Contract methods are now effectively inlined or logic directly in Minimize loop using TransformPoint

        private static void Shrink(
            Span<double> simplex,       // Flat array
            ReadOnlySpan<int> order,    // order[0] is the best point
            ObjectiveFunctionDouble objectiveFunction,
            Span<double> fValues,
            int dimensions,
            double sigmaCoefficient) // Sigma = 0.5
        {
            int bestPointActualIndex = order[0];
            int bestPointStartIndexInSimplex = bestPointActualIndex * dimensions;

            for (int i = 1; i < fValues.Length; i++) // Shrink all points except the best one
            {
                int vertexToShrinkActualIndex = order[i];
                int vertexToShrinkStartIndex = vertexToShrinkActualIndex * dimensions;
                for (int j = 0; j < dimensions; j++)
                {
                    simplex[vertexToShrinkStartIndex + j] = simplex[bestPointStartIndexInSimplex + j] +
                                                          sigmaCoefficient * (simplex[vertexToShrinkStartIndex + j] - simplex[bestPointStartIndexInSimplex + j]);
                }
                fValues[vertexToShrinkActualIndex] = objectiveFunction(simplex.Slice(vertexToShrinkStartIndex, dimensions));
            }
        }
    }
} 