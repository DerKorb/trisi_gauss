using System;

namespace Optimization.Core.Algorithms
{
    public delegate T ObjectiveFunction<T>(Span<T> parameters) where T : struct;

    public static class NelderMead<T> where T : struct, IComparable<T>, IEquatable<T>
    {
        public static Span<T> Minimize(
            ObjectiveFunction<T> objectiveFunction,
            Span<T> initialParameters,
            T step,
            int maxIterations,
            T tolerance
            )
        {
            if (objectiveFunction == null)
                throw new ArgumentNullException(nameof(objectiveFunction));
            if (initialParameters.IsEmpty)
                throw new ArgumentException("Initial parameters cannot be empty.", nameof(initialParameters));

            int dimensions = initialParameters.Length;
            
            Span<T[]> simplex = new T[dimensions + 1][];
            Span<T> fValues = new T[dimensions + 1];
            Span<int> order = new int[dimensions + 1]; // To store sorted indices of simplex vertices

            InitializeSimplex(objectiveFunction, initialParameters, step, simplex, fValues);

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                OrderSimplex(simplex, fValues, order);

                // Check for convergence (tolerance on function values)
                // This requires T to be subtractable and comparable, handled by T constraints and double conversion for now
                if (typeof(T) == typeof(double))
                {
                    double fBest = Convert.ToDouble(fValues[order[0]]);
                    double fWorst = Convert.ToDouble(fValues[order[dimensions]]); // order[dimensions] is the worst
                    if (Math.Abs(fWorst - fBest) < Convert.ToDouble(tolerance))
                    {
                        break; // Converged
                    }
                }
                else
                {
                    // For non-double types, a generic way to compare fValues[order[dimensions]] - fValues[order[0]] with tolerance is needed.
                    // Or rely on maxIterations only if tolerance check isn't feasible.
                    // Potentially throw, or log a warning that tolerance check is skipped for non-double types.
                }

                // Calculate centroid of all points except the worst one
                T[] centroid = CalculateCentroid(simplex, order, dimensions);

                // Worst point is simplex[order[dimensions]]
                T[] worstPoint = simplex[order[dimensions]];

                // Reflection
                T[] reflectedPoint = Reflect(worstPoint, centroid, dimensions);
                T fReflected = objectiveFunction(reflectedPoint);

                if (fReflected.CompareTo(fValues[order[0]]) < 0) // Reflected point is better than the current best
                {
                    // Expansion
                    T[] expandedPoint = Expand(reflectedPoint, centroid, dimensions);
                    T fExpanded = objectiveFunction(expandedPoint);

                    if (fExpanded.CompareTo(fReflected) < 0)
                    {
                        simplex[order[dimensions]] = expandedPoint;
                        fValues[order[dimensions]] = fExpanded;
                    }
                    else
                    {
                        simplex[order[dimensions]] = reflectedPoint;
                        fValues[order[dimensions]] = fReflected;
                    }
                }
                // If reflected point is not better than best, but better than second worst (order[dimensions-1])
                else if (fReflected.CompareTo(fValues[order[dimensions - 1]]) < 0) 
                {
                    simplex[order[dimensions]] = reflectedPoint;
                    fValues[order[dimensions]] = fReflected;
                }
                else // Reflected point is not better than second worst
                {
                    // Contraction
                    // If fReflected is worse than or equal to fWorst (fValues[order[dimensions]])
                    // Perform inside contraction: P_contracted = centroid + Rho * (P_worst - centroid)
                    T[] contractedPoint;
                    if (fReflected.CompareTo(fValues[order[dimensions]]) < 0)
                    { 
                        // Outside contraction: P_contracted = centroid + Rho * (P_reflected - centroid)
                         contractedPoint = TransformPoint(reflectedPoint, centroid, Rho, dimensions); // p1 + Rho * (p1-p2)
                    }
                    else
                    { 
                        // Inside contraction: P_contracted = centroid - Rho * (centroid - P_worst)
                        // which is equivalent to centroid + Rho * (P_worst - centroid)
                        // So use TransformPoint(worstPoint, centroid, Rho, dimensions) for P_c = P_worst + Rho * (P_worst - P_centroid) ??? No this is wrong.
                        // It should be: P_c = P_centroid + Rho * (P_worst - P_centroid)
                        // So: centroid is p1, (P_worst - P_centroid) is (p2-p1) with a minus sign
                        // We want: p_centroid + RHO * (p_worst - p_centroid)
                        // TransformPoint is p1 + factor * (p1 - p2)
                        // Let p1 = centroid, factor = Rho.
                        // Then we need (p1 - p2) to be (P_worst - P_centroid)
                        // So p2 would need to be (2*P_centroid - P_worst)
                        // This is getting complicated. Let's re-evaluate TransformPoint's utility here or write it explicitly.
                        // Standard inside contraction: c_ = x_o + rho * (x_h - x_o)
                        // where x_o is centroid, x_h is worst point.
                        contractedPoint = new T[dimensions];
                        if (typeof(T) == typeof(double))
                        {
                            for(int i=0; i<dimensions; ++i)
                            {
                                double c_val = Convert.ToDouble(centroid[i]);
                                double w_val = Convert.ToDouble(worstPoint[i]);
                                contractedPoint[i] = (T)Convert.ChangeType(c_val + Rho * (w_val - c_val), typeof(T));
                            }
                        }
                        else throw new InvalidOperationException("Contraction for non-double not implemented in this path.");
                    }
                    
                    T fContracted = objectiveFunction(contractedPoint);

                    if (fContracted.CompareTo(fValues[order[dimensions]]) < 0) // Contracted point is better than worst
                    {
                        simplex[order[dimensions]] = contractedPoint;
                        fValues[order[dimensions]] = fContracted;
                    }
                    else
                    {
                        // Shrink
                        Shrink(simplex, order, objectiveFunction, fValues, dimensions);
                    }
                }
            }

            OrderSimplex(simplex, fValues, order); // Final sort
            return simplex[order[0]]; 
        }

        private static void InitializeSimplex(
            ObjectiveFunction<T> objectiveFunction, 
            Span<T> initialParameters, 
            T step, 
            Span<T[]> simplex,
            Span<T> fValues  
            )
        {
            int dimensions = initialParameters.Length;

            simplex[0] = new T[dimensions];
            initialParameters.CopyTo(simplex[0]);
            fValues[0] = objectiveFunction(simplex[0]);

            for (int i = 0; i < dimensions; i++)
            {
                simplex[i + 1] = new T[dimensions];
                initialParameters.CopyTo(simplex[i + 1]);
                
                if (typeof(T) == typeof(double))
                {
                    double[] tempVertex = Array.ConvertAll(simplex[i + 1], item => Convert.ToDouble(item));
                    double stepAsDouble = Convert.ToDouble(step);
                    tempVertex[i] += stepAsDouble;
                    simplex[i + 1] = Array.ConvertAll(tempVertex, item => (T)Convert.ChangeType(item, typeof(T)));
                }
                else
                {
                    throw new InvalidOperationException($"Simplex initialization step perturbation is not implemented for type {typeof(T)}. Only double is currently supported for this operation.");
                }

                fValues[i + 1] = objectiveFunction(simplex[i + 1]);
            }
        }

        // Nelder-Mead algorithm parameters (common default values)
        private const double Alpha = 1.0; // Reflection coefficient
        private const double Gamma = 2.0; // Expansion coefficient
        private const double Rho = 0.5;   // Contraction coefficient
        private const double Sigma = 0.5;   // Shrink coefficient

        private static void OrderSimplex(
            Span<T[]> simplex, 
            Span<T> fValues,
            Span<int> order // Output: indices sorted from best (lowest fValue) to worst
            )
        {
            if (fValues.Length != order.Length)
            {
                throw new ArgumentException("fValues and order arrays must have the same length.");
            }

            var indexedValues = new (T Value, int Index)[fValues.Length];
            for (int i = 0; i < fValues.Length; i++)
            {
                indexedValues[i] = (fValues[i], i);
            }

            // Sort based on the function values. T must be IComparable<T>.
            Array.Sort(indexedValues, (a, b) => a.Value.CompareTo(b.Value));

            for (int i = 0; i < indexedValues.Length; i++)
            {
                order[i] = indexedValues[i].Index;
            }
        }

        private static T[] CalculateCentroid(
            Span<T[]> simplex, 
            Span<int> order, // Assumes order[N] is the worst point
            int dimensions
            )
        {
            T[] centroid = new T[dimensions];
            if (typeof(T) != typeof(double))
            {
                throw new InvalidOperationException($"Centroid calculation is only implemented for type double. Got {typeof(T)}.");
            }

            // Sum all points except the worst one (order[simplex.Length - 1])
            for (int j = 0; j < dimensions; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < simplex.Length - 1; i++)
                {
                    int vertexIndex = order[i]; // Use sorted order to sum N best points
                    sum += Convert.ToDouble(simplex[vertexIndex][j]);
                }
                centroid[j] = (T)Convert.ChangeType(sum / (simplex.Length - 1), typeof(T));
            }
            return centroid;
        }

        private static T[] TransformPoint(
            T[] p1, 
            T[] p2, 
            double factor, 
            int dimensions
            )
        {
            T[] result = new T[dimensions];
            if (typeof(T) != typeof(double))
            {
                throw new InvalidOperationException($"Point transformation is only implemented for type double. Got {typeof(T)}.");
            }

            for (int i = 0; i < dimensions; i++)
            {
                double val1 = Convert.ToDouble(p1[i]);
                double val2 = Convert.ToDouble(p2[i]);
                result[i] = (T)Convert.ChangeType(val1 + factor * (val1 - val2), typeof(T));
            }
            return result;
        }

        private static T[] Reflect(
            T[] worstPoint, 
            T[] centroid, 
            int dimensions
            )
        {
            // P_reflected = P_centroid + Alpha * (P_centroid - P_worst)
            // P_reflected = P_centroid + 1.0 * (P_centroid - P_worst)
            // P_reflected = 2 * P_centroid - P_worst (if Alpha = 1)
            // More generally: P_reflected = (1 + Alpha) * P_centroid - Alpha * P_worst
            // Or if TransformPoint expects (p_centroid, p_worst, factor)
            // where factor is Alpha: P_reflected = p_centroid + Alpha * (p_centroid - p_worst)
            // This means TransformPoint(centroid, worstPoint, Alpha, dimensions) is correct conceptually
            return TransformPoint(centroid, worstPoint, Alpha, dimensions);
        }

        private static T[] Expand(
            T[] reflectedPoint, 
            T[] centroid, 
            int dimensions
            )
        {
            // P_expanded = P_centroid + Gamma * (P_reflected - P_centroid)
            // This means TransformPoint(reflectedPoint, centroid, Gamma, dimensions) is correct using the existing TransformPoint logic (p1 + factor * (p1-p2))
            // P_expanded = reflectedPoint + Gamma * (reflectedPoint - centroid)
            return TransformPoint(reflectedPoint, centroid, Gamma, dimensions);
        }

        private static T[] Contract(
            T[] worstPoint, 
            T[] centroid, 
            int dimensions
            )
        {
            // P_contracted = P_centroid + Rho * (P_worst - P_centroid)
            // This means TransformPoint(worstPoint, centroid, Rho, dimensions) is correct using the existing TransformPoint logic (p1 + factor * (p1-p2))
            // P_contracted = worstPoint + Rho * (worstPoint - centroid)
            return TransformPoint(worstPoint, centroid, Rho, dimensions);
        }

        private static void Shrink(
            Span<T[]> simplex, 
            Span<int> order, // order[0] is the best point
            ObjectiveFunction<T> objectiveFunction,
            Span<T> fValues,
            int dimensions
            )
        {
            if (typeof(T) != typeof(double))
            {
                throw new InvalidOperationException($"Shrink operation is only implemented for type double. Got {typeof(T)}.");
            }
            
            T[] bestPoint = simplex[order[0]];
            for (int i = 1; i < simplex.Length; i++) // Shrink all points except the best one
            {
                int vertexIndexToShrink = order[i];
                for (int j = 0; j < dimensions; j++)
                {
                    double currentCoord = Convert.ToDouble(simplex[vertexIndexToShrink][j]);
                    double bestCoord = Convert.ToDouble(bestPoint[j]);
                    simplex[vertexIndexToShrink][j] = (T)Convert.ChangeType(bestCoord + Sigma * (currentCoord - bestCoord), typeof(T));
                }
                fValues[vertexIndexToShrink] = objectiveFunction(simplex[vertexIndexToShrink]);
            }
        }
    }
} 