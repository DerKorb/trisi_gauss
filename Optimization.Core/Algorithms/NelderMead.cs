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

            InitializeSimplex(objectiveFunction, initialParameters, step, simplex, fValues);

            // TODO: Implement main Nelder-Mead loop (Reflection, Expansion, Contraction, Shrink)
            // TODO: Implement Abbruchkriterium (Toleranz auf Funktion)

            return initialParameters; 
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

        // TODO: Implement main Nelder-Mead loop in Minimize method
    }
} 