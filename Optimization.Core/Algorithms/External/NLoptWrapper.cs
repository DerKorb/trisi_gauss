using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Linq; // Added for LINQ .Select method
using Optimization.Core.Algorithms; // Added to use the common ObjectiveFunctionDouble

namespace Optimization.Core.Algorithms.External
{
    // Delegate for the objective function to be minimized, specific to double precision parameters.
    // This is the C# delegate that our models will use.
    // ObjectiveFunctionDoubleForNLopt delegate is REMOVED from here.
    // We will use Optimization.Core.Algorithms.ObjectiveFunctionDouble

    public static class NLoptWrapper
    {
        private const string LibName = "nlopt"; // Or "libnlopt.so"

        // --- NLopt Enums (subset) ---
        public enum nlopt_algorithm : int
        {
            NLOPT_LN_NELDERMEAD = 11, // From nlopt.h, there are many others
            NLOPT_LN_SBPLX = 12, // Added SBPLX (Subplex algorithm)
            NLOPT_LN_COBYLA = 9, // Added for completeness from earlier tests
            // Add other algorithms as needed
        }

        public enum nlopt_result : int
        {
            NLOPT_FAILURE = -1,             /* generic failure code */
            NLOPT_INVALID_ARGS = -2,
            NLOPT_OUT_OF_MEMORY = -3,
            NLOPT_ROUNDOFF_LIMITED = -4,
            NLOPT_FORCED_STOP = -5,
            NLOPT_SUCCESS = 1,              /* generic success code */
            NLOPT_STOPVAL_REACHED = 2,
            NLOPT_FTOL_REACHED = 3,
            NLOPT_XTOL_REACHED = 4,
            NLOPT_MAXEVAL_REACHED = 5,
            NLOPT_MAXTIME_REACHED = 6
        }

        // --- NLopt Delegate for Objective Function ---
        // typedef double (*nlopt_func)(unsigned n, const double *x, double *gradient, void *func_data);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate double nlopt_func_delegate_csharp_style(uint n, IntPtr x_ptr, IntPtr gradient_ptr, IntPtr func_data_ptr);

        // --- P/Invoke Signatures ---
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr nlopt_create(nlopt_algorithm algorithm, uint n);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nlopt_destroy(IntPtr opt);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_min_objective(IntPtr opt, nlopt_func_delegate_csharp_style f, IntPtr f_data);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_lower_bounds(IntPtr opt, double[] lb);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_upper_bounds(IntPtr opt, double[] ub);
        
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_ftol_rel(IntPtr opt, double tol);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_maxeval(IntPtr opt, int maxeval);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_optimize(IntPtr opt, double[] x, out double minf);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr nlopt_get_algorithm_name(nlopt_algorithm algorithm); // For error messages

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nlopt_set_local_optimizer(IntPtr opt, IntPtr local_opt); // For some global algos

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_initial_step(IntPtr opt, double[] dx); // Added for initial step

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_xtol_rel(IntPtr opt, double tol); // Added for xtol_rel

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_xtol_abs(IntPtr opt, double[] tol_array); // Added for xtol_abs

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_ftol_abs(IntPtr opt, double tol);     // Added for ftol_abs

        // --- Helper to convert nlopt_result to string ---
        public static string ResultToString(nlopt_result result)
        {
             return result switch
            {
                nlopt_result.NLOPT_FAILURE => "Generic failure",
                nlopt_result.NLOPT_INVALID_ARGS => "Invalid arguments",
                nlopt_result.NLOPT_OUT_OF_MEMORY => "Out of memory",
                nlopt_result.NLOPT_ROUNDOFF_LIMITED => "Roundoff errors limited progress",
                nlopt_result.NLOPT_FORCED_STOP => "Forced stop",
                nlopt_result.NLOPT_SUCCESS => "Generic success",
                nlopt_result.NLOPT_STOPVAL_REACHED => "Stopval reached",
                nlopt_result.NLOPT_FTOL_REACHED => "Relative function value tolerance reached",
                nlopt_result.NLOPT_XTOL_REACHED => "Relative parameter tolerance reached",
                nlopt_result.NLOPT_MAXEVAL_REACHED => "Max evaluations reached",
                nlopt_result.NLOPT_MAXTIME_REACHED => "Max time reached",
                _ => "Unknown result code"
            };
        }
        
        // --- Public Optimization Method ---
        public class NLoptResultData
        {
            public double[] OptimalParameters { get; set; }
            public double OptimalValue { get; set; }
            public nlopt_result ResultCode { get; set; }
            public string ResultMessage { get; set; }
            public int Evaluations { get; set; }
        }

        // Store the delegate to prevent garbage collection if it's only referenced by native code
        private static nlopt_func_delegate_csharp_style _cachedObjectiveDelegate;
        private static ObjectiveFunctionDouble _userObjectiveFunction; // Now uses the common delegate
        private static int _currentEvalCount; // To track evaluations for logging
        private static bool _enableNLoptLogging = false;
        private static string _currentNLoptAlgoName = "";

        // This is the actual delegate passed to NLopt C API
        private static double NLoptObjectiveCallback(uint n, IntPtr x_ptr, IntPtr gradient_ptr, IntPtr func_data_ptr)
        {
            _currentEvalCount++;
            double[] x = new double[n];
            Marshal.Copy(x_ptr, x, 0, (int)n);
            double ssr = _userObjectiveFunction(x.AsSpan()); 
            if (_enableNLoptLogging && (_currentEvalCount <= 100 || _currentEvalCount % 500 == 0) )
            {
                if (_currentNLoptAlgoName == "NLOPT_LN_COBYLA" && !(_currentEvalCount <= 20 || _currentEvalCount % 1000 == 0)) {
                    // Skip logging for COBYLA unless it's in the first 20 or a multiple of 1000
                } else {
                    Console.WriteLine($"  NLopt {_currentNLoptAlgoName} Eval #{_currentEvalCount}: SSR={ssr:E3} Params=[{string.Join(", ", x.Select(val => val.ToString("F2")))}]");
                }
            }
            return ssr;
        }

        // Method for NelderMead (keeping it separate for clarity)
        public static NLoptResultData OptimizeNelderMead(
            ObjectiveFunctionDouble objectiveFunction, 
            double[] initialParameters,
            double[] lowerBounds,
            double[] upperBounds,
            double[] initialStepArray, 
            double ftol_rel = 1e-7, 
            double xtol_rel = 1e-7, 
            double ftol_abs = 1e-8, 
            double xtol_abs_val = 1e-8, 
            int maxeval = 20000, bool verboseLogging = false)
        {
            return RunNloptAlgorithm(nlopt_algorithm.NLOPT_LN_NELDERMEAD, objectiveFunction, initialParameters,
                                     lowerBounds, upperBounds, initialStepArray, 
                                     ftol_rel, xtol_rel, ftol_abs, xtol_abs_val, maxeval, verboseLogging);
        }

        // New method for SBPLX
        public static NLoptResultData OptimizeSbplx(
            ObjectiveFunctionDouble objectiveFunction, 
            double[] initialParameters,
            double[] lowerBounds,
            double[] upperBounds,
            double[] initialStepArray, // SBPLX also uses initial step
            double ftol_rel = 1e-7, 
            double xtol_rel = 1e-7, 
            double ftol_abs = 1e-8, 
            double xtol_abs_val = 1e-8, 
            int maxeval = 20000, bool verboseLogging = false)
        {
            return RunNloptAlgorithm(nlopt_algorithm.NLOPT_LN_SBPLX, objectiveFunction, initialParameters,
                                     lowerBounds, upperBounds, initialStepArray, 
                                     ftol_rel, xtol_rel, ftol_abs, xtol_abs_val, maxeval, verboseLogging);
        }

        // Common private method to run an NLopt algorithm
        private static NLoptResultData RunNloptAlgorithm(
            nlopt_algorithm algorithm,
            ObjectiveFunctionDouble objectiveFunction, // Uses the common delegate
            double[] initialParameters,
            double[] lowerBounds,
            double[] upperBounds,
            double[] initialStepArray, 
            double ftol_rel, 
            double xtol_rel, 
            double ftol_abs, 
            double xtol_abs_val, 
            int maxeval, bool verboseLogging)
        {
            _userObjectiveFunction = objectiveFunction; // Store for callback
            _currentEvalCount = 0;
            _enableNLoptLogging = verboseLogging;
            _currentNLoptAlgoName = algorithm.ToString();

            uint numParams = (uint)initialParameters.Length;
            IntPtr opt = IntPtr.Zero;
            // Ensure the delegate is not collected prematurely if this method is called multiple times rapidly
            // and the static field _cachedObjectiveDelegate gets overwritten before native code finishes with the previous one.
            // For simplicity in this context, we rely on _cachedObjectiveDelegate being set just before nlopt_set_min_objective.
            // In a multi-threaded scenario or more complex lifecycle, more robust GCHandle management might be needed per call.
            _cachedObjectiveDelegate = NLoptObjectiveCallback; // Cache it to prevent GC

            try
            {
                opt = nlopt_create(algorithm, numParams);
                if (opt == IntPtr.Zero) throw new Exception($"Failed to create NLopt optimizer for {algorithm}.");
                
                nlopt_set_min_objective(opt, _cachedObjectiveDelegate, IntPtr.Zero);

                if (lowerBounds != null && lowerBounds.Length == numParams) nlopt_set_lower_bounds(opt, lowerBounds);
                if (upperBounds != null && upperBounds.Length == numParams) nlopt_set_upper_bounds(opt, upperBounds);
                
                if (initialStepArray != null) 
                {
                    if (initialStepArray.Length == numParams) nlopt_set_initial_step(opt, initialStepArray);
                    else throw new ArgumentException("initialStepArray length must match dimensions if not null.", nameof(initialStepArray));
                }

                nlopt_set_ftol_rel(opt, ftol_rel);
                nlopt_set_xtol_rel(opt, xtol_rel);
                nlopt_set_ftol_abs(opt, ftol_abs);
                
                double[] xtol_abs_array = new double[numParams];
                for(int i=0; i<numParams; ++i) xtol_abs_array[i] = xtol_abs_val;
                nlopt_set_xtol_abs(opt, xtol_abs_array);

                nlopt_set_maxeval(opt, maxeval);

                double[] x_opt = (double[])initialParameters.Clone(); 
                double minf;
                
                nlopt_result result = nlopt_optimize(opt, x_opt, out minf);

                return new NLoptResultData
                {
                    OptimalParameters = x_opt,
                    OptimalValue = minf,
                    ResultCode = result,
                    ResultMessage = ResultToString(result),
                    Evaluations = _currentEvalCount
                };
            }
            catch (Exception ex)
            {
                 return new NLoptResultData
                {
                    OptimalParameters = (double[])initialParameters.Clone(), // Return initial if failed
                    OptimalValue = double.PositiveInfinity,
                    ResultCode = nlopt_result.NLOPT_FAILURE,
                    ResultMessage = $"Exception in NLoptWrapper ({algorithm}): " + ex.Message,
                    Evaluations = _currentEvalCount
                };
            }
            finally
            {
                if (opt != IntPtr.Zero) nlopt_destroy(opt);
                _userObjectiveFunction = null; // Clear static ref
            }
        }

        public static NLoptResultData OptimizeCobyla(
            ObjectiveFunctionDouble objectiveFunction, double[] initialParameters, double[] lowerBounds, double[] upperBounds,
            double[] initialStepArray, double ftol_rel = 1e-7, double xtol_rel = 1e-7, 
            double ftol_abs = 1e-8, double xtol_abs_val = 1e-4, int maxeval = 50000, bool verboseLogging = false)
        {
            return RunNloptAlgorithm(nlopt_algorithm.NLOPT_LN_COBYLA, objectiveFunction, initialParameters,
                                     lowerBounds, upperBounds, initialStepArray, 
                                     ftol_rel, xtol_rel, ftol_abs, xtol_abs_val, maxeval, verboseLogging);
        }
    }
} 