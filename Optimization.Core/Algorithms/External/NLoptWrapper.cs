using System;
using System.Runtime.InteropServices;
using System.Text;

namespace Optimization.Core.Algorithms.External
{
    public static class NLoptWrapper
    {
        private const string LibName = "nlopt"; // Or "libnlopt.so"

        // --- NLopt Enums (subset) ---
        public enum nlopt_algorithm : int
        {
            NLOPT_LN_NELDERMEAD = 11, // From nlopt.h, there are many others
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
        public delegate double nlopt_func_delegate(uint n, IntPtr x_ptr, IntPtr gradient_ptr, IntPtr func_data_ptr);

        // --- P/Invoke Signatures ---
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr nlopt_create(nlopt_algorithm algorithm, uint n);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void nlopt_destroy(IntPtr opt);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        private static extern nlopt_result nlopt_set_min_objective(IntPtr opt, nlopt_func_delegate f, IntPtr f_data);

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
        }

        // Store the delegate to prevent garbage collection if it's only referenced by native code
        private static nlopt_func_delegate _cachedObjectiveDelegate;

        public static NLoptResultData OptimizeNelderMead(
            ObjectiveFunctionDouble objectiveFunction, // Our C# objective function
            double[] initialParameters,
            double[] lowerBounds,
            double[] upperBounds,
            double ftol_rel = 1e-8,
            int maxeval = 20000)
        {
            uint n = (uint)initialParameters.Length;
            IntPtr opt = IntPtr.Zero;
            try
            {
                opt = nlopt_create(nlopt_algorithm.NLOPT_LN_NELDERMEAD, n);
                if (opt == IntPtr.Zero) throw new Exception("Failed to create NLopt optimizer.");

                // Wrap our C# objective function for NLopt
                // The GCHandle is used to pass a reference to our C# objective function state (if any, not used here)
                // to the C callback. We are not using func_data here, so IntPtr.Zero.
                _cachedObjectiveDelegate = (num_params, x_native_ptr, gradient_native_ptr, func_data_native_ptr) =>
                {
                    double[] x = new double[num_params];
                    Marshal.Copy(x_native_ptr, x, 0, (int)num_params);
                    // NLopt's NelderMead doesn't use gradients, so gradient_native_ptr can be ignored.
                    return objectiveFunction(x);
                };
                
                nlopt_set_min_objective(opt, _cachedObjectiveDelegate, IntPtr.Zero);

                if (lowerBounds != null && lowerBounds.Length == n) nlopt_set_lower_bounds(opt, lowerBounds);
                if (upperBounds != null && upperBounds.Length == n) nlopt_set_upper_bounds(opt, upperBounds);
                
                nlopt_set_ftol_rel(opt, ftol_rel);
                nlopt_set_maxeval(opt, maxeval);

                double[] x_opt = (double[])initialParameters.Clone(); // NLopt modifies this array
                double minf;
                
                nlopt_result result = nlopt_optimize(opt, x_opt, out minf);

                return new NLoptResultData
                {
                    OptimalParameters = x_opt,
                    OptimalValue = minf,
                    ResultCode = result,
                    ResultMessage = ResultToString(result)
                };
            }
            catch (Exception ex)
            {
                 return new NLoptResultData
                {
                    ResultCode = nlopt_result.NLOPT_FAILURE,
                    ResultMessage = "Exception in NLoptWrapper: " + ex.Message
                };
            }
            finally
            {
                if (opt != IntPtr.Zero) nlopt_destroy(opt);
            }
        }
    }
} 