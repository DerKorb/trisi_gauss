set terminal pngcairo enhanced font "arial,10" size 1024,768
set output 'double_gauss_fit_comparison_nlopt.png'

set title "Double Gaussian Fit Comparison (Bounded Optimizers)\n(15% Noise, Bad Initial Guess)"
set xlabel "X-Value"
set ylabel "Y-Value"
set key top right box opaque

set datafile separator comma

# Parameter summary from last run (for context):
# True Parameters: A1=10.00, mu1=20.00, sigma1=3.00, A2=15.00, mu2=40.00, sigma2=5.00
# Our Found:     A1=10.01, mu1=20.01, sigma1=3.13, A2=14.77, mu2=40.18, sigma2=4.82 (SSR: 1.95E+02)
# NLopt Found:   A1=1.00, mu1=10.00, sigma1=1.00, A2=1.00, mu2=50.00, sigma2=10.00 (SSR: 3.87E+03)

plot 'plot_data.csv' using 1:5 with points pointtype 7 pointsize 0.6 lc rgb "#cccccc" title 'Noisy Data (15%)', \
     'plot_data.csv' using 1:2 with lines linewidth 2.5 lc rgb "#0060ad" title 'True Function', \
     'plot_data.csv' using 1:3 with lines linewidth 2 lc rgb "#d95319" dashtype '-' title 'Our Bounded Fit', \
     'plot_data.csv' using 1:4 with lines linewidth 2 lc rgb "#77ac30" dashtype '.' title 'NLopt Bounded Fit'

set output # Close output file
print "Plot 'double_gauss_fit_comparison_nlopt.png' generated."
print "Please ensure 'plot_data.csv' exists in the same directory." 