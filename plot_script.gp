set terminal pngcairo enhanced font "arial,10" size 1024,768
set output 'double_gauss_fit_comparison.png'

set title "Double Gaussian Fit Comparison\n(20% Noise, Bad Initial Guess)"
set xlabel "X-Value"
set ylabel "Y-Value"
set key top right box opaque

set datafile separator comma

# Parameter summary (for context, not directly plotted by gnuplot from here)
# True Parameters: A1=10.00, mu1=20.00, sigma1=3.00, A2=15.00, mu2=40.00, sigma2=5.00
# Our Found:     A1=10.71, mu1=13.46, sigma1=0.07, A2=14.58, mu2=40.21, sigma2=4.85 (SSR: 1.27E+03)
# MathNet Found: A1=-5.10, mu1=-43.07, sigma1=6.83, A2=14.58, mu2=40.21, sigma2=4.85 (SSR: 1.28E+03)

plot 'plot_data.csv' using 1:5 with points pointtype 7 pointsize 0.6 lc rgb "#cccccc" title 'Noisy Data (20%)', \
     'plot_data.csv' using 1:2 with lines linewidth 2.5 lc rgb "#0060ad" title 'True Function', \
     'plot_data.csv' using 1:3 with lines linewidth 2 lc rgb "#d95319" dashtype '-' title 'Our Fit (NelderMeadDouble)', \
     'plot_data.csv' using 1:4 with lines linewidth 2 lc rgb "#77ac30" dashtype '.' title 'MathNet Fit'

set output # Close output file
print "Plot 'double_gauss_fit_comparison.png' generated."
print "Please ensure 'plot_data.csv' exists in the same directory." 