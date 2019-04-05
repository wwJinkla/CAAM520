% specs for K40 GPU (note: K80s are 2x K40s)
bw = 480; % GB/s in single precision
flops = 8.226 * 1e3; % gigaflops in single precision

% axes of roofline plot
roofY = [0, flops, flops]; 
roofX = [0, flops/bw, 30]; % max (GFLOPS/s) / (GB/s) bandwidth 



%% Reduce2 kernels 


%Reduce2 kernel 1024
bw = [0.399, 0.455, 0.526];
gflops_R1024 = [9.78e-4, 6.12e-4, 3.96e-4]; 
arith_intensity_R1024 = gflops_R1024./bw;

% Reduce2 kernel 256
bw = [0.44,0.55,0.61];
gflops_R256 = [6.63e-4, 3.97e-4,2.45e-4]; 
arith_intensity_R256 = gflops_R256./bw;


%% Jacobi kernels 


%J kernel 1024
bw = [14.28, 15.28, 19.62];
gflops_J1024 = [2.38e-3, 2.01e-3, 1.27e-3]; 
arith_intensity_J1024 = gflops_J1024./bw;

% J kernel 256
bw = [16.29,14.94,19.35];
gflops_J256 = [2.31e-3, 1.75e-3,1.24e-3]; 
arith_intensity_J256 = gflops_J256./bw;


%% plot points on roofline plot

plot(roofX,roofY,'linewidth',2)
hold on


plot(arith_intensity_J1024,gflops_J1024,'^','linewidth',2,'markersize',12)
text(arith_intensity_J1024,gflops_J1024,{'J1024-100','J1024-150', 'J1024-200'})
plot(arith_intensity_J256,gflops_J256,'o','linewidth',2,'markersize',12)
text(arith_intensity_J256,gflops_J256,{'J256-100','J256-150', 'J256-200'})

plot(arith_intensity_R1024,gflops_R1024,'.','linewidth',2,'markersize',12)
text(arith_intensity_R1024,gflops_R1024,{'R1024-100','R1024-150', 'R1024-200'})
plot(arith_intensity_R256,gflops_R256,'*','linewidth',2,'markersize',12)
text(arith_intensity_R256,gflops_R256,{'R256-100','R256-150', 'R256-200'})


set(gca,'fontsize',14)
xlabel('Arithmetic intensity','fontsize',14)
ylabel('GFLOPS/sec','fontsize',14)
grid on
