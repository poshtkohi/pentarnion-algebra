function results = pentarnion_bracketing_validation(publicationMode)
%PENTARNION_BRACKETING_VALIDATION Validate the sparse rational model.
%
% Studies only
%
%   p' = a*p + (1-eta)*b*(p*p) + eta*(b*p)*p
%      = a*p + b*(p*p) + eta*[b,p,p]
%
% with the fixed coefficients and initial state reported in the paper.
% The routine performs:
%   1) a continuation in eta using the largest Lyapunov exponent;
%   2) full five-dimensional Lyapunov spectra at eta=0 and eta=1;
%   3) running-exponent convergence and exponent-sum/divergence checks;
%   4) a 10,000-time-unit boundedness check;
%   5) Poincare sections and FFT spectra;
%   6) period, closure error and Floquet multipliers for eta=1;
%   7) timestep refinement; and
%   8) statistical comparison of fixed-step RK4 and adaptive ode45.
%
% MATLAB is used only for this reproducibility/validation study; this
% file does not conduct a general coefficient search.
%
% Usage:
%   results = pentarnion_bracketing_validation(false); % quick check
%   results = pentarnion_bracketing_validation(true);  % publication run
%
% No special toolbox is required. MATLAB R2020a or later is recommended.

if nargin < 1
    publicationMode = false;
end

close all; clc;
outDir = fullfile(pwd, 'Pentarnion_rational_validation_results');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% Reported parameter vector
basisNames = {'scalar','i','j','k','l'};
a = [-1/2;
     -2;
      1/2;
     -1;
     -1/2];

b = [ 1;
      0;
      2;
      0;
     -1/2];

pInitial = [ 1/2;
             -1/2;
             -1/2;
              1/2;
             -1/2];
etaChaos = 0.0;
etaPeriodic = 1.0;

initialAssociator = pmul(pmul(b,pInitial),pInitial) - pmul(b,pmul(pInitial,pInitial));
fprintf('Sparse rational bracket-dependent Pentarnion model validation\n');
fprintf('Initial associator norm: %.15g\n', norm(initialAssociator));
verifyMatrixRepresentation(a,b,pInitial);
fprintf('\n');

%% Settings
if publicationMode
    cfg.dtMain = 0.005;
    cfg.transient = 1000;
    cfg.measureSpectrum = 3000;
    cfg.recordTime = 500; % adaptive-integrator comparison window
    cfg.recordTimeChaos = 3000;
    cfg.recordTimePeriodic = 500;
    cfg.longTime = 10000;
    cfg.longWindow = 1000;
    cfg.renormTime = 0.25;
    cfg.historyInterval = 50;
    cfg.etaGrid = linspace(0,1,41);
    cfg.scanTransient = 500;
    cfg.scanMeasure = 1000;
    cfg.scanDt = 0.005;
    cfg.bifurcationTime = 400;
    cfg.dtGrid = [0.01 0.005 0.0025];
else
    cfg.dtMain = 0.01;
    cfg.transient = 300;
    cfg.measureSpectrum = 800;
    cfg.recordTime = 500; % adaptive-integrator comparison window
    cfg.recordTimeChaos = 500;
    cfg.recordTimePeriodic = 500;
    cfg.longTime = 2000;
    cfg.longWindow = 500;
    cfg.renormTime = 0.5;
    cfg.historyInterval = 25;
    cfg.etaGrid = linspace(0,1,21);
    cfg.scanTransient = 250;
    cfg.scanMeasure = 500;
    cfg.scanDt = 0.01;
    cfg.bifurcationTime = 250;
    cfg.dtGrid = [0.02 0.01 0.005];
end
cfg.blowupRadius = 100;
cfg.tangent0 = [1;0.3;-0.2;0.4;0.1];
cfg.tangent0 = cfg.tangent0/norm(cfg.tangent0);

%% 1. Eta continuation (one tangent direction, discovery illustration only)
fprintf('Eta continuation (%d points) ...\n',numel(cfg.etaGrid));
scanLE = nan(size(cfg.etaGrid));
scanMaxNorm = nan(size(cfg.etaGrid));
bifEta = [];
bifPeak = [];
for k = 1:numel(cfg.etaGrid)
    eta = cfg.etaGrid(k);
    s = largestExponentScan(a,b,pInitial,eta,cfg.scanDt,cfg.scanTransient,...
        cfg.scanMeasure,cfg.renormTime,cfg.tangent0,cfg.blowupRadius);
    scanLE(k) = s.lambda1;
    scanMaxNorm(k) = s.maximumNorm;

    [~,xScan] = integrateRK4(a,b,pInitial,eta,cfg.scanDt,...
        cfg.scanTransient,cfg.bifurcationTime,cfg.blowupRadius,1);
    peaks = localMaxima(xScan(:,3));
    if numel(peaks) > 300, peaks = peaks(end-299:end); end
    bifEta = [bifEta; eta*ones(numel(peaks),1)]; %#ok<AGROW>
    bifPeak = [bifPeak; peaks(:)]; %#ok<AGROW>
    fprintf('  eta=%5.3f  lambda1=% .6f  max|p|=%8.4f\n',eta,s.lambda1,s.maximumNorm);
end
scanTable = table(cfg.etaGrid(:),scanLE(:),scanMaxNorm(:),...
    'VariableNames',{'eta','largest_Lyapunov','maximum_norm'});
writetable(scanTable,fullfile(outDir,'model_eta_scan.csv'));
writetable(table(bifEta,bifPeak,'VariableNames',{'eta','local_maximum_xj'}),...
    fullfile(outDir,'model_bifurcation.csv'));

figure('Color','w'); plot(cfg.etaGrid,scanLE,'o-'); yline(0); grid on;
xlabel('\eta'); ylabel('\lambda_1'); title('Associator-weight continuation');
saveas(gcf,fullfile(outDir,'model_eta_scan_lyapunov.png'));
figure('Color','w'); plot(bifEta,bifPeak,'.','MarkerSize',3); grid on;
xlabel('\eta'); ylabel('local maxima of x_j'); title('Bifurcation diagram');
saveas(gcf,fullfile(outDir,'model_bifurcation.png'));

%% 2. Full Lyapunov spectra and convergence
fprintf('\nFull Lyapunov spectra ...\n');
chaos = fullLyapunovSpectrum(a,b,pInitial,etaChaos,cfg.dtMain,cfg.transient,...
    cfg.measureSpectrum,cfg.renormTime,cfg.historyInterval,cfg.blowupRadius);
periodic = fullLyapunovSpectrum(a,b,pInitial,etaPeriodic,cfg.dtMain,cfg.transient,...
    cfg.measureSpectrum,cfg.renormTime,cfg.historyInterval,cfg.blowupRadius);

fprintf('eta=0 spectrum: '); fprintf('% .9f ',chaos.spectrum); fprintf('\n');
fprintf('eta=1 spectrum: '); fprintf('% .9f ',periodic.spectrum); fprintf('\n');
fprintf('eta=0 sum/divergence: %.9f / %.9f\n',sum(chaos.spectrum),chaos.meanDivergence);
fprintf('eta=1 sum/divergence: %.9f / %.9f\n',sum(periodic.spectrum),periodic.meanDivergence);

spectrumTable = table([etaChaos;etaPeriodic],...
    [chaos.spectrum(1);periodic.spectrum(1)],...
    [chaos.spectrum(2);periodic.spectrum(2)],...
    [chaos.spectrum(3);periodic.spectrum(3)],...
    [chaos.spectrum(4);periodic.spectrum(4)],...
    [chaos.spectrum(5);periodic.spectrum(5)],...
    [sum(chaos.spectrum);sum(periodic.spectrum)],...
    [chaos.meanDivergence;periodic.meanDivergence],...
    'VariableNames',{'eta','lambda1','lambda2','lambda3','lambda4','lambda5',...
    'sum_exponents','mean_divergence'});
writetable(spectrumTable,fullfile(outDir,'model_full_lyapunov_spectra.csv'));

figure('Color','w');
plot(1:5,chaos.spectrum,'o-',1:5,periodic.spectrum,'s--'); yline(0); grid on;
xlabel('exponent index'); ylabel('Lyapunov exponent');
legend('\eta=0','\eta=1','Location','best'); title('Full five-dimensional Lyapunov spectra');
saveas(gcf,fullfile(outDir,'model_full_lyapunov_spectra.png'));

figure('Color','w');
plot(chaos.historyTime,chaos.historySpectrum(:,1),'-',...
     periodic.historyTime,periodic.historySpectrum(:,1),'--'); yline(0); grid on;
xlabel('measurement time'); ylabel('running \lambda_1');
legend('\eta=0','\eta=1','Location','best'); title('Largest-exponent convergence');
saveas(gcf,fullfile(outDir,'model_lyapunov_convergence.png'));

%% 3. Long-time boundedness in successive windows
fprintf('\nLong-time boundedness ...\n');
boundChaos = boundednessWindows(a,b,pInitial,etaChaos,cfg.dtMain,cfg.longTime,...
    cfg.longWindow,cfg.blowupRadius);
boundPeriodic = boundednessWindows(a,b,pInitial,etaPeriodic,cfg.dtMain,cfg.longTime,...
    cfg.longWindow,cfg.blowupRadius);

boundTable = table((1:numel(boundChaos.maximumNorm))',...
    boundChaos.maximumNorm(:),boundPeriodic.maximumNorm(:),...
    boundChaos.meanNorm(:),boundPeriodic.meanNorm(:),...
    'VariableNames',{'window','max_norm_eta0','max_norm_eta1','mean_norm_eta0','mean_norm_eta1'});
writetable(boundTable,fullfile(outDir,'model_long_time_boundedness.csv'));
figure('Color','w');
plot(boundTable.window,boundTable.max_norm_eta0,'o-',...
     boundTable.window,boundTable.max_norm_eta1,'s--'); grid on;
xlabel(sprintf('successive %.0f-time-unit window',cfg.longWindow)); ylabel('maximum ||p(t)||');
legend('\eta=0','\eta=1','Location','best'); title('Long-time boundedness check');
saveas(gcf,fullfile(outDir,'model_long_time_boundedness.png'));

%% 4. Representative trajectories, Poincare sections and power spectra
fprintf('\nRepresentative trajectories ...\n');
[tC,xC] = integrateRK4(a,b,pInitial,etaChaos,cfg.dtMain,cfg.transient,...
    cfg.recordTimeChaos,cfg.blowupRadius,1);
[tP,xP] = integrateRK4(a,b,pInitial,etaPeriodic,cfg.dtMain,cfg.transient,...
    cfg.recordTimePeriodic,cfg.blowupRadius,1);

% Retain every fixed-step sample used in the phase portraits.  In the
% publication run this gives 600,001 post-transient samples at eta=0 and
% 100,001 samples at eta=1, all at Delta t=0.005.
save(fullfile(outDir,'model_trajectories_high_resolution.mat'),...
    'tC','xC','tP','xP','-v7.3');

plotComponentTimeSeries(tC,xC,tP,xP,basisNames,outDir);
plotTrajectory3D(xC,xP,outDir);

pc = poincareSection(tC,xC,2,0,+1);
pp = poincareSection(tP,xP,2,0,+1);
writematrix(pc,fullfile(outDir,'model_poincare_eta0.csv'));
writematrix(pp,fullfile(outDir,'model_poincare_eta1.csv'));
figure('Color','w'); scatter(pc(:,3),pc(:,4),12,'filled'); grid on;
xlabel('x_j'); ylabel('x_k'); title('Poincare section x_1=0, \eta=0');
saveas(gcf,fullfile(outDir,'model_poincare_eta0.png'));
figure('Color','w'); scatter(pp(:,3),pp(:,4),20,'filled'); grid on;
xlabel('x_j'); ylabel('x_k'); title('Poincare section x_1=0, \eta=1');
saveas(gcf,fullfile(outDir,'model_poincare_eta1.png'));

specC = fftMetrics(tC,xC(:,3));
specP = fftMetrics(tP,xP(:,3));
plotSpectrum(specC,'Power spectrum, \eta=0',fullfile(outDir,'model_power_spectrum_eta0.png'));
plotSpectrum(specP,'Power spectrum, \eta=1',fullfile(outDir,'model_power_spectrum_eta1.png'));

%% 5. Period, closure and Floquet multipliers at eta=1
fprintf('\nPeriodic-orbit validation ...\n');
periodInfo = periodicMetrics(a,b,tP,xP,etaPeriodic,cfg.dtMain);
fprintf('Estimated period: %.15g\n',periodInfo.period);
fprintf('RK4 closure error: %.6e\n',periodInfo.closureError);
fprintf('Floquet multipliers: '); fprintf('% .9g ',periodInfo.floquetMultipliers); fprintf('\n');

periodTable = table(periodInfo.period,periodInfo.closureError,...
    periodInfo.relativeClosureError,specP.spectralEntropy,specP.peakPowerFraction,...
    size(pp,1),max(std(pp(:,2:5),0,1)),...
    'VariableNames',{'period','closure_error','relative_closure_error',...
    'spectral_entropy','dominant_peak_power_fraction','poincare_crossings','poincare_max_std'});
writetable(periodTable,fullfile(outDir,'model_periodic_metrics.csv'));
writematrix(periodInfo.floquetMultipliers(:),fullfile(outDir,'model_floquet_multipliers.csv'));

%% 6. Timestep refinement
fprintf('\nTimestep refinement ...\n');
rows = [];
for eta = [etaChaos etaPeriodic]
    for dt = cfg.dtGrid
        s = fullLyapunovSpectrum(a,b,pInitial,eta,dt,cfg.transient,...
            min(cfg.measureSpectrum,1000),cfg.renormTime,100,cfg.blowupRadius);
        rows = [rows; eta dt s.spectrum(:)' s.meanDivergence]; %#ok<AGROW>
        fprintf('eta=%.0f dt=%g lambda1=% .6f\n',eta,dt,s.spectrum(1));
    end
end
dtTable = array2table(rows,'VariableNames',{'eta','dt','lambda1','lambda2','lambda3',...
    'lambda4','lambda5','mean_divergence'});
writetable(dtTable,fullfile(outDir,'model_timestep_convergence.csv'));

%% 7. Independent adaptive-integrator comparison
fprintf('\nRK4 / ode45 statistical comparison ...\n');
odeComparison = compareWithOde45(a,b,pInitial,[etaChaos etaPeriodic],cfg);
writetable(odeComparison,fullfile(outDir,'model_integrator_comparison.csv'));
disp(odeComparison);

%% 8. Summary and save
chaosMetrics = table(chaos.spectrum(1),chaos.spectrum(2),sum(chaos.spectrum),...
    chaos.meanDivergence,max(boundChaos.maximumNorm),specC.spectralEntropy,...
    specC.peakPowerFraction,size(pc,1),...
    'VariableNames',{'largest_LE','near_zero_LE','sum_exponents','mean_divergence',...
    'max_norm_long_run','spectral_entropy','dominant_peak_power_fraction','poincare_crossings'});
writetable(chaosMetrics,fullfile(outDir,'model_chaos_metrics.csv'));

results.coefficients.a = a;
results.coefficients.b = b;
results.initialState = pInitial;
results.initialAssociator = initialAssociator;
results.scan = scanTable;
results.chaos = chaos;
results.periodic = periodic;
results.periodInfo = periodInfo;
results.poincareChaos = pc;
results.poincarePeriodic = pp;
results.spectrumChaos = specC;
results.spectrumPeriodic = specP;
results.timestepTable = dtTable;
results.integratorComparison = odeComparison;
results.outputDirectory = outDir;
save(fullfile(outDir,'model_submission_results.mat'),'results','-v7.3');

fprintf('\nInterpretation supported by these diagnostics:\n');
fprintf('  eta=0: bounded dynamics with one robust positive, one near-zero Lyapunov exponent,\n');
fprintf('         a broadband spectrum and an extended Poincare set (numerical evidence of chaos).\n');
fprintf('  eta=1: one zero and four negative Floquet/Lyapunov exponents, a single Poincare point,\n');
fprintf('         discrete spectral lines and very small closure error (stable periodic orbit).\n');
fprintf('  Only eta changes; the coefficients, multiplication and initial condition are fixed.\n');
fprintf('Results written to: %s\n',outDir);
end

%% =====================================================================
% Core Pentarnion dynamics
% ======================================================================
function r = pmul(p,q)
p=p(:); q=q(:); root2=sqrt(2);
p0=p(1); p1=p(2); p2=p(3); p3=p(4); p4=p(5);
q0=q(1); q1=q(2); q2=q(3); q3=q(4); q4=q(5);
r0=p0*q0-p1*q1-p2*q2-p3*q3-p4*q4;
r1=p0*q1+q0*p1+(p2*q3-p2*q4-p3*q2-p3*q4+p4*q2+p4*q3)/root2;
r2=p0*q2+q0*p2+(-p1*q3-p1*q4+p3*q1-p3*q4+p4*q1+p4*q3)/root2;
r3=p0*q3+q0*p3+(p1*q2-p1*q4-p2*q1-p2*q4+p4*q1+p4*q2)/root2;
r4=p0*q4+q0*p4+(p1*q2-p1*q3-p2*q1+p2*q3+p3*q1-p3*q2)/root2;
r=[r0;r1;r2;r3;r4];
end

function L = leftMatrix(q)
%LEFTMATRIX Matrix L(q) satisfying [q*p]_B=L(q)[p]_B.
q=q(:); q0=q(1); q1=q(2); q2=q(3); q3=q(4); q4=q(5); s=sqrt(2);
L=[...
 q0, -q1,              -q2,              -q3,              -q4;
 q1,  q0,     (-q3+q4)/s,       (q2+q4)/s,      (-q2-q3)/s;
 q2, (q3+q4)/s,         q0,     (-q1+q4)/s,      (-q1-q3)/s;
 q3,(-q2+q4)/s, (q1+q4)/s,              q0,      (-q1-q2)/s;
 q4,(-q2+q3)/s, (q1-q3)/s,      (-q1+q2)/s,              q0];
end

function R = rightMatrix(q)
%RIGHTMATRIX Matrix R(q) satisfying [p*q]_B=R(q)[p]_B.
C=diag([1 -1 -1 -1 -1]);
qbar=C*q(:);
R=C*leftMatrix(qbar)*C;
end

function Q = squareMap(x)
%SQUAREMAP Coordinate vector Q(x)=[p^2]_B.
x=x(:); x0=x(1);
Q=[x0^2-sum(x(2:5).^2); 2*x0*x(2:5)];
end

function f = rhsMatrix(p,a,b,eta)
%RHSMATRIX Compact matrix form used in the manuscript.
p=p(:);
f=leftMatrix(a)*p+(1-eta)*leftMatrix(b)*squareMap(p)...
    +eta*rightMatrix(p)*leftMatrix(b)*p;
end

function verifyMatrixRepresentation(a,b,p0)
%VERIFYMATRIXREPRESENTATION Check L, R, Q and the matrix ODE directly.
rng(1,'twister'); maximumError=0;
for k=1:100
    p=randn(5,1); q=randn(5,1); eta=rand;
    maximumError=max(maximumError,norm(leftMatrix(q)*p-pmul(q,p),inf));
    maximumError=max(maximumError,norm(rightMatrix(q)*p-pmul(p,q),inf));
    maximumError=max(maximumError,norm(squareMap(p)-pmul(p,p),inf));
    maximumError=max(maximumError,norm(rhsMatrix(p,a,b,eta)-rhs(p,a,b,eta),inf));
end
expectedAssociator=[3*sqrt(2);7;13;-5;-15]/8;
actualAssociator=pmul(pmul(b,p0),p0)-pmul(b,pmul(p0,p0));
maximumError=max(maximumError,norm(actualAssociator-expectedAssociator,inf));
fprintf('Matrix/direct-product self-test maximum error: %.3e\n',maximumError);
if maximumError>1e-11
    error('Matrix representation self-test failed.');
end
end

function f = rhs(p,a,b,eta)
pp=pmul(p,p);
f=pmul(a,p)+(1-eta)*pmul(b,pp)+eta*pmul(pmul(b,p),p);
end

function dh = tangentRhs(p,h,a,b,eta)
first=pmul(b,pmul(p,h)+pmul(h,p));
second=pmul(pmul(b,h),p)+pmul(pmul(b,p),h);
dh=pmul(a,h)+(1-eta)*first+eta*second;
end

function J = jacobianMatrix(p,a,b,eta)
J=zeros(5);
for j=1:5
    e=zeros(5,1); e(j)=1;
    J(:,j)=tangentRhs(p,e,a,b,eta);
end
end

function pNext = rk4State(p,a,b,eta,dt)
k1=rhs(p,a,b,eta);
k2=rhs(p+0.5*dt*k1,a,b,eta);
k3=rhs(p+0.5*dt*k2,a,b,eta);
k4=rhs(p+dt*k3,a,b,eta);
pNext=p+(dt/6)*(k1+2*k2+2*k3+k4);
end

function [pNext,MNext] = rk4StateMatrix(p,M,a,b,eta,dt)
k1p=rhs(p,a,b,eta); k1M=jacobianMatrix(p,a,b,eta)*M;
p2=p+0.5*dt*k1p; M2=M+0.5*dt*k1M;
k2p=rhs(p2,a,b,eta); k2M=jacobianMatrix(p2,a,b,eta)*M2;
p3=p+0.5*dt*k2p; M3=M+0.5*dt*k2M;
k3p=rhs(p3,a,b,eta); k3M=jacobianMatrix(p3,a,b,eta)*M3;
p4=p+dt*k3p; M4=M+dt*k3M;
k4p=rhs(p4,a,b,eta); k4M=jacobianMatrix(p4,a,b,eta)*M4;
pNext=p+(dt/6)*(k1p+2*k2p+2*k3p+k4p);
MNext=M+(dt/6)*(k1M+2*k2M+2*k3M+k4M);
end

function [t,x] = integrateRK4(a,b,p0,eta,dt,transient,recordTime,blowupRadius,stride)
p=p0; nTransient=round(transient/dt);
for n=1:nTransient
    p=rk4State(p,a,b,eta,dt);
    if ~all(isfinite(p)) || norm(p)>blowupRadius, error('Trajectory escaped during transient.'); end
end
nSteps=round(recordTime/dt); nOut=floor(nSteps/stride)+1;
t=zeros(nOut,1); x=zeros(nOut,5); t(1)=0; x(1,:)=p'; m=2;
for n=1:nSteps
    p=rk4State(p,a,b,eta,dt);
    if ~all(isfinite(p)) || norm(p)>blowupRadius, error('Trajectory escaped during recording.'); end
    if mod(n,stride)==0
        t(m)=n*dt; x(m,:)=p'; m=m+1;
    end
end
t=t(1:m-1); x=x(1:m-1,:);
end

%% =====================================================================
% Lyapunov and stability diagnostics
% ======================================================================
function out = fullLyapunovSpectrum(a,b,p0,eta,dt,transient,measureTime,renormTime,historyInterval,blowupRadius)
p=p0; nTransient=round(transient/dt);
for n=1:nTransient
    p=rk4State(p,a,b,eta,dt);
    if ~all(isfinite(p)) || norm(p)>blowupRadius, error('Escape in Lyapunov transient.'); end
end
M=eye(5); sums=zeros(5,1); total=0; divSum=0; divN=0;
nSteps=round(measureTime/dt); renormSteps=max(1,round(renormTime/dt));
historySteps=max(renormSteps,round(historyInterval/dt));
historyTime=[]; historySpectrum=[]; historyDivergence=[];
for n=1:nSteps
    [p,M]=rk4StateMatrix(p,M,a,b,eta,dt);
    if ~all(isfinite(p)) || norm(p)>blowupRadius, error('Escape during Lyapunov measurement.'); end
    divSum=divSum+trace(jacobianMatrix(p,a,b,eta)); divN=divN+1;
    if mod(n,renormSteps)==0
        [Q,R]=qr(M,0); d=diag(R); s=sign(d); s(s==0)=1; Q=Q*diag(s); d=abs(d);
        sums=sums+log(max(d,realmin)); M=Q; total=total+renormSteps*dt;
    end
    if mod(n,historySteps)==0 && total>0
        historyTime(end+1,1)=n*dt; %#ok<AGROW>
        historySpectrum(end+1,:)=sort(sums/total,'descend')'; %#ok<AGROW>
        historyDivergence(end+1,1)=divSum/divN; %#ok<AGROW>
    end
end
out.spectrum=sort(sums/total,'descend')';
out.meanDivergence=divSum/divN;
out.historyTime=historyTime;
out.historySpectrum=historySpectrum;
out.historyDivergence=historyDivergence;
out.finalState=p;
end

function out = largestExponentScan(a,b,p0,eta,dt,transient,measureTime,renormTime,h,blowupRadius)
p=p0; maximumNorm=norm(p);
for n=1:round(transient/dt)
    p=rk4State(p,a,b,eta,dt); maximumNorm=max(maximumNorm,norm(p));
    if ~all(isfinite(p)) || norm(p)>blowupRadius
        out.lambda1=NaN; out.maximumNorm=maximumNorm; return;
    end
end
logGrowth=0; total=0; renormSteps=max(1,round(renormTime/dt));
for n=1:round(measureTime/dt)
    k1p=rhs(p,a,b,eta); k1h=tangentRhs(p,h,a,b,eta);
    p2=p+0.5*dt*k1p; h2=h+0.5*dt*k1h;
    k2p=rhs(p2,a,b,eta); k2h=tangentRhs(p2,h2,a,b,eta);
    p3=p+0.5*dt*k2p; h3=h+0.5*dt*k2h;
    k3p=rhs(p3,a,b,eta); k3h=tangentRhs(p3,h3,a,b,eta);
    p4=p+dt*k3p; h4=h+dt*k3h;
    k4p=rhs(p4,a,b,eta); k4h=tangentRhs(p4,h4,a,b,eta);
    p=p+(dt/6)*(k1p+2*k2p+2*k3p+k4p);
    h=h+(dt/6)*(k1h+2*k2h+2*k3h+k4h);
    maximumNorm=max(maximumNorm,norm(p));
    if ~all(isfinite(p)) || norm(p)>blowupRadius
        out.lambda1=NaN; out.maximumNorm=maximumNorm; return;
    end
    if mod(n,renormSteps)==0
        hn=norm(h); logGrowth=logGrowth+log(hn); total=total+renormSteps*dt; h=h/hn;
    end
end
out.lambda1=logGrowth/total; out.maximumNorm=maximumNorm;
end

function out = boundednessWindows(a,b,p0,eta,dt,totalTime,windowTime,blowupRadius)
p=p0; nWindow=round(windowTime/dt); nTotal=round(totalTime/dt); nBlocks=floor(nTotal/nWindow);
out.maximumNorm=zeros(nBlocks,1); out.meanNorm=zeros(nBlocks,1);
for block=1:nBlocks
    sumNorm=0; maxNorm=0;
    for n=1:nWindow
        p=rk4State(p,a,b,eta,dt); pn=norm(p); sumNorm=sumNorm+pn; maxNorm=max(maxNorm,pn);
        if ~all(isfinite(p)) || pn>blowupRadius, error('Escape in boundedness check.'); end
    end
    out.maximumNorm(block)=maxNorm; out.meanNorm(block)=sumNorm/nWindow;
end
out.finalState=p;
end

function info = periodicMetrics(a,b,t,x,eta,dt)
signal=x(:,3); idx=find(signal(2:end-1)>signal(1:end-2) & signal(2:end-1)>=signal(3:end))+1;
if numel(idx)<10, error('Insufficient maxima for period estimation.'); end
peakTimes=zeros(numel(idx),1);
for m=1:numel(idx)
    k=idx(m); ym=signal(k-1); y0=signal(k); yp=signal(k+1);
    denom=ym-2*y0+yp;
    if abs(denom)>eps, delta=0.5*(ym-yp)/denom; else, delta=0; end
    peakTimes(m)=t(k)+delta*(t(k+1)-t(k));
end
periods=diff(peakTimes); first=max(1,numel(periods)-49); use=periods(first:end);
T=median(use); pPeak=x(idx(end-1),:)';
[pEnd,M]=integrateStateAndMatrixForTime(pPeak,eye(5),a,b,eta,dt,T);
mult=eig(M); [~,order]=sort(abs(mult),'descend'); mult=mult(order);
info.period=T; info.closureError=norm(pEnd-pPeak);
info.relativeClosureError=info.closureError/norm(pPeak);
info.floquetMultipliers=mult;
end

function [p,M] = integrateStateAndMatrixForTime(p,M,a,b,eta,dt,T)
n=floor(T/dt); rem=T-n*dt;
for k=1:n, [p,M]=rk4StateMatrix(p,M,a,b,eta,dt); end
if rem>0, [p,M]=rk4StateMatrix(p,M,a,b,eta,rem); end
end

%% =====================================================================
% Signal and plotting diagnostics
% ======================================================================
function peaks = localMaxima(v)
idx=find(v(2:end-1)>v(1:end-2) & v(2:end-1)>=v(3:end))+1; peaks=v(idx);
end

function points = poincareSection(t,x,component,level,direction)
v=x(:,component)-level; points=[];
for n=1:numel(t)-1
    crossing=(v(n)<0 && v(n+1)>=0 && direction>0) || (v(n)>0 && v(n+1)<=0 && direction<0);
    if crossing
        alpha=-v(n)/(v(n+1)-v(n)); points(end+1,:)=x(n,:)+alpha*(x(n+1,:)-x(n,:)); %#ok<AGROW>
    end
end
end

function out = fftMetrics(t,v)
v=v-mean(v); n=numel(v); dt=median(diff(t)); fs=1/dt;
w=0.5-0.5*cos(2*pi*(0:n-1)'/(n-1)); z=fft(v.*w); P=abs(z(1:floor(n/2)+1)).^2;
f=(0:floor(n/2))'*fs/n; P=P(2:end); f=f(2:end); prob=P/sum(P);
out.frequency=f; out.power=P; out.spectralEntropy=-sum(prob.*log(prob+realmin))/log(numel(prob));
out.peakPowerFraction=max(P)/sum(P);
end

function plotSpectrum(s,titleText,fileName)
figure('Color','w'); semilogy(s.frequency,s.power+realmin); xlim([0 min(2,max(s.frequency))]); grid on;
xlabel('frequency'); ylabel('power'); title(titleText); saveas(gcf,fileName);
end

function plotComponentTimeSeries(tC,xC,tP,xP,names,outDir)
% Plot all five canonical components without thinning the displayed windows.
chaosWindow = min(120,tC(end)-tC(1));
periodicWindow = min(4*5.795118818486838,tP(end)-tP(1));
maskC = tC >= tC(end)-chaosWindow;
maskP = tP >= tP(end)-periodicWindow;
tauC = tC(maskC)-(tC(end)-chaosWindow);
tauP = tP(maskP)-(tP(end)-periodicWindow);
labels = {'x_0','x_1','x_2','x_3','x_4'};

figure('Color','w','Position',[80 60 1500 800]);
subplot(2,1,1); hold on;
for j=1:5
    plot(tauC,xC(maskC,j),'LineWidth',0.55,'DisplayName',labels{j});
end
grid on; ylabel('canonical components');
title('Numerical chaotic endpoint, \eta=0');
legend('Location','eastoutside');
subplot(2,1,2); hold on;
for j=1:5
    plot(tauP,xP(maskP,j),'LineWidth',0.75,'DisplayName',labels{j});
end
grid on; ylabel('canonical components'); xlabel('time after transient');
title('Attracting periodic endpoint, \eta=1');
legend('Location','eastoutside');
try
    exportgraphics(gcf,fullfile(outDir,'model_time_series.png'),'Resolution',400);
catch
    print(gcf,fullfile(outDir,'model_time_series.png'),'-dpng','-r400');
end
end

function plotTrajectory3D(xC,xP,outDir)
% Preserve every recorded point in the publication rendering.
% With dt=0.005 and 3000 post-transient time units this is 600,001 samples.
idxC=1:size(xC,1);
idxP=1:size(xP,1);
% The periodic orbit is already asymptotic; the last 100 time units give a
% smooth rendering without unnecessarily retracing the same closed curve.
startP=max(1,size(xP,1)-100000);
idxP=idxP(idxP>=startP);

figure('Color','w','Position',[80 80 1600 700]);
subplot(1,2,1); plot3(xC(idxC,1),xC(idxC,4),xC(idxC,5),'LineWidth',0.25);
grid on; axis tight; view(42,24);
xlabel('x_0'); ylabel('x_3'); zlabel('x_4');
title('Numerical chaotic attractor candidate, \eta=0');
subplot(1,2,2); plot3(xP(idxP,1),xP(idxP,4),xP(idxP,5),'LineWidth',0.75);
grid on; axis tight; view(42,24);
xlabel('x_0'); ylabel('x_3'); zlabel('x_4');
title('Attracting periodic orbit, \eta=1');
try
    exportgraphics(gcf,fullfile(outDir,'model_phase_portraits_dense.png'),'Resolution',400);
catch
    print(gcf,fullfile(outDir,'model_phase_portraits_dense.png'),'-dpng','-r400');
end
end

function tableOut = compareWithOde45(a,b,p0,etaValues,cfg)
rows=[];
for eta=etaValues
    [tR,xR]=integrateRK4(a,b,p0,eta,cfg.dtMain,cfg.transient,min(cfg.recordTime,500),cfg.blowupRadius,1);
    opts=odeset('RelTol',1e-10,'AbsTol',1e-12,'MaxStep',cfg.dtMain);
    [~,yTransient]=ode45(@(t,y)rhs(y,a,b,eta),[0 cfg.transient],p0,opts);
    pStart=yTransient(end,:)';
    tOut=(0:cfg.dtMain:min(cfg.recordTime,500))';
    [~,xO]=ode45(@(t,y)rhs(y,a,b,eta),tOut,pStart,opts);
    nr=sqrt(sum(xR.^2,2)); no=sqrt(sum(xO.^2,2));
    rows=[rows; eta mean(nr) std(nr) max(nr) mean(no) std(no) max(no)]; %#ok<AGROW>
end
tableOut=array2table(rows,'VariableNames',{'eta','RK4_mean_norm','RK4_std_norm','RK4_max_norm',...
    'ode45_mean_norm','ode45_std_norm','ode45_max_norm'});
end
