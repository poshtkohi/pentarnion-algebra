function pentarnion_mechanical_spin_demo()
% Algebra-native mechanical spin: regular and chaotic representative runs.
y0=[0.45;0.65;-0.61]; y0=y0/norm(y0);
Omegas=[0.30,1.20]; names={'regular','chaotic'};
opts=odeset('RelTol',1e-10,'AbsTol',1e-12,'MaxStep',0.02);
outDir=fullfile(fileparts(mfilename('fullpath')),'outputs');
if ~exist(outDir,'dir'), mkdir(outDir); end
for k=1:numel(Omegas)
    Om=Omegas(k);
    [t,y]=ode113(@(t,y) rhs(t,y,Om),[0,300],y0,opts);
    H=zeros(size(t)); gamma=zeros(size(t));
    for n=1:numel(t)
        gamma(n)=0.5*(1+sin(Om*t(n)));
        H(n)=hamiltonian(y(n,:).',gamma(n));
    end
    fprintf('%s: norm error %.3e, H=[%.8f,%.8f]\n',names{k}, ...
        max(abs(vecnorm(y,2,2)-1)),min(H),max(H));
    fig=figure('Color','w');
    plot(t,y(:,1),t,y(:,2),t,y(:,3),t,H);
    xlabel('time'); ylabel('components / energy'); grid on;
    legend('y_h','y_b','y_c','H');
    title(sprintf('%s regime, Omega=%.2f',names{k},Om));
    print(fig,fullfile(outDir,[names{k} '_timeseries.png']),'-dpng','-r180');
    close(fig);
end
end

function dy=rhs(t,y,Omega)
gamma=0.5*(1+sin(Omega*t));
[A,B]=inertia(gamma);
g=[1/sqrt(3+gamma^2);1/sqrt(2);1/sqrt(6)];
grad=[A*y(1);B*y(2);B*y(3)]-g;
dy=cross(y,grad);
end

function H=hamiltonian(y,gamma)
[A,B]=inertia(gamma);
g=[1/sqrt(3+gamma^2);1/sqrt(2);1/sqrt(6)];
H=0.5*(A*y(1)^2+B*(y(2)^2+y(3)^2))-g.'*y;
end

function [A,B]=inertia(gamma)
g2=gamma^2;
A=3*(g2^2+2*g2+3)/((1+g2)*(3+g2));
B=(4*g2+3)/(1+g2);
end
