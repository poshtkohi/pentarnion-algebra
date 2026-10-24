%% associator_235_deformation.m
% Exact associator-compensated 2:3:5 orbit and its destruction when eta<1.
% Authors: Alireza Poshtkohi and Susanne Pumpluen
%
% The family is
%   p' = F_1(p) + (5/2)(1-eta)[ell,r,r],
% where q=w+xi, r=yj+zk+u ell and F_1 is the compensated flow.
% The script evaluates the exact eta=1 orbit at 200001 points and integrates
% eta=0.9 from the same initial state at the same output resolution.

clear; clc; close all;

N = 200001;
t = linspace(0,2*pi,N).';
eta = 0.9;

%% Exact compensated orbit (eta=1)
D2 = 5+4*cos(2*t);
D5 = 2+cos(5*t);
Pexact = [(1+2*cos(2*t))./D2, ...
          2*sin(2*t)./D2, ...
          sqrt(3)*cos(3*t)./D5, ...
          sqrt(3)*sin(3*t)./D5, ...
          sin(5*t)./D5];

w=Pexact(:,1); x=Pexact(:,2); y=Pexact(:,3); z=Pexact(:,4); u=Pexact(:,5);
Fexact = [-2*x+4*w.*x, ...
          2*w-2*w.^2+2*x.^2, ...
          -3*z+5*u.*y, ...
          3*y+5*u.*z, ...
          (5/2)*(1-y.^2-z.^2+u.^2)];

dD2 = -8*sin(2*t); dD5 = -5*sin(5*t);
N0=1+2*cos(2*t); N1=2*sin(2*t);
N2=sqrt(3)*cos(3*t); N3=sqrt(3)*sin(3*t); N4=sin(5*t);
dN0=-4*sin(2*t); dN1=4*cos(2*t);
dN2=-3*sqrt(3)*sin(3*t); dN3=3*sqrt(3)*cos(3*t); dN4=5*cos(5*t);
dPexact = [(dN0.*D2-N0.*dD2)./D2.^2, ...
           (dN1.*D2-N1.*dD2)./D2.^2, ...
           (dN2.*D5-N2.*dD5)./D5.^2, ...
           (dN3.*D5-N3.*dD5)./D5.^2, ...
           (dN4.*D5-N4.*dD5)./D5.^2];
exactResidual = max(abs(Fexact-dPexact),[],'all');

%% Uncompensated associator deformation from the same initial point
p0 = [1/3;0;1/sqrt(3);0;0];
opts = odeset('RelTol',1e-11,'AbsTol',1e-13,'MaxStep',0.01);
[td,Pdef] = ode113(@(tau,p) rhsAssociator235(tau,p,eta),t,p0,opts); %#ok<ASGLU>

deviation = vecnorm(Pdef-Pexact,2,2);
returnDefect = norm(Pdef(end,:).'-p0);

% The exact initial associator is (1/6)(-i+k+2 ell).
initialAssociator = [0;-1/6;0;1/6;1/3];
initialTangentDefect = rhsAssociator235(0,p0,eta)-rhsAssociator235(0,p0,1);
expectedTangentDefect = (5/2)*(1-eta)*initialAssociator;
tangentIdentityError = norm(initialTangentDefect-expectedTangentDefect,inf);

fprintf('Compensated period T = %.15g\n',2*pi);
fprintf('Samples per trajectory = %d\n',N);
fprintf('Maximum exact eta=1 residual = %.3e\n',exactResidual);
fprintf('Initial tangent-identity error = %.3e\n',tangentIdentityError);
fprintf('One-period return defect at eta=%.1f = %.12f\n',eta,returnDefect);
fprintf('Maximum deviation from exact orbit = %.12f\n',max(deviation));

%% High-resolution comparison figure
figure('Position',[80 80 1250 850]);
subplot(2,2,1);
plot3(Pexact(:,1),Pexact(:,3),Pexact(:,5),'LineWidth',0.8); grid on;
xlabel('w'); ylabel('y'); zlabel('u');
title('Compensated exact orbit, \eta=1');

subplot(2,2,2); hold on;
plot(t,Pexact,'LineWidth',0.7); grid on;
xlabel('t'); ylabel('canonical component');
legend('w','x (i)','y (j)','z (k)','u (ell)','Location','best');
title('Exact 2:3:5 component dynamics');

subplot(2,2,3);
plot3(Pdef(:,1),Pdef(:,3),Pdef(:,5),'LineWidth',0.8); grid on; hold on;
plot3(Pdef(1,1),Pdef(1,3),Pdef(1,5),'o');
plot3(Pdef(end,1),Pdef(end,3),Pdef(end,5),'x');
xlabel('w'); ylabel('y'); zlabel('u');
title(sprintf('Same initial point, \\eta=%.1f',eta));

subplot(2,2,4);
plot(t,deviation,'LineWidth',0.9); grid on;
xlabel('t'); ylabel('||p_{\eta}(t)-p_*(t)||_2');
title(sprintf('Return defect at 2\\pi: %.3f',returnDefect));

%% Local function
function f = rhsAssociator235(~,p,eta)
    w=p(1); x=p(2); y=p(3); z=p(4); u=p(5);

    % Compensated eta=1 component field.
    f1 = [-2*x+4*w*x; ...
          2*w-2*w^2+2*x^2; ...
          -3*z+5*u*y; ...
          3*y+5*u*z; ...
          (5/2)*(1-y^2-z^2+u^2)];

    % [ell,r,r] for r=yj+zk+u ell in canonical coordinates.
    a = [-sqrt(2)*y*z; ...
         -0.5*(y+z)*(u+y-z); ...
         -0.5*(4*u*y+u*z+y*z+z^2); ...
          0.5*(-u*y-4*u*z+y^2+y*z); ...
          y^2+z^2];

    f = f1 + (5/2)*(1-eta)*a;
end
