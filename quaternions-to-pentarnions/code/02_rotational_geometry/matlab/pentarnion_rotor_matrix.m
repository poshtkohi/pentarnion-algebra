function R = pentarnion_rotor_matrix(gamma,axisName,theta,basisName)
% Exact rotational-shadow matrix for the switching Pentarnion geometry.
% MATLAB R2020a compatible.
if nargin < 4, basisName='adapted'; end
if gamma < 0 || gamma > 1, error('gamma must be in [0,1]'); end
C=sqrt(3)/sqrt(3+gamma^2);
S=gamma/sqrt(3+gamma^2);
P=[C 0 0 -S; 0 1 0 0; 0 0 1 0; S 0 0 C];
switch lower(axisName)
    case 'h', u=[1;0;0];
    case 'b', u=[0;1;0];
    case 'c', u=[0;0;1];
    otherwise, error('axisName must be h, b, or c');
end
K=[0 -u(3) u(2); u(3) 0 -u(1); -u(2) u(1) 0];
R3=cos(theta)*eye(3)+(1-cos(theta))*(u*u.')+sin(theta)*K;
B=eye(4); B(1:3,1:3)=R3;
R=P*B*P.';
if strcmpi(basisName,'original')
    U=[1/sqrt(3),1/sqrt(2),1/sqrt(6),0; ...
       1/sqrt(3),-1/sqrt(2),1/sqrt(6),0; ...
       1/sqrt(3),0,-2/sqrt(6),0; ...
       0,0,0,1];
    R=U*R*U.';
elseif ~strcmpi(basisName,'adapted')
    error('basisName must be adapted or original');
end
end
