function pentarnion_holonomy_demo()
% Exact four-step switching holonomy.
gamma1=0; gamma2=1; epsilon=deg2rad(45);
H=pentarnion_rotor_matrix(gamma2,'c',-epsilon) * ...
  pentarnion_rotor_matrix(gamma1,'c',-epsilon) * ...
  pentarnion_rotor_matrix(gamma2,'c', epsilon) * ...
  pentarnion_rotor_matrix(gamma1,'c', epsilon);
beta1=atan2(gamma1,sqrt(3)); beta2=atan2(gamma2,sqrt(3));
Phi=4*asin(abs(sin(beta2-beta1))*sin(epsilon/2)^2);
fprintf('Residual angle = %.12f degrees\n',rad2deg(Phi));
fprintf('Orthogonality error = %.3e\n',norm(H.'*H-eye(4),'fro'));
fprintf('det(H) = %.16f\n',det(H));
end
