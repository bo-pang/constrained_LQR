clear;
A = [0.8 1;0 0.9];
B = [0.5;1];
[n,m] = size(B);
% prediction horizon
t = 2;    
% order of persistent excitation
L = n + t;   
% lower limit of 
T_lim = (m+1)*L-1;
% Number of data points to be collected
T = T_lim;
% Check the persistent excitation of inputs
x0d = [0.1,0.2];
xd = zeros(n,T+1);
ud = zeros(m,T);
xd(:,1) = x0d;
rng('default')
rng(1);
ww = -10 + 20*rand(m,2);
for i=1:T
    e = 0.1*sum(sin(ww*i));
%     ud(:,i) = -Kd*xd(:,i) + e ;%randn(m,1);
    ud(:,i) = e ;
    xd(:,i+1) = A*xd(:,i) +B*ud(:,i);
end
xd = xd(:,1:end-1);
xh = dHankel(xd,t);
uh = dHankel(ud,t);
uh_pe = dHankel(ud,L);
uh_rank = rank(uh_pe);
if uh_rank<m*L
    disp('Rank Deficient!');
    return;
end
figure(3);
subplot(2,1,1);
plot(1:T,xd);hold on;
subplot(2,1,2);
plot(1:T,ud);hold on;

% Cost matrix
Q = eye(n);
R = eye(m);
[P,~,K] = dare(A,B,Q,R);

% Check ADP rank
theta = [];
phi = [];
P_est = {Q};
K_est = {};
I = 5000;
P_opt_error = norm(P_est{1}-P);
K_opt_error = [];
for j=1:I
    for i=1:T-1
        theta(end+1,:) = [kronv(xd(:,i));2*kron(xd(:,i),ud(:,i));kronv(ud(:,i))]';
        phi(end+1,1) = xd(:,i+1)'*P_est{end}*xd(:,i+1);
    end
    theta_rank = rank(theta);
    adp_APA_rank = n*(n+1)/2;
    adp_BPA_rank = m*n;
    adp_BPB_rank = m*(m+1)/2;
    adp_rank = adp_APA_rank + adp_BPA_rank + adp_BPB_rank;
    if theta_rank < adp_rank
        disp('ADP Rank Deficient!');
    end
    params = theta\phi;
    APA_est = vec2sm(params(1:adp_APA_rank),n);
    BPA_est = reshape(params(adp_APA_rank+1:adp_APA_rank+adp_BPA_rank),[m,n]);
    BPB_est = vec2sm(params(adp_APA_rank+adp_BPA_rank+1:end),m);
    P_est{end+1} = APA_est - BPA_est'*inv(R + BPB_est)*BPA_est + Q;
    K_est{end+1} = inv(R + BPB_est)*BPA_est;
    P_opt_error(end+1) = norm(P_est{end}-P);
    K_opt_error(end+1) = norm(K_est{end}-K);
    % Check if the controller is stabilizing
    if any(eig(P_est{end-1})>0)
        Test_Stab = APA_est - K_est{end}'*BPA_est - BPA_est'*K_est{end} +...
            K_est{end}'*BPB_est*K_est{end} - P_est{end-1};
        if any(eig(Test_Stab)<0)
%             disp('Lyapunov Function Found');
            if ~any(abs(eig(A-B*K_est{end}))<1)
                disp('Contradition');
            end
        end
    end
end
P_hat = P_est{end};
K_hat = K_est{end};
figure(1);
NVI_plot = 1:200;
subplot(2,1,1);
stairs(NVI_plot-1,P_opt_error(NVI_plot));
xlabel('Iteration');
ylabel('$\Vert\hat{K}_i-K\Vert$','Interpreter','Latex');
subplot(2,1,2);
stairs(NVI_plot-1,K_opt_error(NVI_plot));
xlabel('Iteration');
ylabel('$\Vert\hat{P}_i-P\Vert$','Interpreter','Latex');

% Evaluate Khat
Gamma = [];
Pai = [];
for i=1:T-1
    Gamma = [Gamma;kronv(xd(:,i))'];
    xi = [kronv(xd(:,i))'-kronv(xd(:,i+1))',...
        2*kron(xd(:,i),ud(:,i)+K_hat*xd(:,i))',...
        kron(ud(:,i)-K_hat*xd(:,i),ud(:,i)+K_hat*xd(:,i))'];
    Pai = [Pai;xi];
end
rank_Pai = rank(Pai);
[~,num_col_Pai] = size(Pai);
if rank_Pai<num_col_Pai
    disp('Rank Deficient in Policy Evaluation');
end
params = Pai\(Gamma*sm2vec(Q+K_hat'*R*K_hat));
P_Khat_est = vec2sm(params(1:adp_APA_rank),n);
P_Khat = dlyap((A-B*K_hat)',Q+K_hat'*R*K_hat);

um = 1;
xm = 5;
% compute c such that x'Px<c is admissible
Ac = [eye(n);
    -eye(n);
    -K_hat;
    K_hat;];
Bc = [xm*ones(2*n,1);
    um*ones(2*m,1);];
[nbar,~] = size(Ac);
ctmp = zeros(nbar,1);
for i=1:nbar
    ctmp(i) = Bc(i)^2/(Ac(i,:)*inv(P_Khat_est)*Ac(i,:)');
end
c = min(ctmp);

% MPC horizon
N = t;
F = P_Khat_est;   % the final penalty matrix
while 1
    cvx_begin;
    variables g(T-t+1,N-t+1) u(m,N) x(n,N);% x(n,N) J;
    x0 = [-0.5;-3];    % Most different behavior
%     x0 = [1;-1];
%     x0 = [-4;3];
%     x0 = [3;2];
%     x0 = [-0.5;-2];
    AA = [uh;xd(:,1:T-t+1)];
    xtmp = x0;
    for i=1:N-t+1
        [vec(u(:,i:i+t-1));xtmp] == AA*g(:,i);
        x(:,i:i+t-1) == reshape(xh*g(:,i),[n,t]);
        xtmp = x(:,i+t-1);
    end
    J = 0;
    for i=1:N-1
        J = J + x(:,i)'*Q*x(:,i) + u(:,i)'*R*u(:,i); 
        norm(x(:,i))<=xm;
        norm(u(:,i))<=um;
    end
    norm(x(:,end))<=xm;
    J = J + x(:,end)'*F*x(:,end);
    minimize J;
    cvx_end;
    if strcmp(cvx_status,'Infeasible')==1
        disp('Infeasible Initial Condition');
        N
        return;
    end
    if x(:,end)'*P_Khat_est*x(:,end)<c
        break;
    end
    N = N+1;
end
u = u(:,1:N-1);
if x(:,end)'*P_Khat_est*x(:,end)>c
    disp('The MPC horizon is not sufficiently long!');
    return;
end
% find the coefficient
% up = rand(m,t);
% g = linsolve(AA,BB); 
% do prediction
% xp = reshape(xp_vec,[n,t]);

% % true trajectory
% xp_true = zeros(n,t+1);
% xp_true(:,1) = x0p;
% for i=1:t
%     xp_true(:,i+1) = A*xp_true(:,i) + B*up(:,i);
% end
% xp_true = xp_true(:,1:end-1);
% % norm(xp-xp_true)
Nsim = 4*N;
x_uc = zeros(n,Nsim);
u_uc = zeros(m,Nsim-1);
x_c = zeros(n,Nsim);
u_c = zeros(m,Nsim-1);
u_c(:,1:N-1) = u;
x_uc(:,1) = x0;
x_c(:,1) = x0;
for i=1:Nsim-1
    u_uc(:,i) = -K_hat*x_uc(:,i);
    x_uc(:,i+1) = A*x_uc(:,i) + B*u_uc(:,i);
    if i<N
        x_c(:,i+1) = A*x_c(:,i) + B*u_c(:,i);
    else
        u_c(:,i) = -K_hat*x_c(:,i);
        x_c(:,i+1) = A*x_c(:,i) + B*u_c(:,i);
    end
end
figure(2);
subplot(9,2,[1 3 5 7]);
stairs(0:Nsim-1,x_uc(1,:));hold on;
stairs(0:Nsim-1,x_c(1,:),'--');hold on;
plot((1:Nsim)-1,xm*ones(Nsim),'r:');hold on;
plot((1:Nsim)-1,-xm*ones(Nsim),'r:');hold on;
xlabel('k');
ylabel('x_1','rotation',0);
ylim([-6,6]);
xlim([0 Nsim-1]);
% legend('Unconstrained LQR','Constrained LQR');
% figure(2);
subplot(9,2,[11 13 15 17]);
stairs(0:Nsim-1,x_uc(2,:));hold on;
stairs(0:Nsim-1,x_c(2,:),'--');hold on;
plot((1:Nsim)-1,xm*ones(Nsim),'r:');hold on;
plot((1:Nsim)-1,-xm*ones(Nsim),'r:');hold on;
xlabel('k');
ylabel('x_2','rotation',0);
ylim([-6,6]);
xlim([0 Nsim-1]);
% legend('Unconstrained LQR','Constrained LQR');
subplot(10,2,[6 8 10 12 14]);
stairs((1:Nsim-1)-1,u_uc(1,:));hold on;
stairs((1:Nsim-1)-1,u_c(1,:),'--');hold on;
plot((1:Nsim-1)-1,um*ones(Nsim-1),'r:');hold on;
plot((1:Nsim-1)-1,-um*ones(Nsim-1),'r:');hold on;
xlabel('k');
ylabel('u','rotation',0);
xlim([0 Nsim-2]);
ylim([-2 4]);
% legend('Unconstrained LQR','Constrained LQR');
% ylim([-1.5,1.5]);
N 
c
I
J_uc = x0'*P*x0
P_Khat_est
P
K_hat
K
ww
T

function uh = dHankel(U,L)
[m,T] = size(U);
uh = zeros(m*L,T-L+1);
for i=1:T-L+1
    uh(:,i) = vec(U(:,i:L+i-1));
end
end

function X = kronv(x)
len = length(x);
X = [];
for i=1:len
    for j=i:len
        if i==j
            X(end+1) = x(i)*x(j);
        else
            X(end+1) = sqrt(2)*x(i)*x(j);
        end
    end
end
X = X';
end

function x = sm2vec(X)
[n,~] = size(X);
N = n*(n+1)/2;
x = zeros(N,1);
k = 1;
for i=1:n
    for j=i:n
        if i==j
            x(k) = X(i,j);
        else
            x(k) = sqrt(2)*X(i,j);
        end
        k = k+1;
    end
end
end

function X = vec2sm(x,n)
X = zeros(n);
num = flip(1:n);
for i=1:n
    index = 0;
    for k=1:i-1
        index = index+num(k);
    end
    for j=0:n-i
        if j~=0
            X(i,i+j)=x(index+j+1)/sqrt(2);
            X(j+i,i)=X(i,j+i);
        else
            X(i,j+i)=x(index+j+1);
        end
    end
end
end