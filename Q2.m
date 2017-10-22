
%Assuming I have X
[N,D] = size(X);% X include bias term
X = double(X);
H1 = 128 ;% First Hidden layer
H2 = 64 ; %second Hidden Layer
K = 1 ;% number of output
X(:,1) = 0 ;
w = -0.01 + (0.02)*rand(H1,D); % between input and first hidden layer
v = -0.01+(0.02)*rand(H2,(H1+1)); % first hidden layer and second hidden layer
u = -0.01+(0.02)*rand(K,(H2+1)); % second hidden layer and output

nEpochs = 1 ;
iporder = randperm(N);
Z1 = zeros(H1+1,1);
T1 = zeros(H2+1,1);
eta = 0.01 ;
for epoch = 1:nEpochs
    epoch
    for n = 1:1000
        n
        % the current training point is X(iporder(n), :)
        % forward pass2
        example = X(iporder(n),:); % 1 example of all instance
        
        a = w*(transpose(example));
        a = abs(a);
        Z = sigmf(a,[1 0]);    % applying sigmoid function
        [m1 n1] = size(Z); % adding bias term
        Z1(2:m1+1,:) = Z(1:m1,:);
        b = v*(Z1);
        b = abs(b);
        T = sigmf(b,[1 0]);
        [m2 n2] = size(T); 
        T1(2:m2+1,:) = T(1:m2,:);
        O = u*T1 ;% there is no activation function in output
        y = Y(iporder(n),:);
                % backward pass
        % We will use square error here
        deltaU = eta*(O-y)*transpose(T1);
        
        deltaV = zeros(size(v));
        
        for j = 1:H1+1
            for h = 2:H2+1
                deltaV(h-1,j) = eta*(O-y)*T1(h,1)*(1-T1(h,1))*Z1(j,1); ;
            end
        end
        deltaW = zeros(size(w));
        
        for i = 1:D
            for j = 2:H1+1
                sum = 0 ;
                for h = 2:H2+1
                    sum  = sum + T1(h,1)*(1-T1(h,1));
                end
                deltaW(j-1,i) = eta*(O-y)*sum*Z1(j,1)*(1-Z1(j,1))*example(1,i) ; 
            end
        end
        w = w - deltaW ;
        v = v - deltaV ;
        u = u - deltaU ;
    end
    
end






