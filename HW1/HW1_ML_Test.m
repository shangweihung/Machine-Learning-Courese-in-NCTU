clear all
clc


xy_axis=csvread('X_train.csv');
target=csvread('T_train.csv');
test_axis=csvread('X_test.csv');

%% Parameters setting  
stride=13;
local_size=25;
map_size=1081;


%% K-fold Cross-Validation   
for K=1:1:1

    %% Scanning;
    mu_x=[];
    mu_y=[];
    for y=1:stride:(map_size-local_size)
        for x=1:stride:(map_size-local_size)    
            index=[];
            index=find( xy_axis(1:30000,1)>=x & xy_axis(1:30000,1)<(x+local_size) & xy_axis(1:30000,2)>=y & xy_axis(1:30000,2)<(y+local_size) ); % find return index
            % get local training data
            local_train=[xy_axis(index,:),target(index)];
            
            % calculate local mean and sigma
            if sum(local_train(:,3))==0
                % case when the height of all local training data equal to zero
                % then set the center of the local region as the mu
                mu_x=[mu_x;(x+local_size/2)];
                mu_y=[mu_y;(y+local_size/2)];
            else
                mu_x=[mu_x;dot(local_train(:,1),local_train(:,3))/sum(local_train(:,3))];
                mu_y=[mu_y;dot(local_train(:,2),local_train(:,3))/sum(local_train(:,3))];
            end
            sigma_x=local_size;
            sigma_y=local_size;
        end
    end
    
    %% Create Design Matrix  
    Design=zeros(length(target(1:30000)),length(mu_x));

    for j=1:length(target(1:30000))
        for i=1:length(mu_x)
            Design(j,i)=exp(-(xy_axis(j,1)-mu_x(i))^2/(2*sigma_x^2)-(xy_axis(j,2)-mu_y(i))^2/(2*sigma_y^2));
        end
    end
    Design=[ones(length(target(1:30000)),1),Design];

    %% Optimization  
    W_ML=pinv(Design)*target(1:30000);

    %% Estimation  
    Estimate_Phi=[];

    for j=1:1:length(test_axis)
        buffer=[];
        if mod(j,2000)==0
            disp(j)
        end
        for i=1:length(mu_x)
            buffer=[buffer,exp(-(test_axis(j,1)-mu_x(i))^2/(2*sigma_x^2)-(test_axis(j,2)-mu_y(i))^2/(2*sigma_y^2))];
        end
        Estimate_Phi=[Estimate_Phi;buffer];
    end
    Estimate_Phi=[ones(length(test_axis(1:10000)),1), Estimate_Phi];

    Estimation_ML=W_ML'*Estimate_Phi';
    
    % circular shift the x y coordinates and target for K-fold CV
    % xy_axis=circshift(xy_axis,[10000 0]);
    % target=circshift(target,[10000 0]);
end



