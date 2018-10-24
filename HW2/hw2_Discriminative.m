clear all
close all
clc


%% Load Image from three folders
Data=[];
for i=1:1000
    Img = imread(['Class1/faceTrain1_',num2str(i),'.bmp']);
    image=reshape(Img,[1,900]); 
    Data=[Data;image];
end
for i=1:1000
    Img = imread(['Class2/faceTrain2_',num2str(i),'.bmp']);
    image=reshape(Img,[1,900]); 
    Data=[Data;image];
end
for i=1:1000
    Img = imread(['Class3/faceTrain3_',num2str(i),'.bmp']);
    image=reshape(Img,[1,900]); 
    Data=[Data;image];
end
Data=double(Data);

% new_class1=randperm(1000);
% new_class2=randperm(1000)+1000;
% new_class3=randperm(1000)+2000;
new_class1=1:1000;
new_class2=1001:2000;
new_class3=2001:3000;
new_Data=[Data(new_class1,:);Data(new_class2,:);Data(new_class3,:)];
Train=[new_Data(1:900,:);new_Data(1001:1900,:);new_Data(2001:2900,:)];


%% PCA
[M,N]=size(Train);
total=zeros(1,N);
for i=1:M
    total=total+Train(i,:);
end
mean_vector=total/M;

[row,col]=size(Train);
mean_aug=[];
for i=1:row
    mean_aug=[mean_aug;mean_vector];
end
X_hat=Train-mean_aug; % recenter to zero mean
X_hat=normc(X_hat);
X_hat_H=ctranspose(X_hat);
R=X_hat_H*X_hat;      % covariance matrix

[U,S,V]=svd(R);
%[U,S,V] = svd(A) => A = U*S*V'.
x_pca=[];

mean_aug_test=[];
%% PCA projection
for i=1:size(Data,1)
    mean_aug_test=[mean_aug_test;mean_vector];
end

X_hat_test=new_Data-mean_aug_test;
for i=1:size(Data,1)
    x_pca=[x_pca,U(:,1:2)'*(X_hat_test(i,:)')];
end
display('done');

figure;
plot(x_pca(1,1:900),x_pca(2,1:900),'r*');
hold on
plot(x_pca(1,1001:1900),x_pca(2,1001:1900),'g*');
hold on
plot(x_pca(1,2001:2900),x_pca(2,2001:2900),'b*');


%%
x_pca_train=[x_pca(:,1:900),x_pca(:,1001:1900),x_pca(:,2001:2900)];
x_pca_test=[x_pca(:,901:1000),x_pca(:,1901:2000),x_pca(:,2901:3000)];

% boundary
max_train=round(max(x_pca_train'));
min_train=round(min(x_pca_train'));
x_basis1=linspace(min_train(1),max_train(1),100);
x_basis2=linspace(min_train(2),max_train(2),100);
x_pca_boundary=zeros(2,length(x_basis1)*length(x_basis2));
index=1;
for p=1:length(x_basis2)
    for q=1:length(x_basis1)
        x_pca_boundary(:,index)=[x_basis1(q);x_basis2(p)];
        index=index+1;
    end
end
x_pca_boundary=[x_pca_boundary;ones(1,size(x_pca_boundary,2))];
x_pca_train=[x_pca_train;ones(1,size(x_pca_train,2))];
x_pca_test=[x_pca_test;ones(1,size(x_pca_test,2))];
%bias 1

w1=[0;0;0]; % default
w2=[0;0;0]; 
w3=[0;0;0]; 


flag=0;
iter=0;
% Here phi = x
while flag==0
    I=eye(3);
    H=zeros(3*3);
    First=zeros(9,1);
    
    for i=1:size(x_pca_train,2)
        a1=w1'*x_pca_train(:,i);
        a2=w2'*x_pca_train(:,i);
        a3=w3'*x_pca_train(:,i);

        a_max=max([a1,a2,a3]);
        p_c1_x=exp(a1-a_max)/(exp(a1-a_max)+exp(a2-a_max)+exp(a3-a_max));
        p_c2_x=exp(a2-a_max)/(exp(a1-a_max)+exp(a2-a_max)+exp(a3-a_max));
        p_c3_x=exp(a3-a_max)/(exp(a1-a_max)+exp(a2-a_max)+exp(a3-a_max));
        
        y=[p_c1_x;p_c2_x;p_c3_x];
        %y(find(y==0)) = 1e-300;
        phi=x_pca_train(:,i);
        h=zeros(3*3);
        for k=1:3
            for j=1:3
                h((3*k-2:3*k),(3*j-2:3*j))=y(k)*(I(k,j)-y(j))*(phi*phi');
            end
        end
        H=H+h;
        
        if i>=1 & i<=900
            t=[1;0;0];
        elseif i>=901 & i<=1800
            t=[0;1;0];
        else
            t=[0;0;1];
        end
      
        first_der=zeros(9,1);
        for j=1:3
            first_der((3*j-2:3*j))=(y(j)-t(j))*phi;
        end
        First=First+first_der;

    end
    w_old=[w1;w2;w3];
    w_new=w_old-pinv(H)*First;
    w_new
    if iter==30
        flag=1;
    end
    iter=iter+1;
    
    w1=w_new(1:3);
    w2=w_new(4:6);
    w3=w_new(7:9);
end
%% Testing 
estimation=[];
for i=1:size(x_pca_test,2)
    a1=w1'*x_pca_test(:,i);
    a2=w2'*x_pca_test(:,i);
    a3=w3'*x_pca_test(:,i);

    p_c1_x=exp(a1)/(exp(a1)+exp(a2)+exp(a3));
    p_c2_x=exp(a2)/(exp(a1)+exp(a2)+exp(a3));
    p_c3_x=exp(a3)/(exp(a1)+exp(a2)+exp(a3));
        
        
    [M,I]=max([p_c1_x,p_c2_x,p_c3_x]);
    estimation=[estimation,I];
end
%% Boundary

estimation_boundary=[];
for i=1:length(x_pca_boundary)
    a1=w1'*x_pca_boundary(:,i);
    a2=w2'*x_pca_boundary(:,i);
    a3=w3'*x_pca_boundary(:,i);

    p_c1_x=exp(a1)/(exp(a1)+exp(a2)+exp(a3));
    p_c2_x=exp(a2)/(exp(a1)+exp(a2)+exp(a3));
    p_c3_x=exp(a3)/(exp(a1)+exp(a2)+exp(a3));

    if(abs(p_c1_x-p_c2_x)<0.12 & p_c3_x<0.33)
        plot(x_pca_boundary(1,i),x_pca_boundary(2,i),'k*');
        hold on
    elseif(abs(p_c2_x-p_c3_x)<0.12 & p_c1_x<0.33)
        plot(x_pca_boundary(1,i),x_pca_boundary(2,i),'k*');
        hold on
    elseif(abs(p_c3_x-p_c1_x)<0.12 & p_c2_x<0.33)
        plot(x_pca_boundary(1,i),x_pca_boundary(2,i),'k*');
        hold on
    end
end 



%% Error Rate
check=[1*ones(1,length(x_pca(1,901:1000))),2*ones(1,length(x_pca(1,1901:2000))),3*ones(1,length(x_pca(1,2901:3000)))];
num_error=0;
num_error1=0;
num_error2=0;
num_error3=0;
for j=1:length(check)
    if j>=1 & j<=100
        if check(j)~=estimation(j)
            num_error1=num_error1+1;
        end
    elseif j>=101 & j<=200
        if check(j)~=estimation(j)
            num_error2=num_error2+1;
        end
    else
        if check(j)~=estimation(j)
            num_error3=num_error3+1;
        end
    end
end
title('Training data')
xlabel('Basis1')
ylabel('Basis2')

num_error=num_error1+num_error2+num_error3;
error_rate=num_error/length(check);
num_error1
num_error2
num_error3
display('Error Rate= ');
display(error_rate);