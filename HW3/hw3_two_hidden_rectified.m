clear all
close all
clc


%% Load Image from three folders
Data=[];
TestingData=[];
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

new_class1=randperm(1000);
new_class2=randperm(1000)+1000;
new_class3=randperm(1000)+2000;
% new_class1=1:1000;
% new_class2=1001:2000;
% new_class3=2001:3000;

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
x_pca=[];

mean_aug_test=[];
%% PCA projection
eigen_choose=1:2;
for i=1:size(new_Data,1)
    mean_aug_test=[mean_aug_test;mean_vector];
end
X_hat_test=new_Data-mean_aug_test;
for i=1:size(new_Data,1)
    x_pca=[x_pca,U(:,eigen_choose)'*(X_hat_test(i,:)')];
end

% load('x_pca.mat','x_pca');
x_pca=x_pca./1000;
x_pca_train=[x_pca(:,1:900),x_pca(:,1001:1900),x_pca(:,2001:2900)];
x_pca_train=[x_pca_train;ones(1,900)*1,ones(1,900)*2,ones(1,900)*3];
new_index=randperm(length(x_pca_train));
x_pca_train=x_pca_train(:,new_index);

x_pca_test=[x_pca(:,901:1000),x_pca(:,1901:2000),x_pca(:,2901:3000)];

%% 2 hidden layer
% node setting
D=2;
M=4;
W=4;
K=3;
step=0.005; % learning rate
flag=1;
stop_cri=1*1e-1;
times=1;
%%
w1_MD=randn(M,D)*1;  % default all weightings equal to 1
w2_WM=randn(W,M)*1;
w3_KW=randn(K,W)*1;

while flag==1
     times
     
     w1_MD_old=w1_MD;
     w2_WM_old=w2_WM;
     w3_KW_old=w3_KW;

for i=1:length(x_pca_train)
    x_n=x_pca_train(1:2,i);

    w_j0=1;
    w_k0=1;
    w_w0=1;
    
    a_j=zeros(M,1);
    for j=1:M
        for d=1:D
             a_j(j)=a_j(j)+w1_MD(j,d)*x_n(d);
        end
       a_j(j)=a_j(j)+w_j0;
    end
    z_j=rectified(a_j);   %h

    a_w=zeros(W,1);
    for w=1:W
        for j=1:M
            a_w(w)=a_w(w)+w2_WM(w,j)*z_j(j);
        end
        a_w(w)=a_w(w)+w_w0;
    end
    z_w=rectified(a_w);   %h
      
    a_k=zeros(K,1);
    for k=1:K
        for w=1:W
            a_k(k)=a_k(k)+w3_KW(k,w)*z_w(w);
        end
        a_k(k)=a_k(k)+w_k0;
    end
 
    y_k=zeros(K,1);
    [MAX,index]=max(a_k);
    for k=1:K
        y_k(k)=exp(a_k(k)-MAX)/(exp(a_k(1)-MAX)+exp(a_k(2)-MAX)+exp(a_k(3)-MAX));
    end
     
    if x_pca_train(3,i)==1
        tn=[1;0;0];
    elseif x_pca_train(3,i)==2
        tn=[0;1;0];
    else
        tn=[0;0;1];
    end
    
    delta_k=y_k-tn;

    for k=1:K
        for w=1:W
            w3_KW(k,w)= w3_KW(k,w)-step * (delta_k(k)*z_w(w));
        end
    end
    h_aw=rectified_der(a_w);
    delta_w=zeros(W,1);
    for w=1:W
        buff=0;
        for k=1:K
            buff=buff+w3_KW(k,w)*delta_k(k);
        end
        delta_w(w)=h_aw(w)*buff;
    end
        
   for w=1:W
        for m=1:M
            w2_WM(w,m)= w2_WM(w,m)-step * (delta_w(w)*z_j(m));
        end
    end
    
    h_aj=rectified_der(a_j);
    delta_j=zeros(M,1);
    for j=1:M
        buff=0;
        for w=1:W
            buff=buff+w2_WM(w,j)*delta_w(w);
        end
        delta_j(j)=h_aj(j)*buff;
    end
    
    for m=1:M
        for d=1:D
            w1_MD(m,d)= w1_MD(m,d)-step * (delta_j(m)*x_n(d));
        end
    end
    
%      w1_MD
%      w2_WM
%      w3_KW
    
end
%% Stop Condition
cal1=0;
cal2=0;
cal3=0;
for m=1:M
    for d=1:D
       cal1=cal1+abs(w1_MD(m,d)-w1_MD_old(m,d));
    end
end
for w=1:W
    for m=1:M
        cal2=cal2+abs(w2_WM(w,m)-w2_WM_old(w,m));
    end
end
for k=1:K
    for w=1:W
        cal3=cal3+abs( w3_KW(k,w)- w3_KW_old(k,w));
    end
end
[cal1,cal2,cal3]
if cal1<=stop_cri & cal2<=stop_cri & cal3<=stop_cri
    flag=0;
end
if times==30
    display('terminate')
    break
end

%% Error function
err=0;
for i=1:length(x_pca_train)
    x_n=x_pca_train(1:2,i);

    w_j0=1;
    w_k0=1;
    w_w0=1;
    
    a_j=zeros(M,1);
    for j=1:M
        for d=1:D
             a_j(j)=a_j(j)+w1_MD(j,d)*x_n(d);
        end
       a_j(j)=a_j(j)+w_j0;
    end
    z_j=rectified(a_j);   %h

    a_w=zeros(W,1);
    for w=1:W
        for j=1:M
            a_w(w)=a_w(w)+w2_WM(w,j)*z_j(j);
        end
        a_w(w)=a_w(w)+w_w0;
    end
    z_w=rectified(a_w);   %h
      
    a_k=zeros(K,1);
    for k=1:K
        for w=1:W
            a_k(k)=a_k(k)+w3_KW(k,w)*z_w(w);
        end
        a_k(k)=a_k(k)+w_k0;
    end
 
    y_k=zeros(K,1);
    [MAX,index]=max(a_k);
    for k=1:K
        y_k(k)=exp(a_k(k)-MAX)/(exp(a_k(1)-MAX)+exp(a_k(2)-MAX)+exp(a_k(3)-MAX));
    end
     
    if x_pca_train(3,i)==1
        tn=[1;0;0];
    elseif x_pca_train(3,i)==2
        tn=[0;1;0];
    else
        tn=[0;0;1];
    end
    
    for k=1:K
        err=err+ (-tn(k) * log (y_k(k)) );
    end
    
end

%% Error function
error=0;
for i=1:length(x_pca_test)
     x_n=x_pca_test(:,i);

    w_j0=1;
    w_k0=1;
    w_w0=1;
    
    a_j=zeros(M,1);
    for j=1:M
        for d=1:D
             a_j(j)=a_j(j)+w1_MD(j,d)*x_n(d);
        end
       a_j(j)=a_j(j)+w_j0;
    end
    z_j=rectified(a_j);   %h

    a_w=zeros(W,1);
    for w=1:W
        for j=1:M
            a_w(w)=a_w(w)+w2_WM(w,j)*z_j(j);
        end
        a_w(w)=a_w(w)+w_w0;
    end
    z_w=rectified(a_w);   %h
      
    a_k=zeros(K,1);
    for k=1:K
        for w=1:W
            a_k(k)=a_k(k)+w3_KW(k,w)*z_w(w);
        end
        a_k(k)=a_k(k)+w_k0;
    end
 
    y_k=zeros(K,1);
    [MAX,index]=max(a_k);
    for k=1:K
        y_k(k)=exp(a_k(k)-MAX)/(exp(a_k(1)-MAX)+exp(a_k(2)-MAX)+exp(a_k(3)-MAX));
    end
     
    [max_value,max_index]=max(y_k);
    if i>=1 & i<=100
        if max_index~=1
            error=error+1;
        end
    elseif i>=101 & i<=200
        if max_index~=2
            error=error+1;
        end
    else
        if max_index~=3
            error=error+1;
        end
    end
    
end
[err,error]

times=times+1;
end
%% Decision Boundary
xrange = [-1 1];
yrange = [-1 1];
inc = 0.025;
 
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.
xy=xy';
estimation=zeros(1,length(xy));

for i=1:length(xy)
    x_n=xy(:,i);

    w_j0=1;
    w_k0=1;
    w_w0=1;
    
    a_j=zeros(M,1);
    for j=1:M
        for d=1:D
             a_j(j)=a_j(j)+w1_MD(j,d)*x_n(d);
        end
       a_j(j)=a_j(j)+w_j0;
    end
    z_j=rectified(a_j);   %h

    a_w=zeros(W,1);
    for w=1:W
        for j=1:M
            a_w(w)=a_w(w)+w2_WM(w,j)*z_j(j);
        end
        a_w(w)=a_w(w)+w_w0;
    end
    z_w=rectified(a_w);   %h
      
    a_k=zeros(K,1);
    for k=1:K
        for w=1:W
            a_k(k)=a_k(k)+w3_KW(k,w)*z_w(w);
        end
        a_k(k)=a_k(k)+w_k0;
    end
 
    y_k=zeros(K,1);
    [MAX,index]=max(a_k);
    for k=1:K
        y_k(k)=exp(a_k(k)-MAX)/(exp(a_k(1)-MAX)+exp(a_k(2)-MAX)+exp(a_k(3)-MAX));
    end
    
    [max_value,max_index]=max(y_k);
    estimation(i)=max_index;
end

decisionmap=reshape(estimation,image_size);
figure;
imagesc(xrange,yrange,decisionmap);
cmap = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1];
colormap(cmap);
hold on
for i=1:length(x_pca_train)
    if x_pca_train(3,i)==1
        plot(x_pca_train(1,i),x_pca_train(2,i),'rs');
    elseif x_pca_train(3,i)==2
        plot(x_pca_train(1,i),x_pca_train(2,i),'g^');
    else
        plot(x_pca_train(1,i),x_pca_train(2,i),'bo');
    end
    hold on

end
xlabel('x1')
ylabel('x2')
title('Decision Boudary')




