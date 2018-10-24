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
Data=new_Data;

%% Fisher
 
m1=mean(Data(1:900,:))';
m2=mean(Data(1001:1900,:))';
m3=mean(Data(2001:2900,:))';

%% S_w
s1_sum=zeros(size(Data,2));
for i=1:900
    x=Data(i,:)';
    s1_sum=s1_sum+(x-m1)*(x-m1)';
end
s1=s1_sum;

s2_sum=zeros(size(Data,2));
for i=1001:1900
    x=Data(i,:)';
    s2_sum=s2_sum+(x-m1)*(x-m1)';
end
s2=s2_sum;

s3_sum=zeros(size(Data,2));
for i=2001:2900
    x=Data(i,:)';
    s3_sum=s3_sum+(x-m1)*(x-m1)';
end
s3=s3_sum;

S_w=s1+s2+s3;

%% S_b
Nk=length(1:900);
m=(Nk*m1+Nk*m2+Nk*m3)/(3*Nk);
mk=[m1,m2,m3];
Sb_sum=zeros(size(m,1));
for i=1:3
    Sb_sum=Sb_sum+Nk*(mk(:,i)-m)*(mk(:,i)-m)';
end
S_b=Sb_sum;

[U,S,V]=svd(inv(S_w)*S_b);

mean_aug_test=[];
for i=1:size(Data,1)
    mean_aug_test=[mean_aug_test;m'];
end

X_zero_mean=Data-mean_aug_test;
x_pca=[];
for i=1:size(Data,1)
    x_pca=[x_pca,U(:,1:2)'*(X_zero_mean(i,:)')];
end

 %%
 
figure;
subplot(1,2,1)
plot(x_pca(1,1:900),x_pca(2,1:900),'r*');
hold on
plot(x_pca(1,1001:1900),x_pca(2,1001:1900),'g*');
hold on
plot(x_pca(1,2001:2900),x_pca(2,2001:2900),'b*');
title('Training data')
xlabel('Basis1')
ylabel('Basis2')
subplot(1,2,2)
plot(x_pca(1,901:1000),x_pca(2,901:1000),'r*');
hold on
plot(x_pca(1,1901:2000),x_pca(2,1901:2000),'g*');
hold on
plot(x_pca(1,2901:3000),x_pca(2,2901:3000),'b*');
title('Testing data')
xlabel('Basis1')
ylabel('Basis2')



%% Training
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

mu1=transpose(mean(x_pca(:,1:900)'));
mu2=transpose(mean(x_pca(:,1001:1900)'));
mu3=transpose(mean(x_pca(:,2001:2900)'));

reg1=zeros(2);
reg2=reg1;
reg3=reg1;
for i=1:900
    reg1=reg1+(x_pca(:,i)-mu1)*(x_pca(:,i)-mu1)';
end
reg1=reg1/900;
for i=1001:1900
    reg2=reg2+(x_pca(:,i)-mu2)*(x_pca(:,i)-mu2)';
end
reg2=reg2/900;
for i=2001:2900
    reg3=reg3+(x_pca(:,i)-mu3)*(x_pca(:,i)-mu3)';
end
reg3=reg3/900;

covar=1/3*(reg1+reg2+reg3);

inv_covar=inv(covar);
prop1=length(x_pca(1,1:900))/length(x_pca_train(1,:)); % same probability
prop2=length(x_pca(1,1001:1900))/length(x_pca_train(1,:));
prop3=length(x_pca(1,2001:2900))/length(x_pca_train(1,:));

w1=inv_covar*mu1;
w2=inv_covar*mu2;
w3=inv_covar*mu3;

w10=-1/2*mu1'*inv_covar*mu1+log(prop1);
w20=-1/2*mu2'*inv_covar*mu2+log(prop2);
w30=-1/2*mu3'*inv_covar*mu3+log(prop3);

%% Testing

estimation=[];
for i=1:length(x_pca_test)
    a1=w1'*x_pca_test(:,i)+w10;
    a2=w2'*x_pca_test(:,i)+w20;
    a3=w3'*x_pca_test(:,i)+w30;

    p_c1_x=exp(a1)/(exp(a1)+exp(a2)+exp(a3));
    p_c2_x=exp(a2)/(exp(a1)+exp(a2)+exp(a3));
    p_c3_x=exp(a3)/(exp(a1)+exp(a2)+exp(a3));

    [M,I]=max([p_c1_x,p_c2_x,p_c3_x]);
    estimation=[estimation,I];
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
num_error=num_error1+num_error2+num_error3;
error_rate=num_error/length(check);

display('Error Rate= ');
display(error_rate);

