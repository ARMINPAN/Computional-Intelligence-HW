%% CHW01 - Computational Intelligence - Question.1
% Armin Panjehpour - 98101288

%% Part.1

%% Part.1.a - plot the data
clc; clear all; close all;

Ex1 = load('Ex1.mat');


figure;
scatter3(Ex1.NOemission,Ex1.speed,Ex1.fuelrate);
xlabel('NO Emission','interpreter','latex');
ylabel('Speed','interpreter','latex');
zlabel('Fuel Rate','interpreter','latex');
title('FuelRate vs NO emission and Speed','interpreter','latex')

%% Part.1.b - create train and validation data
clc; close all;

% NO emission --> % speed --> % fuel rate


trainData = zeros(3,700);
valData = zeros(3,length(Ex1.fuelrate) - 700);

% train and validation data
trainData = [Ex1.NOemission(1:700);Ex1.speed(1:700);Ex1.fuelrate(1:700)];
valData = [Ex1.NOemission(701:end);Ex1.speed(701:end);Ex1.fuelrate(701:end)];


%% Part.1.c.1 - linear regression on training data
clc; close all;


% linear regression on training data
X = [ones(size(trainData,2),1) trainData(1,:)' trainData(2,:)'];
y = trainData(3,:)';


% data
figure;
scatter3(trainData(1,:)',trainData(2,:)',trainData(3,:)','MarkerEdgeColor','#A2142F')
xlabel('NO Emission','interpreter','latex');
ylabel('Speed','interpreter','latex');
zlabel('Fuel Rate','interpreter','latex');

% regressed plane
b  = regress(y,X);   % Removes NaN data
hold on;
x1fit = min(trainData(1,:)):10:max(trainData(1,:));
x2fit = min(trainData(2,:)):10:max(trainData(2,:));
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT;
s = mesh(X1FIT,X2FIT,YFIT,'FaceAlpha',0.93);
s.FaceColor = 'flat';
title('Linear Regressed Plane to the Training Data','interpreter','latex');
hold off;


%% Part.1.c.2 - calculate MSE of the fitted plane
clc; close all;
figure;

% on train data
MSETrainPlane = 0;
for i = 1:size(trainData,2)
    fittedValue = b(1) + b(2)*trainData(1,i) + b(3)*trainData(2,i);
    realValue = trainData(3,i);
    MSETrainPlane = MSETrainPlane + (fittedValue-realValue).^2;
end

subplot(1,2,1);
yline(MSETrainPlane/size(trainData,2),'Color','blue','LineWidth',2);
title('Mean Square Error on Training Data for Linear Regression','interpreter','latex');
grid on; grid minor;

    
% on validation data
MSEValPlane = 0;
for i = 1:size(valData,2)
    fittedValue = b(1) + b(2)*valData(1,i) + b(3)*valData(2,i);
    realValue = valData(3,i);
    MSEValPlane = MSEValPlane + (fittedValue-realValue).^2;
end

% for subplot1
ylim([min([MSEValPlane/size(valData,2) MSETrainPlane/size(trainData,2)]-300) ...
    max([MSEValPlane/size(valData,2) MSETrainPlane/size(trainData,2)]+300)]);

subplot(1,2,2);
yline(MSEValPlane/size(valData,2),'Color','red','LineWidth',2);
title('Mean Square Error on Test Data for Linear Regression','interpreter','latex');
grid on; grid minor;
ylim([min([MSEValPlane/size(valData,2) MSETrainPlane/size(trainData,2)]-300) ...
    max([MSEValPlane/size(valData,2) MSETrainPlane/size(trainData,2)]+300)]);

%% Part.1.d.1 - logistic regression
% in order to do this regression, we act like the slides of the course,
% taking the ln of the outputs and do a linear regression.
clc; close all;
 

% logistic regression on training data
X = [ones(size(trainData,2),1) trainData(1,:)' trainData(2,:)'];
Y = max(trainData(3,:))+1;
y = trainData(3,:)';
z = log((Y-y)./y);


% data
figure;
scatter3(trainData(1,:)',trainData(2,:)',trainData(3,:)','MarkerEdgeColor','#A2142F')
xlabel('NO Emission','interpreter','latex');
ylabel('Speed','interpreter','latex');
zlabel('Fuel Rate','interpreter','latex');

% regressed
B  = regress(z,X);   % Removes NaN data
hold on;
x1fit = min(trainData(1,:)):10:max(trainData(1,:));
x2fit = min(trainData(2,:)):10:max(trainData(2,:));
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = Y./(1+exp(B(1) + B(2)*X1FIT + B(3)*X2FIT));
s = mesh(X1FIT,X2FIT,YFIT,'FaceAlpha',0.93);
s.FaceColor = 'flat';
title('Logistic Regressed Plane to the Training Data','interpreter','latex');
hold off;

%% Part.1.d.2 - calculate MSE of the logistic regression 
clc; close all;
figure;

% on train data
MSETrainLogistic = 0;
for i = 1:size(trainData,2)
    fittedValue = Y./(1+exp(B(1) + B(2)*trainData(1,i) + B(3)*trainData(2,i)));
    realValue = trainData(3,i);
    MSETrainLogistic = MSETrainLogistic + (fittedValue-realValue).^2;
end

subplot(1,2,1);
yline(MSETrainLogistic/size(trainData,2),'Color','blue','LineWidth',2);
title('Mean Square Error on Training Data for Logistic Regression','interpreter','latex');
grid on; grid minor;


% on validation data
MSEValLogistic = 0;
for i = 1:size(valData,2)
    fittedValue = Y./(1+exp(B(1) + B(2)*valData(1,i) + B(3)*valData(2,i)));
    realValue = valData(3,i);
    MSEValLogistic = MSEValLogistic + (fittedValue-realValue).^2;
end

% for subplot1
ylim([min([MSEValLogistic/size(valData,2) MSETrainLogistic/size(trainData,2)]-300) ...
    max([MSEValLogistic/size(valData,2) MSETrainLogistic/size(trainData,2)]+300)]);

subplot(1,2,2);
yline(MSEValLogistic/size(valData,2),'Color','red','LineWidth',2);
title('Mean Square Error on Test Data for Logistic Regression','interpreter','latex');
grid on; grid minor;
ylim([min([MSEValLogistic/size(valData,2) MSETrainLogistic/size(trainData,2)]-300) ...
    max([MSEValLogistic/size(valData,2) MSETrainLogistic/size(trainData,2)]+300)]);


%% Part.1.e.1 - fitting the data using nnstart toolbox
clc; close all;


% normalizing data
train_input = trainData(1:2,:);
meanTrainInput = mean(train_input,2);
stdTrainInput = std(train_input,0,2);
normalizedTrainInput = (train_input - meanTrainInput)./stdTrainInput;

train_output = trainData(3,:);
meanTrainOutput = mean(train_output,2);
stdTrainOutput = std(train_output,0,2);
normalizedTrainOutput = (train_output - meanTrainOutput)./stdTrainOutput;

val_input = valData(1:2,:);
normalizedValInput = (val_input - meanTrainInput)./stdTrainInput;

val_output = valData(3,:);
normalizedValOutput = (val_output - meanTrainOutput)./stdTrainOutput;

% structure of the network
hiddenLayerNeuronNum = 7;
net = fitnet(hiddenLayerNeuronNum,'trainlm');
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

% on train data
net = train(net,normalizedTrainInput,normalizedTrainOutput);

% mse on training data
yEstimateTrainNormalized = net(normalizedTrainInput);
MSEOnTrainData = perform(net,yEstimateTrainNormalized*stdTrainOutput + meanTrainOutput...
   ,train_output)

% mse on val data
yEstimateValNormalized = net(normalizedValInput);
MSEOnValData = perform(net,yEstimateValNormalized*stdTrainOutput + meanTrainOutput...
    ,val_output)


%% Part.1.e.2 - fitting the data using nnstart toolbox MSE plots
clc; close all;

figure;
subplot(1,2,1);
yline(MSEOnTrainData,'Color','blue','LineWidth',2);
title("MSE on Training Data Using Fitnet - One Hidden Layer - NeuronsNum = " + hiddenLayerNeuronNum...
    ,'interpreter','latex');
grid on; grid minor;
ylim([min([MSEOnValData MSEOnTrainData])-300 max([MSEOnValData MSEOnTrainData])+300]);

subplot(1,2,2);
yline(MSEOnValData,'Color','red','LineWidth',2);
title("MSE on Test Data Using Fitnet - One Hidden Layer - NeuronsNum = " + hiddenLayerNeuronNum...
    ,'interpreter','latex');grid on; grid minor;
ylim([min([MSEOnValData MSEOnTrainData])-300 max([MSEOnValData MSEOnTrainData])+300]);

%% ALL MSEs in one figure;

AllMSEs = [MSEValPlane/size(valData,2)  MSETrainPlane/size(trainData,2) ...
    MSEValLogistic/size(valData,2) MSETrainLogistic/size(valData,2) ...
    MSEOnValData MSEOnTrainData];

figure;
subplot(2,3,1);
yline(MSETrainPlane/size(trainData,2),'Color','blue','LineWidth',2);
title('MSE on Training Data for Linear Regression','interpreter','latex');
grid on; grid minor;
ylim([min(AllMSEs)-300 max(AllMSEs+300)]);

subplot(2,3,4);
yline(MSEValPlane/size(valData,2),'Color','red','LineWidth',2);
title('MSE on Test Data for Linear Regression','interpreter','latex');
grid on; grid minor;
ylim([min(AllMSEs)-300 max(AllMSEs+300)]);


subplot(2,3,2);
yline(MSETrainLogistic/size(trainData,2),'Color','blue','LineWidth',2);
title('MSE on Training Data for Logistic Regression','interpreter','latex');
grid on; grid minor;
ylim([min(AllMSEs)-300 max(AllMSEs+300)]);


subplot(2,3,5);
yline(MSEValLogistic/size(valData,2),'Color','red','LineWidth',2);
title('MSE on Test Data for Logistic Regression','interpreter','latex');
grid on; grid minor;
ylim([min(AllMSEs)-300 max(AllMSEs+300)]);

subplot(2,3,3);
yline(MSEOnTrainData,'Color','blue','LineWidth',2);
title("MSE on Training Data - MLP - NeuronsNum = " + hiddenLayerNeuronNum...
    ,'interpreter','latex');
grid on; grid minor;
ylim([min(AllMSEs)-300 max(AllMSEs+300)]);


subplot(2,3,6);
yline(MSEOnValData,'Color','red','LineWidth',2);
title("MSE on Test Data - MLP - NeuronsNum = " + hiddenLayerNeuronNum...
    ,'interpreter','latex');grid on; grid minor;
ylim([min(AllMSEs)-300 max(AllMSEs+300)]);
