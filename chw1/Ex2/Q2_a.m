%% CHW01 - Computational Intelligence - Question.2.a
% Armin Panjehpour - 98101288


%% Part.2.a

%% Part.2.0.1 - plot the data
clc; clear all; close all;

Ex2 = load('Ex2.mat');


TrainData = Ex2.TrainData;
TestData = Ex2.TestData;

% seperate two groups
TrainDataZeros = TrainData(1:3,find(TrainData(4,:) == 0));
TrainDataOnes = TrainData(1:3,find(TrainData(4,:) == 1));

figure;
% zeros with blue
scatter3(TrainDataZeros(1,:),TrainDataZeros(2,:),TrainDataZeros(3,:),'filled')
xlabel('x','interpreter','latex')
ylabel('y','interpreter','latex')
zlabel('z','interpreter','latex')

% ones with red
hold on;
scatter3(TrainDataOnes(1,:),TrainDataOnes(2,:),TrainDataOnes(3,:),'filled');


legend('Class 0','Class 1')

%% Part.2.0.2 - create the validation matrix
clc; close all;

valTrainVector = zeros(1,size(TrainData,2));
valTrainVector(randperm(size(TrainData,2),size(TrainData,2)/5)) = 1;

valDataNums = find(valTrainVector == 1);
trainDataNums = find(valTrainVector == 0);
NewTrainData = TrainData(:,trainDataNums);
ValData = TrainData(:,valDataNums);


%% Part.2.a.1 - one hidden layer - one output - [0 1] classifying the data
clc; close all;

train_input = NewTrainData(1:3,:);
train_output = NewTrainData(4,:);
val_input = ValData(1:3,:);
val_output = ValData(4,:);

% structure of the network
hiddenLayerNeuronNum = 10;
net = patternnet(hiddenLayerNeuronNum);


% on train data
net = train(net,train_input,train_output);
yEstimateTrain = net(train_input);

% make the estimates binary with a threshold
threshold = 0.5; 
yEstimateTrain(yEstimateTrain >= threshold) = 1;
yEstimateTrain(yEstimateTrain < threshold) = 0;

% mse on training data
MSEOnTrainData = perform(net,yEstimateTrain,train_output)

% mse on val data
yEstimateVal = net(val_input);

% make the estimates binary with a threshold
yEstimateVal(yEstimateVal >= threshold) = 1;
yEstimateVal(yEstimateVal < threshold) = 0;
MSEOnValData = perform(net,yEstimateVal,val_output)


%% Part.2.a.2 - PLOT MSEs of the trained classifier
clc; close all;

figure;
subplot(1,2,1);
yline(MSEOnTrainData,'Color','blue','LineWidth',2);
title("MSE on Training Data Using Fitnet - One Hidden Layer - NeuronsNum = " + hiddenLayerNeuronNum...
    ,'interpreter','latex');
grid on; grid minor;
ylim([min([MSEOnValData MSEOnTrainData])-5 max([MSEOnValData MSEOnTrainData])+5]);

subplot(1,2,2);
yline(MSEOnValData,'Color','red','LineWidth',2);
title("MSE on Validation Data Using Fitnet - One Hidden Layer - NeuronsNum = " + hiddenLayerNeuronNum...
    ,'interpreter','latex');grid on; grid minor;
ylim([min([MSEOnValData MSEOnTrainData])-5 max([MSEOnValData MSEOnTrainData])+5]);

%% Part.2.a.3 - Apply the classifier on the test data
clc; close all;

yEstimateTest = net(TestData);

% make the estimates binary with a threshold
threshold = 0.5;
yEstimateTest(yEstimateTest >= threshold) = 1;
yEstimateTest(yEstimateTest < threshold) = 0;


% seperate two groups
TestDataZeros = TestData(:,find(yEstimateTest == 0));
TestDataOnes = TestData(:,find(yEstimateTest == 1));

figure;
% zeros with blue
scatter3(TestDataZeros(1,:),TestDataZeros(2,:),TestDataZeros(3,:),'filled')
xlabel('x','interpreter','latex')
ylabel('y','interpreter','latex')
zlabel('z','interpreter','latex')
title('Test Data Estimated Labels','interpreter','latex')

% ones with red
hold on;
scatter3(TestDataOnes(1,:),TestDataOnes(2,:),TestDataOnes(3,:),'filled');
view(-30,10)

legend('Class 0','Class 1')

