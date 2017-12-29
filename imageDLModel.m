digitDatasetPath = fullfile('/home/mackenzie/Fall_Sem17/IndptStudy/Test7Pictures');
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

img = readimage(digitData,1);
size(img) 
trainNumFiles = 1;

[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles); %i took out the randomize 3rd parameter
%dont include the above line in the audio, is it needed here?
layers = [
    imageInputLayer([64 64 3])

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', 'MaxEpochs',3, ..., %getting an error with imageDLModel/trainingOptions? %'ValidationData',valDigitData,... %'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainDigitData,layers,options);
predictedLabels = classify(net,valDigitData);
valLabels = valDigitData.Labels;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels);