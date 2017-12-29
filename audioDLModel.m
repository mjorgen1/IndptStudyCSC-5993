digitDatasetPath = fullfile('/home/mackenzie/Fall_Sem17/IndptStudy/Test3Sounds');
digitData = imageDatastore(digitDatasetPath,...
   'IncludeSubfolders',true,'LabelSource','foldernames', 'FileExtensions', '.mp3', 'ReadFcn', @readAudio);
%y = audioread(filename.wav);
%make a vector and then filter that through the architecture
trainNumFiles = 1;
[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles);
layers = [
    imageInputLayer([33075 1])
    convolution2dLayer([1600,1], 16, 'Stride', [2, 1], 'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([1600, 1],32, 'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([1600, 1], 64, 'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...
    'Verbose',false,...
    'Plots','training-progress');
net = trainNetwork(trainDigitData,layers,options);
predictedLabels = classify(net,valDigitData); 
valLabels = valDigitData.Labels; 
accuracy = sum(predictedLabels == valLabels)/numel(valLabels); 