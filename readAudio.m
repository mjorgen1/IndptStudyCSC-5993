function data = readAudio(audioName) 
    sizevec = zeros(33075,1); %second and a half
    data = audioread(audioName);
    data(numel(sizevec))=0;
end

