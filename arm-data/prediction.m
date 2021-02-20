clear
clc
load('BPSK.mat', 'Data')
bpsk_predict = cell2mat(Data(:,2,1));
load('QPSK.mat', 'Data')
qpsk_predict = cell2mat(Data(:,2,1));
load('PSK8.mat', 'Data')
psk8_predict = cell2mat(Data(:,2,1));
load('QAM16.mat', 'Data')
qam16_predict = cell2mat(Data(:,2,1));
load('QAM64.mat', 'Data')
qam64_predict = cell2mat(Data(:,2,1));
load('noise.mat', 'Data')
noise_predict = cell2mat(Data(:,2,1));
clear Data

bpsk_counter = 0;
for i = 1:length(bpsk_predict)
    if bpsk_predict(i) == 0
        bpsk_counter = bpsk_counter + 1;
    end
end
bpsk_acc = bpsk_counter * 100 / length(bpsk_predict);
fprintf('BPSK acc: %.2f %% - %d frames\n', bpsk_acc, length(bpsk_predict))

qpsk_counter = 0;
for i = 1:length(qpsk_predict)
    if qpsk_predict(i) == 1
        qpsk_counter = qpsk_counter + 1;
    end
end
qpsk_acc = qpsk_counter * 100 / length(qpsk_predict);
fprintf('QPSK acc: %.2f %% - %d frames\n', qpsk_acc, length(qpsk_predict))

psk8_counter = 0;
for i = 1:length(psk8_predict)
    if psk8_predict(i) == 2
        psk8_counter = psk8_counter + 1;
    end
end
psk8_acc = psk8_counter * 100 / length(psk8_predict);
fprintf('PSK8 acc: %.2f %% - %d frames\n', psk8_acc, length(psk8_predict))

qam16_counter = 0;
for i = 1:length(qam16_predict)
    if qam16_predict(i) == 3
        qam16_counter = qam16_counter + 1;
    end
end
qam16_acc = qam16_counter * 100 / length(qam16_predict);
fprintf('QAM16 acc: %.2f %% - %d frames\n', qam16_acc, length(qam16_predict))

qam64_counter = 0;
for i = 1:length(qam64_predict)
    if qam64_predict(i) == 4
        qam64_counter = qam64_counter + 1;
    end
end
qam64_acc = qam64_counter * 100 / length(qam64_predict);
fprintf('QAM64 acc: %.2f %% - %d frames\n', qam64_acc, length(qam64_predict))

noise_counter = 0;
for i = 1:length(noise_predict)
    if noise_predict(i) == 5
        noise_counter = noise_counter + 1;
    end
end
noise_acc = noise_counter * 100 / length(qam64_predict);
fprintf('NOISE acc: %.2f %% - %d frames\n', noise_acc, length(noise_predict))