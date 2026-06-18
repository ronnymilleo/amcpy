%%
bpsk_counter = zeros(1,16);
qpsk_counter = zeros(1,16);
psk8_counter = zeros(1,16);
qam16_counter = zeros(1,16);
qam64_counter = zeros(1,16);
noise_counter = zeros(1,16);
%%
for i = 1:16
    for j = 1:100
        if BPSK(i, j) == 0
            bpsk_counter(i) = bpsk_counter(i) + 1;
        end
        if QPSK(i, j) == 1
            qpsk_counter(i) = qpsk_counter(i) + 1;
        end
        if PSK8(i, j) == 2
            psk8_counter(i) = psk8_counter(i) + 1;
        end
        if QAM16(i, j) == 3
            qam16_counter(i) = qam16_counter(i) + 1;
        end
        if QAM64(i, j) == 4
            qam64_counter(i) = qam64_counter(i) + 1;
        end
        if Noise(i, j) == 5
            noise_counter(i) = noise_counter(i) + 1;
        end
    end
end
%%
figure(1)
SNR_axis = -10:2:20;
plot(SNR_axis, flip(bpsk_counter), 'Color', '#2F8000', 'LineWidth', 2)
hold on
plot(SNR_axis, flip(qpsk_counter), 'Color', '#DEAA0B', 'LineWidth', 2)
plot(SNR_axis, flip(psk8_counter), 'Color', '#FF3300', 'LineWidth', 2)
plot(SNR_axis, flip(qam16_counter), 'Color', '#AD00E6', 'LineWidth', 2)
plot(SNR_axis, flip(qam64_counter), 'Color', '#0066FF', 'LineWidth', 2)
plot(SNR_axis, flip(noise_counter), 'k', 'LineWidth', 2)
hold off
axis([-10 20 -2 102])
xlabel('SNR (dB)')
xticks(SNR_axis)
ylabel('Accuracy (%)')
legend('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'Noise', 'FontSize', 12, 'Location', 'West')
ax = gca;
ax.FontSize = 12;
%%
save('embedded_system.mat')