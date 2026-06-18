figure
SNR_axis = -10:2:20;
plot(SNR_axis, acc(1,:)*100, 'Color', '#2F8000', 'LineWidth', 2)
hold on
plot(SNR_axis, acc(2,:)*100, 'Color', '#DEAA0B', 'LineWidth', 2)
plot(SNR_axis, acc(3,:)*100, 'Color', '#FF3300', 'LineWidth', 2)
plot(SNR_axis, acc(4,:)*100, 'Color', '#AD00E6', 'LineWidth', 2)
plot(SNR_axis, acc(5,:)*100, 'Color', '#0066FF', 'LineWidth', 2)
plot(SNR_axis, acc(6,:)*100, 'k', 'LineWidth', 2)
plot(SNR_axis, 23.7*ones(1,16), 'k--');
hold off
axis([-10 20 -2 102])
xlabel('SNR (dB)')
xticks(SNR_axis)
ylabel('Accuracy (%)')
legend('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'Noise', 'Reference', 'FontSize', 12, 'Location', 'West')
ax = gca;
ax.FontSize = 12;