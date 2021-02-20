figure(1)
plot(project.accuracy, '-ok', 'MarkerSize', 4)
hold on
plot(24, 0.9119, '*k', 'MarkerSize', 14)
hold off
ax = gca;
ax.FontSize = 12;
legend('Trained Neural Networks', 'Best Neural Network', 'FontSize', 12)
xlabel('Neural network number identification')
ylabel('Training Accuracy')

figure(2)
plot(project.val_accuracy, '-ok', 'MarkerSize', 4)
hold on
plot(17, 0.9147, '*k', 'MarkerSize', 14)
hold off
ax = gca;
ax.FontSize = 12;
legend('Trained Neural Networks', 'Best Neural Network', 'FontSize', 12)
xlabel('Neural network number identification')
ylabel('Validation Accuracy')

figure(3)
plot(project.loss, '-ok', 'MarkerSize', 4)
hold on
plot(16, 0.1924, '*k', 'MarkerSize', 14)
hold off
ax = gca;
ax.FontSize = 12;
legend('Trained Neural Networks', 'Best Neural Network', 'FontSize', 12)
xlabel('Neural network number identification')
ylabel('Training Loss')

figure(4)
plot(project.val_loss, '-ok', 'MarkerSize', 4)
hold on
plot(16, 0.1927, '*k', 'MarkerSize', 14)
hold off
ax = gca;
ax.FontSize = 12;
legend('Trained Neural Networks', 'Best Neural Network', 'FontSize', 12)
xlabel('Neural network number identification')
ylabel('Validation Loss')

%%
a = movmean(project.accuracy, 10);
figure(1)
scatter(project.layer_size_hl1,project.accuracy)
hold on
plot(a)
hold off
ax = gca;
ax.FontSize = 12;
legend('Accuracy for the number of neurons', 'Accuracy moving average', 'FontSize', 12)
xlabel('Number of neurons on hidden layer 1')
ylabel('Accuracy')

figure(2)
scatter(project.layer_size_hl2,project.accuracy)
hold on
plot(a)
hold off
ax = gca;
ax.FontSize = 12;
legend('Accuracy for the number of neurons', 'Accuracy moving average', 'FontSize', 12)
xlabel('Number of neurons on hidden layer 2')
ylabel('Accuracy')

figure(3)
scatter(project.layer_size_hl3,project.accuracy)
hold on
plot(a)
hold off
ax = gca;
ax.FontSize = 12;
legend('Accuracy for the number of neurons', 'Accuracy moving average', 'FontSize', 12)
xlabel('Number of neurons on hidden layer 3')
ylabel('Accuracy')