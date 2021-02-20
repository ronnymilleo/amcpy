pred = zeros(1,16);
for i = 1:160
    if predictions(i) == 0 && i <= 10
        pred(1) = pred(1) + 1;
    end
    if predictions(i) == 0 && i > 10 && i <= 20
        pred(2) = pred(2) + 1;
    end
    if predictions(i) == 0 && i > 20 && i <= 30
        pred(3) = pred(3) + 1;
    end
    if predictions(i) == 0 && i > 30 && i <= 40
        pred(4) = pred(4) + 1;
    end
    if predictions(i) == 0 && i > 40 && i <= 50
        pred(5) = pred(5) + 1;
    end
    if predictions(i) == 0 && i > 50 && i <= 60
        pred(6) = pred(6) + 1;
    end
    if predictions(i) == 0 && i > 60 && i <= 70
        pred(7) = pred(7) + 1;
    end
    if predictions(i) == 0 && i > 70 && i <= 80
        pred(8) = pred(8) + 1;
    end
    if predictions(i) == 0 && i > 80 && i <= 90
        pred(9) = pred(9) + 1;
    end
    if predictions(i) == 0 && i > 90 && i <= 100
        pred(10) = pred(10) + 1;
    end
    if predictions(i) == 0 && i > 100 && i <= 110
        pred(11) = pred(11) + 1;
    end
    if predictions(i) == 0 && i > 110 && i <= 120
        pred(12) = pred(12) + 1;
    end
    if predictions(i) == 0 && i > 120 && i <= 130
        pred(13) = pred(13) + 1;
    end
    if predictions(i) == 0 && i > 130 && i <= 140
        pred(14) = pred(14) + 1;
    end
    if predictions(i) == 0 && i > 140 && i <= 150
        pred(15) = pred(15) + 1;
    end
    if predictions(i) == 0 && i > 150 && i <= 160
        pred(16) = pred(16) + 1;
    end
end
plot(pred)