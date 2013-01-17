function model = backprop (hiddenUnits, inputs, targets, trainIterations)
    inputUnits  = size (inputs,  1);
    outputUnits = size (targets, 1);
    samples     = size (inputs,  2);

    t  = targets;
    m  = samples;
    w1 = initWeights (hiddenUnits, inputUnits);
    w2 = initWeights (outputUnits, hiddenUnits);
    y0 = inputs;

    for i = 1:trainIterations
        y1 = sigmoid (w1 * y0);
        y2 = sigmoid (w2 * y1);

        % E = 1 / (2 * m) * sum (sum ((t - y2) .^ 2));

        d_E_z2 = sigmoid (y2) .* (1 - sigmoid (y2)) .* (y2 - t);
        d_E_z1 = sigmoid (y1) .* (1 - sigmoid (y1)) .* (w2' * d_E_z2);

        d_E_w2 = d_E_z2 * y1' / m;
        d_E_w1 = d_E_z1 * y0' / m;

        w2 -= 0.5 * d_E_w2;
        w1 -= 0.5 * d_E_w1;
    end
    model.hiddenWeights = w1;
    model.outputWeights = w2;
end

function w = initWeights (in, out)
    w = reshape (0.1 * sin (1:in * out), in, out);
end

function y = sigmoid (x)
    y = 1 ./ (1 + exp (-x));
end
