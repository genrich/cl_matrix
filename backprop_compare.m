function backprop_compare (hiddenUnits, inputs, targets, trainIterations)
    [model t]     = backprop    (hiddenUnits, inputs, targets, trainIterations);
    [modelCl tCl] = backprop_cl (hiddenUnits, inputs, targets, trainIterations);

    fprintf (["                                              \n"...
             "         backprop  time    %10.6f              \n"...
             "ratio = ---------------- = ----------- = %f    \n"...
             "        backprop_cl time   %10.6f            \n\n"], t, t / tCl, tCl);

    m        = size (inputs,  2);
    t        = targets;
    output   = sigmoid (model.outputWeights   * sigmoid (model.hiddenWeights   * inputs));
    outputCl = sigmoid (modelCl.outputWeights * sigmoid (modelCl.hiddenWeights * inputs));
    e        = 1 / (2 * m) * sum (sum ((t - output)   .^ 2));
    eCl      = 1 / (2 * m) * sum (sum ((t - outputCl) .^ 2));

    fprintf ("error   = %f\n", e);
    fprintf ("errorCl = %f\n", eCl);
end

function y = sigmoid (x)
    y = 1 ./ (1 + exp (-x));
end
