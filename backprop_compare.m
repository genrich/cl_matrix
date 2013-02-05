function backprop_compare (hiddenUnits, inputs, targets, trainIterations)
    [model t]     = backprop    (hiddenUnits, inputs, targets, trainIterations);
    [modelCl tCl] = backprop_cl (hiddenUnits, inputs, targets, trainIterations);

    fprintf (["                                     \n"...
             "          time       %10.6f           \n"...
             "ratio = --------- = ------------ = %f \n"...
             "         cl time     %10.6f           \n\n"], t, t / tCl, tCl);

    m        = size (inputs,  2);
    t        = targets;
    output   = sigmoid (model.outputWeights   * sigmoid (model.hiddenWeights   * inputs));
    outputCl = sigmoid (modelCl.outputWeights * sigmoid (modelCl.hiddenWeights * inputs));
    err      = 1 / (2 * m) * sum (sum ((t - output)   .^ 2));
    errCl    = 1 / (2 * m) * sum (sum ((t - outputCl) .^ 2));

    assert (err, errCl, 1e-7);
end

function y = sigmoid (x)
    y = 1 ./ (1 + exp (-x));
end
