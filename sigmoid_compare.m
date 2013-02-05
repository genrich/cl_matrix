function sigmoid_compare (r, c, iterations)
    m = rand (r, c);

    tic;
    for i = 1:iterations
        x = sigmoid (m);
    end
    t = toc;
    result = x;

    mCl = cl_matrix (m);
    tic;
    for i = 1:iterations
        x = cl_fun ("1 / (1 + exp (-x))", mCl);
    end
    tCl = toc;
    resultCl = double (x);

    assert (result, resultCl, 1e-7);

    fprintf (["                                     \n"...
             "          time       %10.6f           \n"...
             "ratio = --------- = ------------ = %f \n"...
             "         cl time     %10.6f           \n\n"], t, t / tCl, tCl);
end

function y = sigmoid (x)
    y = 1 ./ (1 + exp (-x));
end
