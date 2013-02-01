function sum_compare (r, c, iterations)
    m = rand (r, c);

    tic;
    for i = 1:iterations
        x = sum (sum (m));
    end
    t = toc;
    result = x;

    mCl = cl_matrix (m);
    tic;
    for i = 1:iterations
        x = cl_sum (mCl);
    end
    tCl = toc;
    resultCl = double (x);

    assert (result, resultCl, 1e-7);

    fprintf (["                                           \n"...
             "           sum time     %10.6f              \n"...
             "ratio = ------------- = ----------- = %f    \n"...
             "         cl_sum time    %10.6f            \n\n"], t, t / tCl, tCl);
end
