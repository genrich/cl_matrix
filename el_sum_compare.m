function el_sum_compare (r, c, iterations)
    m1 = rand (r, c);
    m2 = rand (r, c);

    tic;
    for i = 1:iterations
        x = m1 + m2;
    end
    t = toc;
    result = x;

    m1Cl = cl_matrix (m1);
    m2Cl = cl_matrix (m2);
    tic;
    for i = 1:iterations
        x = cl_fun ("x1 + x2", m1Cl, m2Cl);
    end
    tCl = toc;
    resultCl = double (x);

    assert (result, resultCl, 1e-7);

    fprintf (["                                     \n"...
             "          time       %10.6f           \n"...
             "ratio = --------- = ------------ = %f \n"...
             "         cl time     %10.6f           \n\n"], t, t / tCl, tCl);
end
