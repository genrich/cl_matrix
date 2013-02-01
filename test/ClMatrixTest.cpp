#include <string>

using namespace std;

#define BOOST_TEST_MODULE ClMatrixTest

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <ClMatrix.hpp>
#include <ClService.hpp>

constexpr double tolerance = 1e-7;
extern ClService clSrvc;

BOOST_AUTO_TEST_CASE (mul_SingleElementMitrixTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {2},
           data2[] = {3},
           result[1],
           expectedResult[] = {6};
    ClMatrix mat1 {1, 1, data1},
             mat2 {1, 1, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_Vectors1Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[1],
           expectedResult[] = {11};
    ClMatrix mat1 {1, 2, data1},
             mat2 {2, 1, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_Vectors2Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {2, 1, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {23, 34, 31, 46};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_ErrorTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3, 4};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    BOOST_CHECK_THROW (mat1.mul (mat2), runtime_error);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (trans_mul_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {17, 39, 23, 53};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.trans_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (trans_mul_VectorsTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1.trans_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_trans_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {26, 38, 30, 44};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.mul_trans (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_trans_Vectors1Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[1],
           expectedResult[] = {11};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1.mul_trans (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (mul_trans_Vectors2Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {2, 1, data1},
             mat2 {2, 1, data2};

    ClMatrix mat = mat1.mul_trans (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (add_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 2, INFINITY, -2},
           result[4],
           expectedResult[] = {6, 4, INFINITY, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.add (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (add_ErrorTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {1, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {1, 2, data2};

    BOOST_CHECK_THROW (mat1.add (mat2), runtime_error);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (sub_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 2, INFINITY, 2},
           result[4],
           expectedResult[] = {-4, 0, -INFINITY, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.sub (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (sub_ErrorTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {1, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 1, data2};

    BOOST_CHECK_THROW (mat1.sub (mat2), runtime_error);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (sub_scalar_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {-4, -3, -2, -1};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.sub (5);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (scalar_sub_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {4, 3, 2, 1};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.subtrahend (5);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (div_scalar_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {2, 4, 8, 16},
           result[4],
           expectedResult[] = {1, 2, 4, 8};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.div (2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (div_scalar0_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {INFINITY, INFINITY, INFINITY, INFINITY};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.div (0);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (scalar_div_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {2, 4, 8, 0},
           result[4],
           expectedResult[] = {4, 2, 1, INFINITY};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.divisor (8);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (el_mul_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {5, 12, 21, 32};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.el_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (el_mul_ErrorTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2},
           data2[] = {3};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 1, data2};

    BOOST_CHECK_THROW (mat1.el_mul (mat2), runtime_error);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (el_div_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {2, 4, 0, 16},
           data2[] = {2, 0, 4, -8},
           result[4],
           expectedResult[] = {1, INFINITY, 0, -2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.el_div (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (el_div_ErrorTest)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data1[] = {1, 2, 3, 4},
           data2[] = {1};
    ClMatrix mat1 {2, 2, data1},
             mat2 {1, 1, data2};

    BOOST_CHECK_THROW (mat1.el_div (mat2), runtime_error);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (uminus_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {2, INFINITY, 0, 16},
           result[4],
           expectedResult[] = {-2, -INFINITY, 0, -16};
    ClMatrix mat {2, 2, data};

    mat.uminus ().copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (transpose_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {1, 3, 2, 4};
    ClMatrix mat {2, 2, data};

    mat.transpose ().copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (sigmoid_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    constexpr int size = 4;
    double data[] = {1, 2, 3, 4},
           result[size];
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.sigmoid ();
    mat.copyTo (result);

    for (int i = 0; i < size; i++)
    {
        BOOST_CHECK_SMALL (result[i] - (1 / (1 + exp (-data[i]))), tolerance);
    }
}
//__________________________________________________________________________________________________

BOOST_AUTO_TEST_CASE (sum_Test)
{
    BOOST_CHECK_MESSAGE (clSrvc.initialized, clSrvc.statusMsg);
    constexpr int rows = 111, cols = 111, size = rows * cols;
    vector<double> data (size);

    uniform_real_distribution<double> distribution (-1, 1);
    mt19937 engine;
    auto generator = bind(distribution, engine);

    generate_n(data.begin (), size, generator);

    double expectedResult = accumulate (data.begin (), data.end (), 0.0);

    ClMatrix mat1 {rows, cols, data.data ()};
    ClMatrix mat = mat1.sum ();

    BOOST_CHECK_EQUAL (mat.rows, 1);
    BOOST_CHECK_EQUAL (mat.cols, 1);

    double result;
    mat. copyTo (&result);
    BOOST_CHECK_SMALL (expectedResult - result, tolerance);
}
