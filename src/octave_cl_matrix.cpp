#include <octave/config.h>
#include <octave/ops.h>
#include <octave/ov-re-mat.h>
#include <octave/ov-scalar.h>
#include <octave/ov-str-mat.h>

#include "octave_cl_matrix.hpp"
#include "ClService.hpp"

using namespace std;

DEFINE_OCTAVE_ALLOCATOR (octave_cl_matrix);
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_cl_matrix, "cl_matrix", "cl_matrix");

static bool type_loaded = false;

extern ClService clSrvc;

#define CL_MATRIX_BINOP(fn, method, a1, a2)               \
    static octave_cl_matrix* fn (const a1, const a2)      \
    {                                                     \
        try                                               \
        {                                                 \
            return new octave_cl_matrix {mat.method (x)}; \
        }                                                 \
        catch (exception& e)                              \
        {                                                 \
            report_error (e.what ());                     \
        }                                                 \
    }
//__________________________________________________________________________________________________

void report_error (string msg)
{
    (*current_liboctave_error_handler) (msg.c_str ());
}
//__________________________________________________________________________________________________

octave_cl_matrix::octave_cl_matrix ():
    matrix {1, 1, nullptr}
{
}
//__________________________________________________________________________________________________

octave_cl_matrix::octave_cl_matrix (const Matrix& m):
    matrix {m.rows (), m.cols (), m.data ()}
{
}
//__________________________________________________________________________________________________

octave_cl_matrix::octave_cl_matrix (ClMatrix m):
    matrix {move (m)}
{
}
//__________________________________________________________________________________________________

octave_cl_matrix::~octave_cl_matrix ()
{
}
//__________________________________________________________________________________________________

const ClMatrix& octave_cl_matrix::cl_matrix_value () const
{
    return matrix;
}
//__________________________________________________________________________________________________

void octave_cl_matrix::print (std::ostream& os, bool pr_as_read_syntax) const
{
    os << "cl_matrix (" << matrix.rows << ", " << matrix.cols << ");  content: double (CL_MATRIX)\n";
}
//__________________________________________________________________________________________________

bool octave_cl_matrix::is_constant () const
{
    return true;
}
//__________________________________________________________________________________________________

bool octave_cl_matrix::is_defined () const
{
    return true;
}
//__________________________________________________________________________________________________

dim_vector octave_cl_matrix::dims () const
{
    return dim_vector (matrix.rows, matrix.cols);
}
//__________________________________________________________________________________________________

size_t octave_cl_matrix::byte_size () const
{
    return matrix.byteSize;
}
//__________________________________________________________________________________________________

Matrix octave_cl_matrix::matrix_value (bool = false) const
{
    Matrix retval {static_cast<int> (matrix.rows), static_cast<int> (matrix.cols)};
    matrix.copyTo (retval.fortran_vec ());
    return retval;
}
//__________________________________________________________________________________________________

octave_value octave_cl_matrix::resize (const dim_vector&, bool) const
{
    report_error ("octave_cl_matrix::resize (): wrong type argument 'cl_matrix'");
}
//__________________________________________________________________________________________________

static octave_cl_matrix* uminus (const ClMatrix& mat)
{
    try
    {
        return new octave_cl_matrix {mat.uminus ()};
    }
    catch (exception& e)
    {
        report_error (e.what ());
    }
}
//__________________________________________________________________________________________________

static octave_cl_matrix* transpose (const ClMatrix& mat)
{
    try
    {
        return new octave_cl_matrix {mat.transpose ()};
    }
    catch (exception& e)
    {
        report_error (e.what ());
    }
}
//__________________________________________________________________________________________________

CL_MATRIX_BINOP (add,            add,        ClMatrix& mat, ClMatrix& x)
CL_MATRIX_BINOP (add_mat_scalar, add,        ClMatrix& mat, double x)
CL_MATRIX_BINOP (add_scalar_mat, add,        double x,      ClMatrix& mat)
CL_MATRIX_BINOP (sub,            sub,        ClMatrix& mat, ClMatrix& x)
CL_MATRIX_BINOP (sub_mat_scalar, sub,        ClMatrix& mat, double x)
CL_MATRIX_BINOP (sub_scalar_mat, subtrahend, double x,      ClMatrix& mat)
CL_MATRIX_BINOP (mul,            mul,        ClMatrix& mat, ClMatrix& x)
CL_MATRIX_BINOP (mul_mat_scalar, mul,        ClMatrix& mat, double x)
CL_MATRIX_BINOP (mul_scalar_mat, mul,        double x,      ClMatrix& mat)
CL_MATRIX_BINOP (pow_mat_scalar, pow,        ClMatrix& mat, double x)
CL_MATRIX_BINOP (pow_scalar_mat, exponent,   double x,      ClMatrix& mat)
CL_MATRIX_BINOP (el_mul,         el_mul,     ClMatrix& mat, ClMatrix& x)
CL_MATRIX_BINOP (trans_mul,      trans_mul,  ClMatrix& mat, ClMatrix& x)
CL_MATRIX_BINOP (mul_trans,      mul_trans,  ClMatrix& mat, ClMatrix& x)

static octave_cl_matrix* div_mat_scalar (const ClMatrix& mat, const double x)
{
    if (x == 0)
        gripe_divide_by_zero ();

    try
    {
        return new octave_cl_matrix {mat.div (x)};
    }
    catch (exception& e)
    {
        report_error (e.what ());
    }
}
//__________________________________________________________________________________________________

CL_MATRIX_BINOP (div_scalar_mat, divisor, double x,      ClMatrix& mat)
CL_MATRIX_BINOP (el_div,         el_div,  ClMatrix& mat, ClMatrix& x)

DEFUNOP_FN (uminus,    cl_matrix, uminus)
DEFUNOP_FN (transpose, cl_matrix, transpose)

DEFBINOP_FN (add,            cl_matrix, cl_matrix, add)
DEFBINOP_FN (add_mat_scalar, cl_matrix, scalar,    add_mat_scalar)
DEFBINOP_FN (add_scalar_mat, scalar,    cl_matrix, add_scalar_mat)
DEFBINOP_FN (sub,            cl_matrix, cl_matrix, sub)
DEFBINOP_FN (sub_mat_scalar, cl_matrix, scalar,    sub_mat_scalar)
DEFBINOP_FN (sub_scalar_mat, scalar,    cl_matrix, sub_scalar_mat)
DEFBINOP_FN (mul,            cl_matrix, cl_matrix, mul)
DEFBINOP_FN (mul_mat_scalar, cl_matrix, scalar,    mul_mat_scalar)
DEFBINOP_FN (mul_scalar_mat, scalar,    cl_matrix, mul_scalar_mat)
DEFBINOP_FN (pow_mat_scalar, cl_matrix, scalar,    pow_mat_scalar)
DEFBINOP_FN (pow_scalar_mat, scalar,    cl_matrix, pow_scalar_mat)
DEFBINOP_FN (el_mul,         cl_matrix, cl_matrix, el_mul)
DEFBINOP_FN (trans_mul,      cl_matrix, cl_matrix, trans_mul)
DEFBINOP_FN (mul_trans,      cl_matrix, cl_matrix, mul_trans)
DEFBINOP_FN (div_mat_scalar, cl_matrix, scalar,    div_mat_scalar)
DEFBINOP_FN (div_scalar_mat, scalar,    cl_matrix, div_scalar_mat)
DEFBINOP_FN (el_div,         cl_matrix, cl_matrix, el_div)

DEFCONV(cl_matrix_to_matrix, octave_cl_matrix, octave_matrix)
{
  CAST_CONV_ARG (const octave_cl_matrix&);
  return new octave_matrix {v.matrix_value ()};
}
//__________________________________________________________________________________________________

DEFUN_DLD (cl_matrix, args, nargout,
"-*- texinfo -*-                                                           \n\
@deftypefn{Loadable Function} {@var{ret} =} cl_matrix (@var{double_matrix})\n\
Create OpenCL matrix                                                       \n\
@end deftypefn                                                             \n")
{
    int nargin = args.length ();
    if (nargin == 1 && (args (0).is_real_scalar () || args (0).is_real_matrix ()))
    {
        if (!type_loaded)
        {
            if (!clSrvc.initialized)
            {
                report_error ("Unable to initialize! " + clSrvc.statusMsg);
                return {};
            }

            octave_cl_matrix::register_type ();
            mlock ();

            INSTALL_UNOP (op_uminus,    octave_cl_matrix, uminus);
            INSTALL_UNOP (op_transpose, octave_cl_matrix, transpose);
            INSTALL_UNOP (op_hermitian, octave_cl_matrix, transpose);

            INSTALL_BINOP (op_add,       octave_cl_matrix, octave_cl_matrix, add);
            INSTALL_BINOP (op_add,       octave_cl_matrix, octave_scalar,    add_mat_scalar);
            INSTALL_BINOP (op_add,       octave_scalar,    octave_cl_matrix, add_scalar_mat);
            INSTALL_BINOP (op_sub,       octave_cl_matrix, octave_cl_matrix, sub);
            INSTALL_BINOP (op_sub,       octave_cl_matrix, octave_scalar,    sub_mat_scalar);
            INSTALL_BINOP (op_sub,       octave_scalar,    octave_cl_matrix, sub_scalar_mat);
            INSTALL_BINOP (op_mul,       octave_cl_matrix, octave_cl_matrix, mul);
            INSTALL_BINOP (op_mul,       octave_cl_matrix, octave_scalar,    mul_mat_scalar);
            INSTALL_BINOP (op_mul,       octave_scalar,    octave_cl_matrix, mul_scalar_mat);
            INSTALL_BINOP (op_pow,       octave_cl_matrix, octave_scalar,    pow_mat_scalar);
            INSTALL_BINOP (op_el_mul,    octave_cl_matrix, octave_cl_matrix, el_mul);
            INSTALL_BINOP (op_el_mul,    octave_cl_matrix, octave_scalar,    mul_mat_scalar);
            INSTALL_BINOP (op_el_mul,    octave_scalar,    octave_cl_matrix, mul_scalar_mat);
            INSTALL_BINOP (op_el_pow,    octave_cl_matrix, octave_scalar,    pow_mat_scalar);
            INSTALL_BINOP (op_el_pow,    octave_scalar,    octave_cl_matrix, pow_scalar_mat);
            INSTALL_BINOP (op_trans_mul, octave_cl_matrix, octave_cl_matrix, trans_mul);
            INSTALL_BINOP (op_mul_trans, octave_cl_matrix, octave_cl_matrix, mul_trans);
            INSTALL_BINOP (op_herm_mul,  octave_cl_matrix, octave_cl_matrix, trans_mul);
            INSTALL_BINOP (op_mul_herm,  octave_cl_matrix, octave_cl_matrix, mul_trans);
            INSTALL_BINOP (op_div,       octave_cl_matrix, octave_scalar,    div_mat_scalar);
            INSTALL_BINOP (op_div,       octave_scalar,    octave_cl_matrix, div_scalar_mat);
            INSTALL_BINOP (op_el_div,    octave_cl_matrix, octave_cl_matrix, el_div);
            INSTALL_BINOP (op_el_div,    octave_cl_matrix, octave_scalar,    div_mat_scalar);
            INSTALL_BINOP (op_el_div,    octave_scalar,    octave_cl_matrix, div_scalar_mat);

            INSTALL_CONVOP (octave_cl_matrix, octave_matrix, cl_matrix_to_matrix);

            type_loaded = true;
        }


        Matrix m = args (0).matrix_value ();
        if (!error_state)
        {
            try
            {
                return octave_value {new octave_cl_matrix {m}};
            }
            catch (exception& e)
            {
                report_error (e.what ());
            }
        }
    }
    else
    {
        print_usage ();
    }
    return {};
}
//__________________________________________________________________________________________________

DEFUN_DLD(cl_fun, args, nargout,
"-*- texinfo -*-                                                               \n\
@deftypefn {Mapping Function} {@var{C} =} cl_fun (@var{str}, @var{A})          \n\
@deftypefnx{Mapping Function} {@var{C} =} cl_fun (@var{str}, @var{A}, @var{B}) \n\
Apply function expression @var{str} to matrix elements.                        \n\
                                                                               \n\
Function expression @var{str} must be a string.                                \n\
It must be a valid OpenCL C kernel language expression.                        \n\
Matrices @var{A}, @var{B} must be instances of cl_matrix type.                 \n\
                                                                               \n\
If only one matrix @var{A} is provided then its elements can be accessed       \n\
in function expression as @code{x}                                             \n\
                                                                               \n\
@example                                                                       \n\
C = cl_fun ('1 / (1 + exp (-x))', A);                                          \n\
@end example                                                                   \n\
                                                                               \n\
If both matrices @var{A} and @var{B} are provided their corresponding          \n\
elements are @code{x1} @code{x2}                                               \n\
                                                                               \n\
@example                                                                       \n\
C = cl_fun ('x1 + x2', A, B);                                                  \n\
@end example                                                                   \n\
@end deftypefn                                                                 \n")
{
    int nargin = args.length ();
    if (nargin == 2
        && args (0). is_char_matrix ()
        && args (1). type_id () == octave_cl_matrix::static_type_id ())
    {
        try
        {
            const octave_char_matrix& s = dynamic_cast<const octave_char_matrix&> (args (0). get_rep ());
            const octave_cl_matrix&   m = dynamic_cast<const octave_cl_matrix&>   (args (1). get_rep ());

            const string& funExpr = s.string_value ();
            const ClMatrix& mat   = m.cl_matrix_value ();

            return octave_value {new octave_cl_matrix {mat.fun (funExpr)}};
        }
        catch (exception& e)
        {
            report_error (e.what ());
        }
    }
    else if (nargin == 3
        && args (0). is_char_matrix ()
        && args (1). type_id () == octave_cl_matrix::static_type_id ()
        && args (2). type_id () == octave_cl_matrix::static_type_id ())
    {
        try
        {
            const octave_char_matrix& s  = dynamic_cast<const octave_char_matrix&> (args (0). get_rep ());
            const octave_cl_matrix&   m1 = dynamic_cast<const octave_cl_matrix&>   (args (1). get_rep ());
            const octave_cl_matrix&   m2 = dynamic_cast<const octave_cl_matrix&>   (args (2). get_rep ());

            const string& funExpr = s.string_value ();
            const ClMatrix& mat1  = m1.cl_matrix_value ();
            const ClMatrix& mat2  = m2.cl_matrix_value ();

            return octave_value {new octave_cl_matrix {mat1.fun (funExpr, mat2)}};
        }
        catch (exception& e)
        {
            report_error (e.what ());
        }
    }
    else
    {
        print_usage ();
    }
    return {};
}
//__________________________________________________________________________________________________

DEFUN_DLD(cl_sigmoid, args, nargout,
"-*- texinfo -*-                                                               \n\
@deftypefn {Mapping Function} {@var{cl_matrix} =} cl_sigmoid (@var{cl_matrix}) \n\
Apply @code{sigmoid} function to @var{cl_matrix}                               \n\
                                                                               \n\
@example                                                                       \n\
@group                                                                         \n\
   1                                                                           \n\
-------                                                                        \n\
     -X                                                                        \n\
1 + e                                                                          \n\
@end group                                                                     \n\
@end example                                                                   \n\
@end deftypefn                                                                 \n")
{
    int nargin = args.length ();
    if (nargin == 1
        && args (0). type_id () == octave_cl_matrix::static_type_id ())
    {
        try
        {
            const octave_cl_matrix& m = dynamic_cast<const octave_cl_matrix&> (args (0). get_rep ());
            return octave_value {new octave_cl_matrix {m. cl_matrix_value (). sigmoid ()}};
        }
        catch (exception& e)
        {
            report_error (e.what ());
        }
    }
    else
    {
        print_usage ();
    }
    return {};
}
//__________________________________________________________________________________________________

DEFUN_DLD(cl_sum, args, nargout,
"-*- texinfo -*-                                                            \n\
@deftypefn {Loadable Function} {@var{cl_matrix} =} cl_sum (@var{cl_matrix}) \n\
Sum all elements of @var{cl_matrix} to produce @var{cl_matrix (1, 1)}       \n\
@end deftypefn                                                              \n")
{
    int nargin = args.length ();
    if (nargin == 1
        && args (0). type_id () == octave_cl_matrix::static_type_id ())
    {
        try
        {
            const octave_cl_matrix& m = dynamic_cast<const octave_cl_matrix&> (args (0). get_rep ());
            return octave_value {new octave_cl_matrix {m. cl_matrix_value (). sum ()}};
        }
        catch (exception& e)
        {
            report_error (e.what ());
        }
    }
    else
    {
        print_usage ();
    }
    return {};
}
