#ifndef OCTAVE_CL_MATRIX_HPP
#define OCTAVE_CL_MATRIX_HPP

#include <ostream>
#include <octave/oct.h>

#include <ClMatrix.hpp>

class octave_cl_matrix: public octave_base_value
{
    const ClMatrix   matrix;

public:
                    octave_cl_matrix  ();
                    octave_cl_matrix  (const Matrix&);
                    octave_cl_matrix  (ClMatrix);
                    ~octave_cl_matrix ();
    const ClMatrix& cl_matrix_value   ()                                                 const;
    void            print             (std::ostream& os, bool pr_as_read_syntax = false) const;
    bool            is_constant       ()                                                 const;
    bool            is_defined        ()                                                 const;
    dim_vector      dims              ()                                                 const;
    size_t          byte_size         ()                                                 const;
    Matrix          matrix_value      (bool)                                             const;
    octave_value    resize            (const dim_vector&, bool)                          const;

    DECLARE_OCTAVE_ALLOCATOR
    DECLARE_OV_TYPEID_FUNCTIONS_AND_DATA
};

#endif // OCTAVE_CL_MATRIX_HPP
