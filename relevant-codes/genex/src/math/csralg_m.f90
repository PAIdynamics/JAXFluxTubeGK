module csralg_m
    !! Module implementing features of the matrix algebra with csr matrices

    use genex_fortran_env_m, only: GP
    ! from PARALLAX
    use csrmat_m, only: csrmat_t

contains

    pure real(kind = GP) function csrmv(mat, v, i)
        !! Computes the i-th component of the matrix vector product between
        !! the given csr matrix mat and the vector v
        !! NOTE: This subroutine has been hand optimized
        type(csrmat_t), intent(in) :: mat
        real(kind = GP), dimension(:), intent(in) :: v
        integer, intent(in) :: i

        csrmv = sum(  mat%val(mat%i(i) : mat%i(i + 1) - 1) &
                    * v(mat%j(mat%i(i) : mat%i(i + 1) - 1)))
    end function
end module
