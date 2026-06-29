module op_sheath_m
    !! This module contains the implementation of the op_sheath_t type.
    use genex_fortran_env_m, only: GP
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    ! from PARALLAX
    use csrmat_m, only: csrmat_t

    implicit none
    private

    type, public, abstract :: op_sheath_t
        !! The op_sheath operator provides the functionality to calculate
        !! the cutoff velocity, the sheath potential and the type of the sheath
        !! given the distribution function
        !!
        !! NOTE: Currently the operator only functions as a skeleton and
        !!       implements no functionality
        class(mesh_5d_t), pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), pointer :: dcomm_handler
        !! Pointer to the dcomm_handler
    contains
        procedure(initialize_interface), public, deferred :: initialize
        procedure(apply_interface), public, deferred :: apply
    end type

    abstract interface
        subroutine initialize_interface(this, dcomm_handler, mesh)
            import op_sheath_t, dcomm_handler_t, mesh_5d_t
            class(op_sheath_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

        subroutine apply_interface(this, lb, ub, lb_stripped, ub_stripped, &
                                   dist_func, is_electron_sheath, &
                                   cutoff_index, cutoff_index_fraction, &
                                   sheath_pot)
            import op_sheath_t, GP
            !! Calculates the sheath potential and the cutoff velocities given
            !! dist_func
            class(op_sheath_t), intent(inout) :: this
            !! Instance of the type
            integer, dimension(5), intent(in) :: lb, ub
            !! Lower and upper bounds of the dist_func array
            integer, dimension(5), intent(in) :: lb_stripped, ub_stripped
            !! Lower and upper bounds of the dist_func array excluding ghost
            !! cells
            real(kind = GP), intent(in) :: dist_func(lb(1) : ub(1), &
                                                     lb(2) : ub(2), &
                                                     lb(3) : ub(3), &
                                                     lb(4) : ub(4), &
                                                     lb(5) : ub(5) )
            !! Distribution function
            logical, intent(out) :: is_electron_sheath( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            !! .true. if there is an electron sheath at a given point and
            !! .false. otherwise
            integer, intent(out) :: cutoff_index( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            !! Cutoff index
            real(kind = GP), intent(out) :: cutoff_index_fraction( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            !! Weight of the cutoff index. If the the cutoff velocity falls
            !! in between two grid points only a fraction of particles at the
            !! cutoff velocity are reflected to ensure exact charge
            !! conservation.
            real(kind = GP), intent(out) :: sheath_pot( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            !! Sheath potential
        end subroutine
    end interface

    type, public, extends(op_sheath_t) :: op_sheath_cpu_t
        !! Implementation of the op_sheath_t operator on the CPU
    contains
        procedure, public :: initialize => initialize_cpu
        procedure, public :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, dcomm_handler, mesh)
            class(op_sheath_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_cpu(this, lb, ub, lb_stripped, ub_stripped, &
                                   dist_func, is_electron_sheath, &
                                   cutoff_index, cutoff_index_fraction, &
                                   sheath_pot)
            class(op_sheath_cpu_t), intent(inout) :: this
            integer, dimension(5), intent(in) :: lb, ub
            integer, dimension(5), intent(in) :: lb_stripped, ub_stripped
            real(kind = GP), intent(in) :: dist_func(lb(1) : ub(1), &
                                                     lb(2) : ub(2), &
                                                     lb(3) : ub(3), &
                                                     lb(4) : ub(4), &
                                                     lb(5) : ub(5) )
            logical, intent(out) :: is_electron_sheath( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            integer, intent(out) :: cutoff_index( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            real(kind = GP), intent(out) :: cutoff_index_fraction( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
            real(kind = GP), intent(out) :: sheath_pot( &
                lb(1)          : ub(1), &
                lb_stripped(2) : ub_stripped(2))
        end subroutine
    end interface
end module
