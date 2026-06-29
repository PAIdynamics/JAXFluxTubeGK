module op_source_m
    !! Contains the implementation of different source operators
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_5d_t
    use mesh_5d_m, only: mesh_5d_t
    use op_base_m, only: op_base_t
    use params_source_m, only: params_source_loc_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_source_t
        !! Base type for all kind of source operators
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
    contains
        procedure(apply_interface), deferred :: apply
        procedure(initialize_interface), deferred :: initialize
    end type

    abstract interface
        subroutine initialize_interface(this, mesh)
            !! Initializes the source operator
            import op_source_t, mesh_5d_t
            class(op_source_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

        subroutine apply_interface(this, da_f_out)
            !! Applies the source operator by adding the source term to f_out
            import op_source_t, data_array_5d_t
            class(op_source_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(inout) :: da_f_out
            !! Distribution function
        end subroutine
    end interface

    type, public, abstract, extends(op_source_t) :: op_source_loc_t
        !! Base type for all kind of localized source operators
        integer, private, allocatable, dimension(:) :: n_source_loc
        !! Number of localized sources to apply
        real(kind=GP), private, allocatable, dimension(:,:) :: norm_fac
        !! Normalization factor of the localized source
        real(kind=GP), private, allocatable, dimension(:,:) :: amp
        !! Amplitudes of the localized sources for different species
        real(kind=GP), private, allocatable, dimension(:,:) :: temp
        !! Temperatures of the localized sources for different species
        real(kind=GP), private, allocatable, dimension(:,:) :: rho_mid
        !! Centers of the localized sources for the different species
        real(kind=GP), private, allocatable, dimension(:,:) :: width
        !! Widths of the localized sources for the different species
        real(kind=GP), private, allocatable, dimension(:,:) :: prefac_is_pure
        !! Prefactor to account for if the source is pure. Is allocated only
        !! for the density source
    contains
        procedure, private :: initialize_source_norm_fac
    end type

    interface
        module subroutine initialize_source_norm_fac(this, s, n)
            !! Initializes the normalization factor of the localized sources
            class(op_source_loc_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: s
            !! Localized source label
            integer, intent(in) :: n
            !! Species label
        end subroutine
    end interface

    type, public, abstract, extends(op_source_loc_t) :: op_source_loc_dens_t
       !! Operator representing a localized particle source
       !! for the grid approach.
    end type

    type, public, abstract, extends(op_source_loc_t) :: op_source_loc_torque_t
       !! Operator representing a localized torque source
       !! for the grid approach.
    end type

    type, public, abstract, extends(op_source_loc_t) :: op_source_loc_heat_t
        !! Operator representing a localized heat source
        !! for the grid approach.
    end type

    type, public, abstract, extends(op_source_loc_t) :: &
                                                    op_source_loc_dens_vspec_t
        !! Operator representing a localized particle source
        !! for the spectral approach.
    end type

    type, public, abstract, extends(op_source_loc_t) :: &
                                                    op_source_loc_torque_vspec_t
        !! Operator representing a localized torque source
        !! for the spectral approach.
    end type

    type, public, abstract, extends(op_source_loc_t) :: &
                                                    op_source_loc_heat_vspec_t
        !! Operator representing a localized heat source
        !! for the spectral approach.
    end type

    type, public, extends(op_source_loc_dens_t) :: op_source_loc_dens_cpu_t
        !! Operator representing a localized particle source.
        !! CPU implementation for grid approach.
    contains
        procedure :: initialize => initialize_dens_cpu
        procedure :: apply => apply_dens_cpu
    end type

    interface
        module subroutine initialize_dens_cpu(this, mesh)
            class(op_source_loc_dens_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_dens_cpu(this, da_f_out)
           class(op_source_loc_dens_cpu_t), target, intent(inout) :: this
           class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_source_loc_torque_t) :: op_source_loc_torque_cpu_t
        !! Operator representing a localized torque source.
        !! CPU implementation for grid approach.
    contains
        procedure :: initialize => initialize_torque_cpu
        procedure :: apply => apply_torque_cpu
    end type

    interface
        module subroutine initialize_torque_cpu(this, mesh)
            class(op_source_loc_torque_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_torque_cpu(this, da_f_out)
           class(op_source_loc_torque_cpu_t), target, intent(inout) :: this
           class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_source_loc_heat_t) :: op_source_loc_heat_cpu_t
        !! Operator representing a localized heat source.
        !! CPU implementation for grid approach.
    contains
        procedure :: initialize => initialize_heat_cpu
        procedure :: apply => apply_heat_cpu
    end type

    interface
        module subroutine initialize_heat_cpu(this, mesh)
            class(op_source_loc_heat_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_heat_cpu(this, da_f_out)
            class(op_source_loc_heat_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_source_loc_dens_vspec_t) :: &
                                                op_source_loc_dens_vspec_cpu_t
        !! Operator representing a localized particle source.
        !! CPU implementation for spectral approach.
        real(kind=GP), private, allocatable, dimension(:,:,:) :: h0p
        !! Integral of Hermite polynomial times vp^0
        real(kind=GP), private, allocatable, dimension(:,:,:) :: l0j
        !! Integral of Laguerre polynomial times mu^0
        real(kind=GP), private, allocatable, dimension(:,:,:) :: h2p
        !! Integral of Hermite polynomial times vp^2
        real(kind=GP), private, allocatable, dimension(:,:,:) :: l1j
        !! Integral of Laguerre polynomial times mu^1
        real(kind=GP), private, allocatable, dimension(:,:) :: beta
        !! Array of ratios between spectral reference and
        !! source temperatures.
    contains
        procedure :: initialize => initialize_dens_vspec_cpu
        procedure :: apply => apply_dens_vspec_cpu
    end type

    interface
        module subroutine initialize_dens_vspec_cpu(this, mesh)
            class(op_source_loc_dens_vspec_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_dens_vspec_cpu(this, da_f_out)
            class(op_source_loc_dens_vspec_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_source_loc_torque_vspec_t) :: &
                                                op_source_loc_torque_vspec_cpu_t
        !! Operator representing a localized torque source.
        !! CPU implementation for spectral approach.
        real(kind=GP), private, allocatable, dimension(:,:,:) :: h1p
        !! Integral of Hermite polynomial times vp^1
        real(kind=GP), private, allocatable, dimension(:,:,:) :: l0j
        !! Integral of Laguerre polynomial times mu^0
        real(kind=GP), private, allocatable, dimension(:,:) :: beta
        !! Array of ratios between spectral reference and
        !! source temperatures.
    contains
        procedure :: initialize => initialize_torque_vspec_cpu
        procedure :: apply => apply_torque_vspec_cpu
    end type

    interface
        module subroutine initialize_torque_vspec_cpu(this, mesh)
            class(op_source_loc_torque_vspec_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_torque_vspec_cpu(this, da_f_out)
            class(op_source_loc_torque_vspec_cpu_t), target, &
                                                          intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_source_loc_heat_vspec_t) :: &
                                                op_source_loc_heat_vspec_cpu_t
        !! Operator representing a localized heat source
        !! CPU implementation for spectral approach.
        real(kind=GP), private, allocatable, dimension(:,:,:) :: h2p
        !! Integral of Hermite polynomial times vp^2
        real(kind=GP), private, allocatable, dimension(:,:,:) :: h0p
        !! Integral of Hermite polynomial times vp^0
        real(kind=GP), private, allocatable, dimension(:,:,:) :: l1j
        !! Integral of Laguerre polynomial times mu^1
        real(kind=GP), private, allocatable, dimension(:,:,:) :: l0j
        !! Integral of Laguerre polynomial times mu^0
        real(kind=GP), private, allocatable, dimension(:,:) :: beta
        !! Array of ratios between spectral reference and
        !! source temperatures.
    contains
        procedure :: initialize => initialize_heat_vspec_cpu
        procedure :: apply => apply_heat_vspec_cpu
    end type

    interface
        module subroutine initialize_heat_vspec_cpu(this, mesh)
            class(op_source_loc_heat_vspec_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_heat_vspec_cpu(this, da_f_out)
            class(op_source_loc_heat_vspec_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, abstract, extends(op_source_t) :: op_source_lapd_t
        !! Source for LAPD simulations described in
        !! Pan et al, Physics of Plasmas 25, 062303 (2018)
        real(kind=GP), private, allocatable, dimension(:,:) :: rho_buffer
        !! Buffer for the values of the flux surface label rho
        real(kind=GP), private :: extent
        !! Radial extent of the source
        real(kind=GP), private :: width
        !! Radial fall off length of the source
        real(kind=GP), private :: strength
        !! Strength of the source
    contains
        procedure(initialize_lapd_interface), deferred :: initialize
    end type

    abstract interface
        subroutine initialize_lapd_interface(this, mesh)
            import op_source_lapd_t, mesh_5d_t
            class(op_source_lapd_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine
    end interface

    type, public, extends(op_source_lapd_t) :: op_source_lapd_cpu_t
        !! Source for LAPD simulations implemented on CPU
    contains
        procedure :: initialize => initialize_lapd_cpu
        procedure :: apply => apply_lapd_cpu
    end type

    interface
        module subroutine initialize_lapd_cpu(this, mesh)
            class(op_source_lapd_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_lapd_cpu(this, da_f_out)
            class(op_source_lapd_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

end module
