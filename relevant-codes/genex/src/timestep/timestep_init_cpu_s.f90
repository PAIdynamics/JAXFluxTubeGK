submodule (timestep_m) timestep_init_cpu_s
    !! Contains implementation of the initialization of CPU-supported operators
    !! for timestep_t
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use params_collisions_m, only: get_coll_type
    use params_source_m, only: get_source_type, get_n_source
    use params_bnd_cond_m, only: get_bnd_cond_type
    use op_diag_mom_0d_m, only: op_diag_mom_0d_t, op_diag_mom_0d_cpu_t
    use op_diag_mom_2d_m, only: op_diag_mom_2d_t, op_diag_mom_2d_cpu_t
    use op_rhs_vlasov_eq_static_m, only: op_rhs_vlasov_eq_static_cpu_t
    use op_rhs_vlasov_eq_dynamic_m, only: op_rhs_vlasov_eq_dynamic_cpu_t
    use op_mom_maxwells_eq_m, only: op_mom_maxwells_eq_cpu_t
    use op_mom_ohms_law_m, only: op_mom_ohms_law_cpu_t
    use op_coll_m, only: op_coll_t, op_coll_bgk_cpu_t, op_coll_lbd_cpu_t, &
                         op_coll_lorentz_cpu_t, op_coll_fpl_cpu_t
    use op_mom_coll_m, only: op_mom_coll_cpu_t
    use op_source_m, only: op_source_lapd_cpu_t, &
                           op_source_loc_dens_cpu_t, &
                           op_source_loc_heat_cpu_t, &
                           op_source_loc_torque_cpu_t
    use op_bnd_cond_m, only: op_bnd_cond_mms_cpu_t, op_bnd_cond_dir_cpu_t, &
                             op_bnd_cond_neum_cpu_t

    implicit none

contains

    module subroutine initialize_operators_grid_cpu(this)
        class(timestep_t), target, intent(inout) :: this

        character(len=:), allocatable :: coll_type, source_type, bnd_cond_type
        integer :: s

        allocate(op_diag_mom_0d_cpu_t :: this%op_mom_0d)
        allocate(op_diag_mom_2d_cpu_t :: this%op_mom_2d)
        allocate(op_rhs_vlasov_eq_static_cpu_t :: &
                 this%op_rhs_vlasov_eq_static)
        allocate(op_rhs_vlasov_eq_dynamic_cpu_t :: &
                 this%op_rhs_vlasov_eq_dynamic)
        allocate(op_mom_maxwells_eq_cpu_t :: this%op_mom_maxwells_eq)
        allocate(op_mom_ohms_law_cpu_t :: this%op_mom_ohms_law)

        select type(op => this%op_mom_0d)
            class is(op_diag_mom_0d_t)
                call op%initialize(this%dcomm_handler, this%mesh, &
                                   this%bsg_op)
        end select
        select type(op => this%op_mom_2d)
            class is(op_diag_mom_2d_t)
                call op%initialize(this%dcomm_handler, this%mesh, &
                                   this%bsg_op)
        end select

        select type(op => this%op_rhs_vlasov_eq_static)
            type is(op_rhs_vlasov_eq_static_cpu_t)
                call op%initialize(this%mesh, this%bsg_op)
        end select

        select type(op => this%op_rhs_vlasov_eq_dynamic)
            type is(op_rhs_vlasov_eq_dynamic_cpu_t)
                call op%initialize(this%mesh, this%bsg_op)
        end select

        select type(op => this%op_mom_maxwells_eq)
            type is(op_mom_maxwells_eq_cpu_t)
                call op%initialize(this%dcomm_handler, this%mesh, this%bsg_op)
        end select

        select type(op => this%op_mom_ohms_law)
            type is(op_mom_ohms_law_cpu_t)
                call op%initialize(this%dcomm_handler, this%mesh, this%bsg_op)
        end select

        ! Allocate collision operator
        coll_type = get_coll_type()

        select case(coll_type)
            case("none")
                this%apply_collisions = .false.

            case("bgk")
                allocate(op_coll_bgk_cpu_t :: this%op_coll)
                this%apply_collisions = .true.

            case("lbd")
                allocate(op_coll_lbd_cpu_t :: this%op_coll)
                this%apply_collisions = .true.

            case("lorentz")
                allocate(op_coll_lorentz_cpu_t :: this%op_coll)
                this%apply_collisions = .true.

            case("fpl")
                allocate(op_coll_fpl_cpu_t :: this%op_coll)
                this%apply_collisions = .true.

            case default
                call handle_error(&
                        "Selected collision type is not supported!", &
                        GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                        additional_info=&
                            error_info_t("Collision type was: "//coll_type))
        end select

        if(this%apply_collisions) then
            select type(op => this%op_coll)
                class is(op_coll_t)
                    call op%initialize(this%dcomm_handler, this%mesh)
            end select
            allocate(op_mom_coll_cpu_t :: this%op_mom_coll)
            call this%op_mom_coll%initialize(this%dcomm_handler, this%mesh)
        end if

        ! Allocate source container
        this%apply_sources = .true.
        this%n_source = get_n_source()
        if(this%n_source == 0) then
            this%apply_sources = .false.
        end if
        if(this%apply_sources) then
            allocate(this%op_source_container(this%n_source))
            do s = 1, this%n_source
                source_type = get_source_type(s)
                select case(source_type)
                    case("lapd")
                        allocate(op_source_lapd_cpu_t :: &
                                 this%op_source_container(s)%op_source)
                    case("dens")
                        allocate(op_source_loc_dens_cpu_t :: &
                                 this%op_source_container(s)%op_source)
                    case("torque")
                        allocate(op_source_loc_torque_cpu_t :: &
                                 this%op_source_container(s)%op_source)
                    case("heat")
                        allocate(op_source_loc_heat_cpu_t :: &
                                 this%op_source_container(s)%op_source)
                end select
                call this%op_source_container(s)&
                         %op_source%initialize(this%mesh)
            end do
        end if

        ! Allocate boundary condition operator
        bnd_cond_type = get_bnd_cond_type()
        if(this%run_mms) then
            allocate(op_bnd_cond_mms_cpu_t :: this%op_bnd_cond)
        else
            select case(bnd_cond_type)
                case('dirichlet')
                    allocate(op_bnd_cond_dir_cpu_t :: this%op_bnd_cond)
                case('neumann')
                    allocate(op_bnd_cond_neum_cpu_t :: this%op_bnd_cond)
                case default
                    call handle_error(&
                        "Selected boundary condition type &
                        &is not supported!", &
                        GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                        additional_info=&
                        error_info_t("Boundary condition type was: "&
                                     //bnd_cond_type))
            end select
        end if
        call this%op_bnd_cond%initialize(this%dcomm_handler, this%mesh)
    end subroutine

end submodule
