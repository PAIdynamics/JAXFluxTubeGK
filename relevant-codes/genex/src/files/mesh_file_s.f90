submodule (diag_files_m) mesh_file_s

    implicit none

contains

    module subroutine initialize_mesh(this, dcomm_handler, filename, &
                                      n_points_phi, is_axisymmetric, read_mesh)
        class(mesh_file_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n_points_phi
        logical, intent(in) :: is_axisymmetric
        logical, optional, intent(in) :: read_mesh

        logical :: read_mesh_local
        integer :: ierr, file_id, id_mgr_grp, id_map_grp, id_ubound
        integer :: k
        character(len=50) :: group_name

        read_mesh_local = .false.
        if(present(read_mesh)) read_mesh_local = read_mesh

        this%dcomm_handler => dcomm_handler
        this%filename = filename

        ! Allocate storage for multigrid and map matrix group IDs. For
        ! axisymmetric equilibria, groups only exist for the first plane
        if(is_axisymmetric) then
            id_ubound = 1
        else
            id_ubound = n_points_phi
        endif

        allocate(this%id_multigrid(id_ubound))
        allocate(this%id_map_positive1, &
                 this%id_map_positive2, &
                 this%id_map_negative1, &
                 this%id_map_negative2, &
                 mold=this%id_multigrid)

        if(read_mesh_local) then
            call prepare_read()
        else
            call prepare_write()
        endif

    contains

        subroutine prepare_read()
            !! Opens mesh file and stores multigrid and map matrix group IDs for
            !! use in PARALLAX read routines

            ! Open file
            ierr = nf90_open(this%filename, NF90_NOWRITE, file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            this%file_id = file_id

            ! Get multigrid and map matrix group IDs
            ierr = nf90_inq_grp_ncid(file_id, "multigrid", id_mgr_grp)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_inq_grp_ncid(file_id, "map_matrices", id_map_grp)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            do k = 1, id_ubound
                write(group_name, "(A, I3.3)") "multigrid_plane", k
                ierr = nf90_inq_grp_ncid(id_mgr_grp, group_name, &
                                         this%id_multigrid(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_positive1_plane", k
                ierr = nf90_inq_grp_ncid(id_map_grp, group_name, &
                                         this%id_map_positive1(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_positive2_plane", k
                ierr = nf90_inq_grp_ncid(id_map_grp, group_name, &
                                         this%id_map_positive2(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_negative1_plane", k
                ierr = nf90_inq_grp_ncid(id_map_grp, group_name, &
                                         this%id_map_negative1(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_negative2_plane", k
                ierr = nf90_inq_grp_ncid(id_map_grp, group_name, &
                                         this%id_map_negative2(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
            enddo

        end subroutine

        subroutine prepare_write()
            !! Creates mesh file, creates multigrid and map matrix groups, and
            !! stores their group IDs for use in PARALLAX write routines

            ! Only master writes the file
            if(.not. this%dcomm_handler%is_master()) return

            ! Create file
            ierr = nf90_create(this%filename, NF90_NETCDF4, file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            this%file_id = file_id

            ! Create multigrid and map matrix groups
            ierr = nf90_def_grp(file_id, "multigrid", id_mgr_grp)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ierr = nf90_def_grp(file_id, "map_matrices", id_map_grp)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! Create multigrid and map matrix subgroups. For axisymmetric
            ! equilibria, only create a group for the first plane
            do k = 1, id_ubound
                write(group_name, "(A, I3.3)") "multigrid_plane", k
                ierr = nf90_def_grp(id_mgr_grp, group_name, &
                                    this%id_multigrid(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_positive1_plane", k
                ierr = nf90_def_grp(id_map_grp, group_name, &
                                    this%id_map_positive1(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_positive2_plane", k
                ierr = nf90_def_grp(id_map_grp, group_name, &
                                    this%id_map_positive2(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_negative1_plane", k
                ierr = nf90_def_grp(id_map_grp, group_name, &
                                    this%id_map_negative1(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)

                write(group_name, "(A, I3.3)") "map_negative2_plane", k
                ierr = nf90_def_grp(id_map_grp, group_name, &
                                    this%id_map_negative2(k))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
            enddo

        end subroutine

    end subroutine

    module subroutine write_mesh(this, equi_type, size_RZ_max, not_ghost, &
            not_filler, buf_zone, R, Z, RZw, x_ind, z_ind, RZ_ind, jacobian, &
            phi, mu, vp)
        class(mesh_file_t), intent(inout) :: this
        integer, intent(in) :: equi_type
        integer, intent(in) :: size_RZ_max
        real(kind=GP), intent(in), dimension(:,:) :: not_ghost
        real(kind=GP), intent(in), dimension(:,:) :: not_filler
        integer, intent(in), dimension(:,:) :: buf_zone
        real(kind=GP), intent(in), dimension(:,:) :: R
        real(kind=GP), intent(in), dimension(:,:) :: Z
        real(kind=GP), intent(in), dimension(:,:) :: RZw
        integer, dimension(:,:), intent(in) :: x_ind
        integer, dimension(:,:), intent(in) :: z_ind
        integer, dimension(:,:), intent(in) :: RZ_ind
        real(kind=GP), intent(in), dimension(:,:) :: jacobian
        real(kind=GP), dimension(:), intent(in) :: phi
        real(kind=GP), dimension(:), intent(in) :: mu
        real(kind=GP), dimension(:), intent(in) :: vp

        integer, dimension(2) :: shp
        integer :: ierr, file_id
        integer :: id_RZ_grp, id_phi_grp, id_vp_grp, id_mu_grp, &
            id_dim_RZ, id_dim_phi, id_dim_RZphi(2), &
            id_dim_mu_grid, id_dim_vp_grid, &
            id_not_ghost, id_not_filler, id_buf_zone, &
            id_R, id_Z, id_RZw, id_x_ind, id_z_ind, id_RZ_ind, id_jacobian, &
            id_phi, id_vp, id_mu

        ! Only master writes the file
        if(.not. this%dcomm_handler%is_master()) return

        file_id = this%file_id

        ! Check array shapes
        shp = [size_RZ_max, size(phi)]
        call this%check_array(shp, shape(not_ghost), __LINE__, __FILE__)
        call this%check_array(shp, shape(not_filler), __LINE__, __FILE__)
        call this%check_array(shp, shape(buf_zone), __LINE__, __FILE__)
        call this%check_array(shp, shape(R), __LINE__, __FILE__)
        call this%check_array(shp, shape(Z), __LINE__, __FILE__)
        call this%check_array(shp, shape(RZw), __LINE__, __FILE__)
        call this%check_array(shp, shape(x_ind), __LINE__, __FILE__)
        call this%check_array(shp, shape(z_ind), __LINE__, __FILE__)
        call this%check_array(shp, shape(RZ_ind), __LINE__, __FILE__)
        call this%check_array(shp, shape(jacobian), __LINE__, __FILE__)
        call this%check_array([shp(2)], shape(phi), __LINE__, __FILE__)

        ! Global attributes
        ierr = nf90_put_att(file_id, NF90_GLOBAL,'equi_type', equi_type)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_att(file_id, NF90_GLOBAL,'size_RZ_max', size_RZ_max)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Start definitions

        ! Groups
        ierr = nf90_def_grp(file_id, "RZ_grid", id_RZ_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_grp(file_id, "vp_grid", id_vp_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_grp(file_id, "mu_grid", id_mu_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_grp(file_id, "phi_grid", id_phi_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Dimensions
        ierr = nf90_def_dim(file_id, 'dim_RZ', shp(1), id_dim_RZ)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_dim(file_id, 'dim_phi', shp(2), id_dim_phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        id_dim_RZphi = [id_dim_RZ, id_dim_phi]
        ! Store RZ and phi dimension IDs for use in other write subroutines
        this%id_dim_RZphi = id_dim_RZphi

        ierr = nf90_def_dim(id_vp_grp, 'dim_vp_grid', size(vp), id_dim_vp_grid)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_dim(id_mu_grp, 'dim_mu_grid', size(mu), id_dim_mu_grid)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! RZ grid
        ierr = nf90_def_var(id_RZ_grp, 'not_ghost', NF90_GP, id_dim_RZphi, &
                            id_not_ghost)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'not_filler', NF90_GP, id_dim_RZphi, &
                            id_not_filler)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'buf_zone', NF90_INT, id_dim_RZphi, &
                            id_buf_zone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'R', NF90_GP, id_dim_RZphi, id_R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'Z', NF90_GP, id_dim_RZphi, id_Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'RZw', NF90_GP, id_dim_RZphi, id_RZw)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'x_ind', NF90_INT, id_dim_RZphi, &
                            id_x_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'z_ind', NF90_INT, id_dim_RZphi, &
                            id_z_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'RZ_ind', NF90_INT, id_dim_RZphi, &
                            id_RZ_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_RZ_grp, 'jacobian', NF90_GP, id_dim_RZphi, &
                            id_jacobian)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! phi grid
        ierr = nf90_def_var(id_phi_grp, 'phi', NF90_GP, id_dim_RZphi(2), id_phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! vp grid
        ierr = nf90_def_var(id_vp_grp, 'vp', NF90_GP, id_dim_vp_grid, id_vp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! mu grid
        ierr = nf90_def_var(id_mu_grp, 'mu', NF90_GP, id_dim_mu_grid, id_mu)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_enddef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End definitions

        ! Start put

        ! RZ grid
        ierr = nf90_put_var(id_RZ_grp, id_not_ghost, not_ghost)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_not_filler, not_filler)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_buf_zone, buf_zone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_R, R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_Z, Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_RZw, RZw)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_x_ind, x_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_z_ind, z_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_RZ_ind, RZ_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_RZ_grp, id_jacobian, jacobian)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! phi grid
        ierr = nf90_put_var(id_phi_grp, id_phi, phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! vp grid
        ierr = nf90_put_var(id_vp_grp, id_vp, vp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! mu grid
        ierr = nf90_put_var(id_mu_grp, id_mu, mu)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End put

        ! Redef for next write routine
        ierr = nf90_redef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine write_polar_mesh(this, polar_theta, polar_rho, phi)
        class(mesh_file_t), intent(inout) :: this
        real(kind=GP), dimension(:,:), intent(in) :: polar_theta
        real(kind=GP), dimension(:,:), intent(in) :: polar_rho
        real(kind=GP), dimension(:),   intent(in) :: phi

        integer, dimension(2) :: shp_theta, shp_rho
        integer :: ierr, file_id
        integer :: id_pg_grp, &
                   id_dim_TRphi(3), id_dim_Tphi(2), id_dim_Rphi(2), &
                   id_dim_polar_theta, id_dim_polar_rho, id_dim_phi, &
                   id_polar_theta, id_polar_rho

        ! Only master writes the file
        if(.not. this%dcomm_handler%is_master()) return

        file_id = this%file_id

        shp_theta = [size(polar_theta, 1), size(phi)]
        shp_rho   = [size(polar_rho,   1), size(phi)]

        ! Check array shaped
        call this%check_array(shp_theta, shape(polar_theta), __LINE__, __FILE__)
        call this%check_array(shp_rho  , shape(polar_rho  ), __LINE__, __FILE__)

        ! Start definitions

        ! Groups
        ierr = nf90_def_grp(file_id, "polar_grid", id_pg_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Dimensions
        ierr = nf90_def_dim(file_id, "dim_polar_theta", shp_theta(1), &
                            id_dim_polar_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_dim(file_id, "dim_polar_rho",   shp_rho(1),   &
                            id_dim_polar_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        id_dim_phi = this%id_dim_RZphi(2)

        id_dim_TRphi = [id_dim_polar_theta, id_dim_polar_rho, id_dim_phi]
        ! Store theta, rho and phi dimension IDs for use in other subroutines
        this%id_dim_TRphi = id_dim_TRphi

        id_dim_Tphi = [id_dim_polar_theta, id_dim_phi]
        id_dim_Rphi = [id_dim_polar_rho,   id_dim_phi]

        ! polar mesh
        ierr = nf90_def_var(id_pg_grp, "polar_theta", NF90_GP, &
                            id_dim_Tphi, id_polar_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_pg_grp, "polar_rho",   NF90_GP, &
                            id_dim_Rphi, id_polar_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_enddef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End definitions

        ! Start put

        ierr = nf90_put_var(id_pg_grp, id_polar_theta, polar_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_pg_grp, id_polar_rho,   polar_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End put

        ! Redef for next write routine
        ierr = nf90_redef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine write_magnetic_field(this, &
            psi_norm_fac, psi_lo, psi_up, absB_max, &
            absB, normb_R, normb_Z, normb_tor, curl_normb_y, &
            dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, dgyzdx_over_g, &
            inv_g, dabsBdx, dabsBdz, dabsBdy, rho, theta, psi)
        class(mesh_file_t), intent(in) :: this
        real(kind=GP), intent(in) :: psi_norm_fac
        real(kind=GP), intent(in) :: psi_lo
        real(kind=GP), intent(in) :: psi_up
        real(kind=GP), intent(in) :: absB_max
        real(kind=GP), dimension(:,:), intent(in) :: absB
        real(kind=GP), dimension(:,:), intent(in) :: normb_R
        real(kind=GP), dimension(:,:), intent(in) :: normb_Z
        real(kind=GP), dimension(:,:), intent(in) :: normb_tor
        real(kind=GP), dimension(:,:), intent(in) :: curl_normb_y
        real(kind=GP), dimension(:,:), intent(in) :: dgyxdy_over_g
        real(kind=GP), dimension(:,:), intent(in) :: dgyzdy_over_g
        real(kind=GP), dimension(:,:), intent(in) :: dgyxdz_over_g
        real(kind=GP), dimension(:,:), intent(in) :: dgyzdx_over_g
        real(kind=GP), dimension(:,:), intent(in) :: inv_g
        real(kind=GP), dimension(:,:), intent(in) :: dabsBdx
        real(kind=GP), dimension(:,:), intent(in) :: dabsBdz
        real(kind=GP), dimension(:,:), intent(in) :: dabsBdy
        real(kind=GP), dimension(:,:), intent(in) :: rho
        real(kind=GP), dimension(:,:), intent(in) :: theta
        real(kind=GP), dimension(:,:), intent(in) :: psi

        integer :: ierr, file_id, id_mf_grp
        integer :: id_dim_RZphi(2), id_absB, id_normb_R, id_normb_Z, &
                   id_normb_tor, id_curl_normb_y, id_dgyxdy_over_g, &
                   id_dgyzdy_over_g, id_dgyxdz_over_g, id_dgyzdx_over_g, &
                   id_inv_g, id_dabsBdx, id_dabsBdz, id_dabsBdy, id_rho, &
                   id_theta, id_psi
        integer, dimension(2) :: shp

        ! Only master writes the file
        if(.not. this%dcomm_handler%is_master()) return

        ! File ID and dimensions were defined in the mesh write subroutine
        file_id = this%file_id
        id_dim_RZphi = this%id_dim_RZphi

        ! Check all arrays
        shp = shape(absB)
        call this%check_array(shp, shape(absB), __LINE__, __FILE__)
        call this%check_array(shp, shape(normb_R), __LINE__, __FILE__)
        call this%check_array(shp, shape(normb_Z), __LINE__, __FILE__)
        call this%check_array(shp, shape(normb_tor), __LINE__, __FILE__)
        call this%check_array(shp, shape(curl_normb_y), __LINE__, __FILE__)
        call this%check_array(shp, shape(dgyxdy_over_g), __LINE__, __FILE__)
        call this%check_array(shp, shape(dgyzdy_over_g), __LINE__, __FILE__)
        call this%check_array(shp, shape(dgyxdz_over_g), __LINE__, __FILE__)
        call this%check_array(shp, shape(dgyzdx_over_g), __LINE__, __FILE__)
        call this%check_array(shp, shape(inv_g), __LINE__, __FILE__)
        call this%check_array(shp, shape(dabsBdx), __LINE__, __FILE__)
        call this%check_array(shp, shape(dabsBdz), __LINE__, __FILE__)
        call this%check_array(shp, shape(dabsBdy), __LINE__, __FILE__)
        call this%check_array(shp, shape(rho), __LINE__, __FILE__)
        call this%check_array(shp, shape(theta), __LINE__, __FILE__)
        call this%check_array(shp, shape(psi), __LINE__, __FILE__)

        ! Start definitions

        ! Groups
        ierr = nf90_def_grp(file_id, "magnetic_field", id_mf_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Global attributes for magnetic_field group
        ierr = nf90_put_att(id_mf_grp, NF90_GLOBAL, 'psi_norm_fac', &
                            psi_norm_fac)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_att(id_mf_grp, NF90_GLOBAL, 'psi_lo', psi_lo)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_att(id_mf_grp, NF90_GLOBAL, 'psi_up', psi_up)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_att(id_mf_grp, NF90_GLOBAL, 'absB_max', absB_max)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! absB
        ierr = nf90_def_var(id_mf_grp, 'absB', NF90_GP, id_dim_RZphi, id_absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! normb_R
        ierr = nf90_def_var(id_mf_grp, 'normb_R', NF90_GP, id_dim_RZphi, &
                            id_normb_R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! normb_Z
        ierr = nf90_def_var(id_mf_grp, 'normb_Z', NF90_GP, id_dim_RZphi, &
                            id_normb_Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! normb_tor
        ierr = nf90_def_var(id_mf_grp, 'normb_tor', NF90_GP, id_dim_RZphi, &
                            id_normb_tor)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! curl_normb_y
        ierr = nf90_def_var(id_mf_grp, 'curl_normb_y', NF90_GP, id_dim_RZphi, &
                            id_curl_normb_y)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyxdy_over_g
        ierr = nf90_def_var(id_mf_grp, 'dgyxdy_over_g', NF90_GP, id_dim_RZphi, &
                            id_dgyxdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyzdy_over_g
        ierr = nf90_def_var(id_mf_grp, 'dgyzdy_over_g', NF90_GP, id_dim_RZphi, &
                            id_dgyzdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyxdz_over_g
        ierr = nf90_def_var(id_mf_grp, 'dgyxdz_over_g', NF90_GP, id_dim_RZphi, &
                            id_dgyxdz_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyzdx_over_g
        ierr = nf90_def_var(id_mf_grp, 'dgyzdx_over_g', NF90_GP, id_dim_RZphi, &
                            id_dgyzdx_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! inv_g
        ierr = nf90_def_var(id_mf_grp, 'inv_g', NF90_GP, id_dim_RZphi, id_inv_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dabsBdx
        ierr = nf90_def_var(id_mf_grp, 'dabsBdx', NF90_GP, id_dim_RZphi, &
                            id_dabsBdx)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dabsBdz
        ierr = nf90_def_var(id_mf_grp, 'dabsBdz', NF90_GP, id_dim_RZphi, &
                            id_dabsBdz)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dabsBdy
        ierr = nf90_def_var(id_mf_grp, 'dabsBdy', NF90_GP, id_dim_RZphi, &
                            id_dabsBdy)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! rho
        ierr = nf90_def_var(id_mf_grp, 'rho', NF90_GP, id_dim_RZphi, id_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! theta
        ierr = nf90_def_var(id_mf_grp, 'theta', NF90_GP, id_dim_RZphi, id_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! psi
        ierr = nf90_def_var(id_mf_grp, 'psi', NF90_GP, id_dim_RZphi, id_psi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_enddef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End definitions

        ! Start put

        ! absB
        ierr = nf90_put_var(id_mf_grp, id_absB, absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! normb_R
        ierr = nf90_put_var(id_mf_grp, id_normb_R, normb_R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! normb_Z
        ierr = nf90_put_var(id_mf_grp, id_normb_Z, normb_Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! normb_tor
        ierr = nf90_put_var(id_mf_grp, id_normb_tor, normb_tor)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! curl_normb_y
        ierr = nf90_put_var(id_mf_grp, id_curl_normb_y, curl_normb_y)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyxdy_over_g
        ierr = nf90_put_var(id_mf_grp, id_dgyxdy_over_g, dgyxdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyzdy_over_g
        ierr = nf90_put_var(id_mf_grp, id_dgyzdy_over_g, dgyzdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyxdz_over_g
        ierr = nf90_put_var(id_mf_grp, id_dgyxdz_over_g, dgyxdz_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dgyzdx_over_g
        ierr = nf90_put_var(id_mf_grp, id_dgyzdx_over_g, dgyzdx_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! inv_g
        ierr = nf90_put_var(id_mf_grp, id_inv_g, inv_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dabsBdx
        ierr = nf90_put_var(id_mf_grp, id_dabsBdx, dabsBdx)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dabsBdz
        ierr = nf90_put_var(id_mf_grp, id_dabsBdz, dabsBdz)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! dabsBdy
        ierr = nf90_put_var(id_mf_grp, id_dabsBdy, dabsBdy)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! rho
        ierr = nf90_put_var(id_mf_grp, id_rho, rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! theta
        ierr = nf90_put_var(id_mf_grp, id_theta, theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! psi
        ierr = nf90_put_var(id_mf_grp, id_psi, psi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End put

        ! Redef for next write routine
        ierr = nf90_redef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine write_polar_magnetic_field(this, polar_absB, loss_cone)
        class(mesh_file_t), intent(inout) :: this
        real(kind=GP), dimension(:,:,:), intent(in) :: polar_absB
        real(kind=GP), dimension(:,:),   intent(in) :: loss_cone

        integer :: ierr, file_id, id_mfp_grp
        integer :: id_dim_TRphi(3), id_dim_RZphi(2), id_polar_absB, id_loss_cone

        ! Only master writes the file
        if(.not. this%dcomm_handler%is_master()) return

        ! File ID was defined in the mesh write subroutine
        file_id = this%file_id

        id_dim_TRphi = this%id_dim_TRphi
        id_dim_RZphi = this%id_dim_RZphi

        ! Start definitions

        ! Groups
        ierr = nf90_def_grp(file_id, "magnetic_field_polar", id_mfp_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! absB
        ierr = nf90_def_var(id_mfp_grp, "polar_absB", NF90_GP, &
                            id_dim_TRphi, id_polar_absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_def_var(id_mfp_grp, "loss_cone", NF90_GP, &
                            id_dim_RZphi, id_loss_cone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_enddef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End definitions

        ! Start put

        ! absB
        ierr = nf90_put_var(id_mfp_grp, id_polar_absB, polar_absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_put_var(id_mfp_grp, id_loss_cone, loss_cone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End put

        ! Redef for next write routine
        ierr = nf90_redef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine write_parcon(this, not_in_target, maps_on_mesh, &
                                   fll_positive1, fll_positive2, &
                                   fll_negative1, fll_negative2, &
                                   parcon_positive1, parcon_positive2, &
                                   parcon_negative1, parcon_negative2)
        class(mesh_file_t), intent(in) :: this
        integer, intent(in), dimension(:,:) :: not_in_target
        real(kind=GP), intent(in), dimension(:,:) :: maps_on_mesh
        real(kind=GP), intent(in), dimension(:,:) :: fll_positive1
        real(kind=GP), intent(in), dimension(:,:) :: fll_positive2
        real(kind=GP), intent(in), dimension(:,:) :: fll_negative1
        real(kind=GP), intent(in), dimension(:,:) :: fll_negative2
        integer, intent(in), dimension(:,:) :: parcon_positive1
        integer, intent(in), dimension(:,:) :: parcon_positive2
        integer, intent(in), dimension(:,:) :: parcon_negative1
        integer, intent(in), dimension(:,:) :: parcon_negative2

        integer :: ierr, file_id, id_pc_grp, id_dim_RZphi(2), shp(2), &
                   id_not_in_target, id_maps_on_mesh, &
                   id_fll_pos1, id_fll_neg1, id_fll_pos2, id_fll_neg2, &
                   id_pc_pos1, id_pc_neg1, id_pc_pos2, id_pc_neg2

        ! Only master writes the file
        if(.not. this%dcomm_handler%is_master()) return

        ! File ID and dimensions were defined in the mesh write subroutine
        file_id = this%file_id
        id_dim_RZphi = this%id_dim_RZphi

        ! Check dimensions of the input arrays
        shp = shape(not_in_target)
        call this%check_array(shp, shape(not_in_target), __LINE__, __FILE__)
        call this%check_array(shp, shape(maps_on_mesh), __LINE__, __FILE__)
        call this%check_array(shp, shape(fll_positive1), __LINE__, __FILE__)
        call this%check_array(shp, shape(fll_positive2), __LINE__, __FILE__)
        call this%check_array(shp, shape(fll_negative1), __LINE__, __FILE__)
        call this%check_array(shp, shape(fll_negative2), __LINE__, __FILE__)
        call this%check_array(shp, shape(parcon_positive1), __LINE__, __FILE__)
        call this%check_array(shp, shape(parcon_positive2), __LINE__, __FILE__)
        call this%check_array(shp, shape(parcon_negative1), __LINE__, __FILE__)
        call this%check_array(shp, shape(parcon_negative2), __LINE__, __FILE__)

        ! Start definitions

        ! Groups
        ierr = nf90_def_grp(file_id, 'parcon', id_pc_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! not_in_target
        ierr = nf90_def_var(id_pc_grp, 'not_in_target', NF90_GP, &
                            id_dim_RZphi, id_not_in_target)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! maps_on_mesh
        ierr = nf90_def_var(id_pc_grp, 'maps_on_mesh', NF90_GP, &
                            id_dim_RZphi, id_maps_on_mesh)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_positive1
        ierr = nf90_def_var(id_pc_grp, 'fll_positive1', NF90_GP, &
                            id_dim_RZphi, id_fll_pos1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_positive2
        ierr = nf90_def_var(id_pc_grp, 'fll_positive2', NF90_GP, &
                            id_dim_RZphi, id_fll_pos2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_negative1
        ierr = nf90_def_var(id_pc_grp, 'fll_negative1', NF90_GP, &
                            id_dim_RZphi, id_fll_neg1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_negative2
        ierr = nf90_def_var(id_pc_grp, 'fll_negative2', NF90_GP, &
                            id_dim_RZphi, id_fll_neg2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_positive1
        ierr = nf90_def_var(id_pc_grp, 'parcon_positive1', NF90_INT, &
                            id_dim_RZphi, id_pc_pos1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_positive2
        ierr = nf90_def_var(id_pc_grp, 'parcon_positive2', NF90_INT, &
                            id_dim_RZphi, id_pc_pos2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_negative1
        ierr = nf90_def_var(id_pc_grp, 'parcon_negative1', NF90_INT, &
                            id_dim_RZphi, id_pc_neg1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_negative2
        ierr = nf90_def_var(id_pc_grp, 'parcon_negative2', NF90_INT, &
                            id_dim_RZphi, id_pc_neg2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_enddef(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End definitions

        ! Start put

        ! not_in_target
        ierr = nf90_put_var(id_pc_grp, id_not_in_target, not_in_target)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! maps_on_mesh
        ierr = nf90_put_var(id_pc_grp, id_maps_on_mesh, maps_on_mesh)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_positive1
        ierr = nf90_put_var(id_pc_grp, id_fll_pos1, fll_positive1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_positive2
        ierr = nf90_put_var(id_pc_grp, id_fll_pos2, fll_positive2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_negative1
        ierr = nf90_put_var(id_pc_grp, id_fll_neg1, fll_negative1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! fll_negative2
        ierr = nf90_put_var(id_pc_grp, id_fll_neg2, fll_negative2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_positive1
        ierr = nf90_put_var(id_pc_grp, id_pc_pos1, parcon_positive1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_positive2
        ierr = nf90_put_var(id_pc_grp, id_pc_pos2, parcon_positive2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_negative1
        ierr = nf90_put_var(id_pc_grp, id_pc_neg1, parcon_negative1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! parcon_negative2
        ierr = nf90_put_var(id_pc_grp, id_pc_neg2, parcon_negative2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! End put

        ! Close file
        ierr = nf90_close(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

    end subroutine

    module subroutine read_dimensions(this, n_points_RZ, n_points_phi)
        class(mesh_file_t), intent(inout) :: this
        integer, intent(out) :: n_points_RZ
        integer, intent(out) :: n_points_phi

        integer :: ierr, file_id, id_dim_RZ, id_dim_phi

        file_id = this%file_id

        ierr = nf90_inq_dimid(file_id, 'dim_RZ', id_dim_RZ)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_inquire_dimension(file_id, id_dim_RZ, len=n_points_RZ)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_dimid(file_id, 'dim_phi', id_dim_phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_inquire_dimension(file_id, id_dim_phi, len=n_points_phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

    end subroutine

    module subroutine read_polar_dimensions(this, polar_n_theta, polar_n_rho)
        class(mesh_file_t), intent(inout) :: this
        integer, intent(out) :: polar_n_theta
        integer, intent(out) :: polar_n_rho

        integer :: ierr, file_id, id_dim_polar_theta, id_dim_polar_rho

        file_id = this%file_id

        ierr = nf90_inq_dimid(file_id, "dim_polar_theta", id_dim_polar_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_inquire_dimension(file_id, id_dim_polar_theta, &
                                      len=polar_n_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_dimid(file_id, "dim_polar_rho", id_dim_polar_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_inquire_dimension(file_id, id_dim_polar_rho, &
                                      len=polar_n_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

    end subroutine

    module subroutine read_mesh(this, equi_type, size_RZ_max, not_ghost, &
                                not_filler, buf_zone, R, Z, RZw, RZ_ind, &
                                jacobian, phi)
        class(mesh_file_t), intent(in) :: this
        integer, intent(out) :: equi_type
        integer, intent(out) :: size_RZ_max
        real(kind=GP), dimension(:,:), intent(out) :: not_ghost
        real(kind=GP), dimension(:,:), intent(out) :: not_filler
        integer, dimension(:,:), intent(out) :: buf_zone
        real(kind=GP), dimension(:,:), intent(out) :: R
        real(kind=GP), dimension(:,:), intent(out) :: Z
        real(kind=GP), dimension(:,:), intent(out) :: RZw
        integer, dimension(:,:), intent(out) :: RZ_ind
        real(kind=GP), dimension(:,:), intent(out) :: jacobian
        real(kind=GP), dimension(:), intent(out) :: phi

        integer :: ierr, file_id, id_RZ_grp, id_phi_grp, &
                   id_not_ghost, id_not_filler, id_buf_zone, &
                   id_R, id_Z, id_RZw, id_RZ_ind, &
                   id_jacobian, id_phi

        file_id = this%file_id

        ! Get global attributes
        ierr = nf90_get_att(file_id, NF90_GLOBAL, 'equi_type', equi_type)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_get_att(file_id, NF90_GLOBAL, 'size_RZ_max', size_RZ_max)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get group IDs
        ierr = nf90_inq_grp_ncid(file_id, 'RZ_grid', id_RZ_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_grp_ncid(file_id, 'phi_grid', id_phi_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get variables
        ierr = nf90_inq_varid(id_RZ_grp, 'not_ghost', id_not_ghost)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_not_ghost, not_ghost)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'not_filler', id_not_filler)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_not_filler, not_filler)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'buf_zone', id_buf_zone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_buf_zone, buf_zone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'R', id_R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_R, R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'Z', id_Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_Z, Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'RZw', id_RZw)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_RZw, RZw)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'RZ_ind', id_RZ_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_RZ_ind, RZ_ind)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_RZ_grp, 'jacobian', id_jacobian)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_RZ_grp, id_jacobian, jacobian)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_phi_grp, 'phi', id_phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_phi_grp, id_phi, phi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine read_polar_mesh(this, polar_theta, polar_rho)
        class(mesh_file_t), intent(in) :: this
        real(kind=GP), dimension(:,:), intent(out) :: polar_theta
        real(kind=GP), dimension(:,:), intent(out) :: polar_rho

        integer :: ierr, file_id, id_pg_grp, &
                   id_polar_theta, id_polar_rho

        file_id = this%file_id

        ! Get group IDs
        ierr = nf90_inq_grp_ncid(file_id, "polar_grid", id_pg_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get variables
        ierr = nf90_inq_varid(id_pg_grp, "polar_theta", id_polar_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pg_grp, id_polar_theta, polar_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pg_grp, "polar_rho", id_polar_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pg_grp, id_polar_rho, polar_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine read_magnetic_field(this, psi_norm_fac, psi_lo, &
        psi_up, absB_max, absB, normb_R, normb_Z, normb_tor, curl_normb_y, &
        dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, dgyzdx_over_g, &
        inv_g, dabsBdx, dabsBdz, dabsBdy, rho, theta, psi)
        class(mesh_file_t), intent(in) :: this
        real(kind=GP), intent(out) :: psi_norm_fac
        real(kind=GP), intent(out) :: psi_lo
        real(kind=GP), intent(out) :: psi_up
        real(kind=GP), intent(out) :: absB_max
        real(kind=GP), dimension(:,:), intent(out) :: absB
        real(kind=GP), dimension(:,:), intent(out) :: normb_R
        real(kind=GP), dimension(:,:), intent(out) :: normb_Z
        real(kind=GP), dimension(:,:), intent(out) :: normb_tor
        real(kind=GP), dimension(:,:), intent(out) :: curl_normb_y
        real(kind=GP), dimension(:,:), intent(out) :: dgyxdy_over_g
        real(kind=GP), dimension(:,:), intent(out) :: dgyzdy_over_g
        real(kind=GP), dimension(:,:), intent(out) :: dgyxdz_over_g
        real(kind=GP), dimension(:,:), intent(out) :: dgyzdx_over_g
        real(kind=GP), dimension(:,:), intent(out) :: inv_g
        real(kind=GP), dimension(:,:), intent(out) :: dabsBdx
        real(kind=GP), dimension(:,:), intent(out) :: dabsBdz
        real(kind=GP), dimension(:,:), intent(out) :: dabsBdy
        real(kind=GP), dimension(:,:), intent(out) :: rho
        real(kind=GP), dimension(:,:), intent(out) :: theta
        real(kind=GP), dimension(:,:), intent(out) :: psi

        integer :: ierr, file_id, id_mf_grp, id_absB, id_normb_R, id_normb_Z, &
            id_normb_tor, id_curl_normb_y, id_dgyxdy_over_g, id_dgyzdy_over_g, &
            id_dgyxdz_over_g, id_dgyzdx_over_g, id_inv_g, id_dabsBdx, &
            id_dabsBdz, id_dabsBdy, id_rho, id_theta, id_psi

        file_id = this%file_id

        ! Get group ID
        ierr = nf90_inq_grp_ncid(file_id, 'magnetic_field', id_mf_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get global attributes
        ierr = nf90_get_att(id_mf_grp, NF90_GLOBAL, 'psi_norm_fac', &
                            psi_norm_fac)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_get_att(id_mf_grp, NF90_GLOBAL, 'psi_lo', psi_lo)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_get_att(id_mf_grp, NF90_GLOBAL, 'psi_up', psi_up)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_get_att(id_mf_grp, NF90_GLOBAL, 'absB_max', absB_max)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get variables
        ierr = nf90_inq_varid(id_mf_grp, 'absB', id_absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_absB, absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'normb_R', id_normb_R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_normb_R, normb_R)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'normb_Z', id_normb_Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_normb_Z, normb_Z)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'normb_tor', id_normb_tor)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_normb_tor, normb_tor)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'curl_normb_y', id_curl_normb_y)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_curl_normb_y, curl_normb_y)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'dgyxdy_over_g', id_dgyxdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dgyxdy_over_g, dgyxdy_over_g)

        ierr = nf90_inq_varid(id_mf_grp, 'dgyzdy_over_g', id_dgyzdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dgyzdy_over_g, dgyzdy_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'dgyxdz_over_g', id_dgyxdz_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dgyxdz_over_g, dgyxdz_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'dgyzdx_over_g', id_dgyzdx_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dgyzdx_over_g, dgyzdx_over_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'inv_g', id_inv_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_inv_g, inv_g)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'dabsBdx', id_dabsBdx)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dabsBdx, dabsBdx)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'dabsBdz', id_dabsBdz)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dabsBdz, dabsBdz)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'dabsBdy', id_dabsBdy)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_dabsBdy, dabsBdy)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'rho', id_rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_rho, rho)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'theta', id_theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_theta, theta)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, 'psi', id_psi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_psi, psi)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine read_polar_magnetic_field(this, polar_absB, loss_cone)
        class(mesh_file_t), intent(in) :: this
        real(kind=GP), dimension(:,:,:), intent(out) :: polar_absB
        real(kind=GP), dimension(:,:),   intent(out) :: loss_cone

        integer :: ierr, file_id, id_mf_grp, id_polar_absB, id_loss_cone

        file_id = this%file_id

        ! Get group ID
        ierr = nf90_inq_grp_ncid(file_id, "magnetic_field_polar", id_mf_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get variables
        ierr = nf90_inq_varid(id_mf_grp, "polar_absB", id_polar_absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_polar_absB, polar_absB)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_mf_grp, "loss_cone", id_loss_cone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_mf_grp, id_loss_cone, loss_cone)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

    module subroutine read_parcon(this, not_in_target, maps_on_mesh, &
                                  fll_positive1, fll_positive2, &
                                  fll_negative1, fll_negative2, &
                                  parcon_positive1, parcon_positive2, &
                                  parcon_negative1, parcon_negative2)
        class(mesh_file_t), intent(in) :: this
        integer, intent(out), dimension(:,:) :: not_in_target
        real(kind=GP), intent(out), dimension(:,:) :: maps_on_mesh
        real(kind=GP), intent(out), dimension(:,:) :: fll_positive1
        real(kind=GP), intent(out), dimension(:,:) :: fll_positive2
        real(kind=GP), intent(out), dimension(:,:) :: fll_negative1
        real(kind=GP), intent(out), dimension(:,:) :: fll_negative2
        integer, intent(out), dimension(:,:) :: parcon_positive1
        integer, intent(out), dimension(:,:) :: parcon_positive2
        integer, intent(out), dimension(:,:) :: parcon_negative1
        integer, intent(out), dimension(:,:) :: parcon_negative2

        integer :: ierr, file_id, id_pc_grp, id_not_in_target, &
                   id_maps_on_mesh, &
                   id_fll_pos1, id_fll_pos2, id_fll_neg1, id_fll_neg2, &
                   id_pc_pos1, id_pc_pos2, id_pc_neg1, id_pc_neg2

        file_id = this%file_id

        ! Get group ID
        ierr = nf90_inq_grp_ncid(file_id, 'parcon', id_pc_grp)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Get variables
        ierr = nf90_inq_varid(id_pc_grp, 'not_in_target', id_not_in_target)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_not_in_target, not_in_target)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'maps_on_mesh', id_maps_on_mesh)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_maps_on_mesh, maps_on_mesh)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'fll_positive1', id_fll_pos1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_fll_pos1, fll_positive1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'fll_positive2', id_fll_pos2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_fll_pos2, fll_positive2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'fll_negative1', id_fll_neg1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_fll_neg1, fll_negative1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'fll_negative2', id_fll_neg2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_fll_neg2, fll_negative2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'parcon_positive1', id_pc_pos1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_pc_pos1, parcon_positive1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'parcon_positive2', id_pc_pos2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_pc_pos2, parcon_positive2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'parcon_negative1', id_pc_neg1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_pc_neg1, parcon_negative1)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ierr = nf90_inq_varid(id_pc_grp, 'parcon_negative2', id_pc_neg2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ierr = nf90_get_var(id_pc_grp, id_pc_neg2, parcon_negative2)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! Close file
        ierr = nf90_close(file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
    end subroutine

end submodule
