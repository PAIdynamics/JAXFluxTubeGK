submodule (test_params_m) test_params_dist_initial_s
    ! Submodule that contains helpers for unit tests to initialize mesh
    ! with non-default parameters.
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_dist_initial_maxw_vspec(comm, rank, &
                                                         n, params)
        !! Initializes initial Maxwellian distribution function
        !! with the given test parameters using the spectral approach.
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, intent(in) :: n
        !! Species index
        type(params_dist_initial_maxw_vspec_t), intent(in) :: params
        !! Parameter type

        type(params_dist_initial_maxw_vspec_t) :: params_local
        character(:), allocatable :: species_file
        integer :: iunit, io_error, ierr

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_dist_initial_maxw_vspec")
            call write_nml(iunit, "params_dist_initial_maxw_vspec", &
                           "drift_vpar", &
                           params_local%drift_vpar)
            call close_nml(iunit, "params_dist_initial_maxw_vspec")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_dist_initial_maxw_vspec(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

    module subroutine setup_test_dist_initial_bi_maxw(comm, rank, n, params)
        !! Initializes initial bi Maxwellian distribution function
        !! with the given test parameters.
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, intent(in) :: n
        !! Species index
        type(params_dist_initial_bi_maxw_t), intent(in) :: params
        !! Parameter type

        type(params_dist_initial_bi_maxw_t) :: params_local
        character(:), allocatable :: species_file
        integer :: iunit, io_error, ierr

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_dist_initial_bi_maxw")
            call write_nml(iunit, "params_dist_initial_bi_maxw", &
                           "drift_vpar", &
                            params_local%drift_vpar)
            call write_nml(iunit, "params_dist_initial_bi_maxw", &
                           "tau_par", &
                            params_local%tau_par)
            call write_nml(iunit, "params_dist_initial_bi_maxw", &
                           "tau_perp", &
                            params_local%tau_perp)
            call close_nml(iunit, "params_dist_initial_bi_maxw")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_dist_initial_bi_maxw(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

    module subroutine setup_test_dist_initial_double_maxw(comm, rank, n, params)
        !! Initializes initial double Maxwellian distribution function
        !! with the given test parameters.
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, intent(in) :: n
        !! Species index
        type(params_dist_initial_double_maxw_t), intent(in) :: params

        Type(params_dist_initial_double_maxw_t) :: params_local

        character(:), allocatable :: species_file
        integer :: iunit, io_error, ierr

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_dist_initial_double_maxw")
            call write_nml(iunit, "params_dist_initial_double_maxw", &
                           "drift_par1", params_local%drift_par1)
            call write_nml(iunit, "params_dist_initial_double_maxw", &
                           "drift_par2", params_local%drift_par2)
            call write_nml(iunit, "params_dist_initial_double_maxw", &
                           "amp1", params_local%amp1)
            call write_nml(iunit, "params_dist_initial_double_maxw", &
                           "amp2", params_local%amp2)
            call close_nml(iunit, "params_dist_initial_double_maxw")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call mpi_barrier(comm, ierr)
        call read_params_dist_initial_double_maxw(species_file, n)

    end subroutine

    module subroutine setup_test_dist_initial_ring(comm, rank, n, params)
        !! Initializes initial ring distribution function
        !! with the given test parameters.
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, intent(in) :: n
        !! Species index
        type(params_dist_initial_ring_t), intent(in) :: params
        !! Parameter of type

        type(params_dist_initial_ring_t) :: params_local
        integer :: iunit, io_error, ierr
        character(:), allocatable :: species_file

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_dist_initial_ring")
            call write_nml(iunit, "params_dist_initial_ring", &
                           "drift_par", params_local%drift_par)
            call write_nml(iunit, "params_dist_initial_ring", &
                           "drift_perp", params_local%drift_perp)
            call write_nml(iunit, "params_dist_initial_ring", &
                           "width_par", params_local%width_par)
            call write_nml(iunit, "params_dist_initial_ring", &
                           "width_perp", params_local%width_perp)
            call close_nml(iunit, "params_dist_initial_ring")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_dist_initial_ring(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

    module subroutine setup_test_dist_initial_slowing_down(comm, rank, &
                                                           n, params)
        !! Initializes initial slowing down distribution function
        !! with the given test parameters.
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank
        integer, intent(in) :: n
        !! Species index
        type(params_dist_initial_slowing_down_t), intent(in) :: params
        !! Parameter of type

        type(params_dist_initial_slowing_down_t) :: params_local
        integer :: iunit, io_error, ierr
        character(:), allocatable :: species_file

        params_local = params

        species_file = trim(get_test_params_file()) &
                     //"_"//trim(get_name(n))//".txt"

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=species_file, action='write', &
                 status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_dist_initial_slowing_down")
            call write_nml(iunit, "params_dist_initial_slowing_down", &
                           "vbirth", params_local%vbirth)
            call close_nml(iunit, "params_dist_initial_slowing_down")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_dist_initial_slowing_down(species_file, n)
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
