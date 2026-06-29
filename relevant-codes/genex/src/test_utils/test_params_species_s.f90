submodule(test_params_m) test_params_species_s
    ! Submodule that contains helpers for unit tests to initialize the species
    ! with non-default parameters.
    use genex_fortran_env_m, only : GP
    use test_params_io_m
    implicit none

contains

    module subroutine setup_test_species(comm, rank, &
                                         names, masses, charges, &
                                         temp_scalings)
        !! Initialize the simulation with the given test parameters for the
        !! species parameters
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        character(len=24), dimension(n_spec_supported), optional, &
            intent(in) :: names
        real(kind=GP), dimension(n_spec_supported), optional, &
            intent(in) :: masses
        real(kind=GP), dimension(n_spec_supported), optional, &
            intent(in) :: charges
        real(kind=GP), dimension(n_spec_supported), optional, &
            intent(in) :: temp_scalings

        character(len=24), dimension(n_spec_supported) :: names_local
        real(kind=GP), dimension(n_spec_supported) :: masses_local, &
                                                      charges_local, &
                                                      temp_scalings_local
        integer :: n, iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        do n=1, n_spec_supported
            names_local(n) = get_name(n)
            masses_local(n) = get_mass(n)
            charges_local(n) = get_charge(n)
            temp_scalings_local(n) = get_temp_scaling(n)
        end do

        if(present(names)) names_local = names
        if(present(masses)) masses_local = masses
        if(present(charges)) charges_local = charges
        if(present(temp_scalings)) temp_scalings_local = temp_scalings

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_species")

            call write_nml(iunit, "params_species", "names", &
                           names_local)
            call write_nml(iunit, "params_species", "masses", &
                           masses_local)
            call write_nml(iunit, "params_species", "charges", &
                           charges_local)
            call write_nml(iunit, "params_species", "temp_scalings", &
                           temp_scalings_local)

            call close_nml(iunit, "params_species")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_species(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)
    end subroutine
end submodule
