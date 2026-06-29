submodule(test_params_m) test_params_neutrals_s
    ! Submodule that contains helpers for unit tests to initialize the
    ! neutrals with non-default parameters.
    use genex_fortran_env_m, only : GP
    use test_params_io_m
    use file_handling_m, only: file_exists
    implicit none

contains

    module subroutine setup_test_neutrals(comm, rank, &
                                          names, masses, mapped_ions_names)
        !! Initialize the simulation with the given test parameters for the
        !! neutrals parameters
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        character(len=24), dimension(n_neut_supported), optional, &
            intent(in) :: names
        real(kind=GP), dimension(n_neut_supported), optional, &
            intent(in) :: masses
        character(len=24), dimension(n_neut_supported), optional, &
            intent(in) :: mapped_ions_names

        character(len=24), dimension(n_neut_supported) :: names_local, &
                                                         mapped_ions_names_local
        real(kind=GP), dimension(n_neut_supported) :: masses_local
        integer :: o, iunit, io_error, ierr
        character(len=48) :: neutrals_file, default_neutrals_file, &
                             test_neutrals_file

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        do o=1, n_neut_supported
            names_local(o) = get_neut_name(o)
            masses_local(o) = get_neut_mass(o)
            mapped_ions_names_local(o) = get_neut_mapped_ion_name(o)
        end do

        if(present(names)) names_local = names
        if(present(masses)) masses_local = masses
        if(present(mapped_ions_names)) then
            mapped_ions_names_local = mapped_ions_names
        end if

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_neutrals")

            call write_nml(iunit, "params_neutrals", "names", &
                           names_local)
            call write_nml(iunit, "params_neutrals", "masses", &
                           masses_local)
            call write_nml(iunit, "params_neutrals", "mapped_ions_names", &
                           mapped_ions_names_local)

            call close_nml(iunit, "params_neutrals")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read mesh parameters
        call read_params_neutrals(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

        ! Check if default parameters file exist for all the species
        ! If not, create a file for the test
        if(rank == 0) then
        do o=1, get_n_points_neut()
            neutrals_file         = trim(default_params_file)&
                                  //"_"//trim(get_neut_name(o))//".txt"
            default_neutrals_file = trim(default_params_file)&
                                  //"_hydrogen.txt"
            test_neutrals_file    = trim(get_test_params_file())&
                                  //"_"//trim(get_neut_name(o))//".txt"
            if(.not. file_exists(neutrals_file)) then
                call execute_command_line("cp "&
                                         //default_neutrals_file//" "&
                                         //test_neutrals_file)
            end if
        end do
        end if

    end subroutine
end submodule
