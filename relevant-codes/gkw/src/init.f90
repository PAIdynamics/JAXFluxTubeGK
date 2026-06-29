module init

  use general, only : gkw_abort, gkw_warn
  use mpiinterface, only : root_processor
  
  implicit none

  private

  
  ! public subroutines
  public :: initialize,deallocate_runtime_arrays

  ! number of species to read - should go in species
  integer :: n_spec
  
contains

  subroutine initialize

    ! -------------------------------------------------------
    ! SVN:$Id: init.f90 1009 2009-07-02 17:08:44Z  $
    ! This subroutine should do the initalization of all 
    ! the quantities
    !

    use control,         only : read_file, non_linear, & 
         & collisions, method, control_initt
    use grid,            only : setup_grid
    use mode,            only : mode_init, mode_box_recon, mode_check_params
    use mode,            only : kgrid, krbal
    use diagnostic,      only : readfile, noreadfile
    use dist,            only : dist_init
    use components,      only : components_input_species
    use geom,            only : geom_init_grids, parallelize_geom
    use geom,            only : geom_output, geom_check_params, geom_type
    use exp_integration, only : init_explicit
    use non_linear_terms,only : nonlinear_init 
    use rotation,        only : shear_init, need_fft
    use velocitygrid,    only : velgrid_init
    
    ! Read, broadcast and check all the namelist items; initialize anything
    ! necessary before allocation.
    call init_get_params
    
    ! anything else in control
    call control_initt

    ! setup_grid reads the sizes of the grids and calculates the local
    ! grid sizes on each processor from this input. Additional communicators
    ! are set up for a mpi cartesian topology if necessary. 
    call setup_grid
    !Now write parallel setup to input.out?
    !if (last_processor) call grid_write_nml(ilun_out,io_stat)
    
    ! read the diagnostic input parameters
    !call read_diagnostic
    
    ! read the rotation and shear parameters
    !call read_rotation

    ! This routine also sets up and allocates everything required in
    ! dist 
    call dist_init
    
    ! Allocate all the arrays necessary for the computations 
    call allocate_everything 

    ! need to read the species into the now allocated arrays
    call components_input_species
    
    ! and read the geometry dependent parameters
    !call geom_init
    !now done in init_get_parms

    ! Read the mode information
    !APS: now only POST allocate
    call mode_init

    ! Initialize the grids
    call geom_init_grids

    !Since chease read can modify q and shat, some checks need repeating
    if(geom_type=='chease') then 
      call geom_check_params(2)
      call mode_check_params(2)
    endif

    ! if 2D array of modes is used one must call kgrid 
    ! Must be called before krbal. For a single mode one 
    ! must still call this routine since it normalises 
    ! the wave vectors used in the code 
    call kgrid 

    !Write geom quantities to file before parallelize_geom!
    !Note these cannot be written with other write_run_params
    call geom_output
    call mode_box_recon

    !Must be done before parallelize_geom
    call shear_init

    !Warning DO NOT change the order of this and the above routines
    call parallelize_geom

    ! calculate the parameters along the field line 
    call krbal

    !Allocates and initialises the velocity grid (uniform or not)
    !For vptrap case must be called after parallelize_geom
    call velgrid_init

    ! read in the species dependent parameters
    !call components_init
    !now done in init_get_parms

    ! set up the grids for the distribution function
    !call dist_grid_setup <! done by dist

    ! and initialise the distribution 
    call init_fdisi 

    ! read from file (restart) if requested 
    if (read_file) then
       call readfile 
    else
       call noreadfile
    endif

    ! Initialize the parameters for the explict/implicit time step 
    select case(method) 
    case('EXP') 
       call init_explicit
    case default 
       call gkw_abort('init: unknown integration scheme (method)')
    end select

    !The sizes of the real space box and location of the modes are calculated
    !This may be needed even if non_linear = false
    !This must be called after kgrid
    !if(non_linear.or.mode_box.or.shear_real) then
    if (need_fft) then
       call nonlinear_init
    end if
    
  end subroutine initialize

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine init_get_params

  use mpiinterface, only : root_processor, last_processor, mpibarrier
  use control,       only : control_bcast_nml,      control_read_nml ,       &
                          & control_check_params, control_write_nml,         &
                          &  collisions
  use grid,          only : grid_read_nml, grid_write_nml, number_of_species,&
                          & grid_bcast_nml, grid_check_params
  use diagnostic,    only : diagnostic_read_nml, diagnostic_write_nml,       &
                          & diagnostic_bcast_nml, diagnostic_check_params
  use rotation,      only : rotation_read_nml, rotation_write_nml,           &
                          & rotation_bcast_nml, rotation_check_params
  use geom,          only : geom_read_nml, geom_write_nml,                   &
                          & geom_bcast_nml, geom_check_params
  use mode,          only : mode_read_nml, mode_write_nml,                   &
                          & mode_bcast_nml, mode_check_params
  use components,    only : components_read_nml_spcg,                        &
                          & components_bcast_nml_spcg,                       &
                          & components_check_params_spcg,                    &
                          & components_write_nml_spcg,                       & 
                          & components_read_nml_spec,                        &
                          & components_bcast_nml_spec,                       &
                          & components_check_params_spec,                    &
                          & components_write_nml_spec, adiabatic_electrons

  use collisionop,   only : collisionop_read_nml, collisionop_write_nml,     &
                          & collisionop_bcast_nml, collisionop_check_params
                          
  integer, parameter :: ilun_in = 101, ilun_out = 102
  
  integer :: io_stat, i
  logical :: lread_input
    
  ! exit for now when no input file exists
  inquire(FILE='input.dat',EXIST=lread_input)
  if (.not. lread_input) call gkw_abort('input.dat NOT FOUND')
 
  ! open the output file
  if (last_processor) then
    open(ilun_out,file='input.out',FORM='formatted',STATUS='unknown')
  end if

  ! Read and check the various namelists
  ! Set io_stat to zero; only root_processor will obtain a different value.
  io_stat = 0
  
  ! control
  if (root_processor) call read_nml(control_read_nml,ilun_in,io_stat)
  call namelist_error_check('control',io_stat)
  call control_bcast_nml
  call control_check_params
  if (last_processor) call control_write_nml(ilun_out,io_stat)
  
  ! grid
  if (root_processor) call read_nml(grid_read_nml,ilun_in,io_stat)
  call namelist_error_check('gridsize',io_stat)
  call grid_bcast_nml
  call grid_check_params
  if (last_processor) call grid_write_nml(ilun_out,io_stat)

  ! diagnostic
  if (root_processor) call read_nml(diagnostic_read_nml,ilun_in,io_stat)
  call namelist_error_check('diagnostic',io_stat)
  call diagnostic_bcast_nml
  call diagnostic_check_params
  if (last_processor) call diagnostic_write_nml(ilun_out,io_stat)
  
  ! geom
  if (root_processor) call read_nml(geom_read_nml,ilun_in,io_stat)
  call namelist_error_check('geom',io_stat)
  call geom_bcast_nml
  call geom_check_params(1) !may be called again later
  if (last_processor) call geom_write_nml(ilun_out,io_stat)

  ! mode
  if (root_processor) call read_nml(mode_read_nml,ilun_in,io_stat)
  call namelist_error_check('mode',io_stat)
  call mode_bcast_nml
  call mode_check_params(1) !may be called again later
  if (last_processor) call mode_write_nml(ilun_out,io_stat)

  ! rotation
  if (root_processor) call read_nml(rotation_read_nml,ilun_in,io_stat)
  call namelist_error_check('rotation',io_stat)
  call rotation_bcast_nml
  call rotation_check_params
  if (last_processor) call rotation_write_nml(ilun_out,io_stat)

  ! components spcgeneral
  if (root_processor) call read_nml(components_read_nml_spcg,ilun_in,io_stat)
  call namelist_error_check('spcgeneral',io_stat)
  call components_bcast_nml_spcg
  call components_check_params_spcg
  if (last_processor) call components_write_nml_spcg(ilun_out,io_stat)
  
  ! components species
  ! *** The species can be read for checking, but not initialised till after
  ! *** allocate.
  n_spec = number_of_species
  if (adiabatic_electrons) n_spec = n_spec + 1
  do i=1,n_spec
    if (root_processor) then
      call read_nml(components_read_nml_spec,ilun_in,io_stat,SPC=i)
    end if
    call namelist_error_check('species',io_stat,i)
    call components_bcast_nml_spec
    call components_check_params_spec
    if (last_processor) call components_write_nml_spec(ilun_out,io_stat)
  end do

  ! collisions (optional)
  if (collisions) then
    if (root_processor) call read_nml(collisionop_read_nml,ilun_in,io_stat)
    call namelist_error_check('collisions',io_stat)
    call collisionop_bcast_nml
    call collisionop_check_params
    if (last_processor) call collisionop_write_nml(ilun_out,io_stat)
  end if

  if (last_processor) close(ilun_out)
  ! don't want to use input.out till it is written
  call mpibarrier()
  
end subroutine init_get_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> process the namelist given in read_namelist_from_input
!----------------------------------------------------------------------------

subroutine read_nml(read_namelist_from_input,ilun,io_stat,SPC)
  
  use control,       only : control_read_nml
  use grid,          only : grid_read_nml
  use diagnostic,    only : diagnostic_read_nml
  use rotation,      only : rotation_read_nml
  use geom,          only : geom_read_nml
  use mode,          only : mode_read_nml
  use components,    only : components_read_nml_spcg,                        &
                          & components_read_nml_spec
  use collisionop,   only : collisionop_read_nml
  
  interface
    subroutine read_namelist_from_input(i_lun,i_stat,l_write)
      integer, intent(in)  :: i_lun
      integer, intent(out) :: i_stat
      logical, optional, intent(in)  :: l_write
    end subroutine read_namelist_from_input
  end interface

  integer, intent(in)  :: ilun
  integer, intent(out) :: io_stat
  integer, optional, intent(in) :: SPC

  logical, parameter :: l_write = .false.
  integer :: io_stat_check
    
  ! Open the file, unless we are reading species other than the
  ! first one.
  if (present(SPC)) then
    if (SPC == 1) then
      open(ilun,file='input.dat',FORM='formatted',STATUS='old')
    end if
  else
    open(ilun,file='input.dat',FORM='formatted',STATUS='old')
  end if

  ! call the read_input routine in the appropriate module
  call read_namelist_from_input(ilun,io_stat_check,l_write)
  io_stat = io_stat_check
  
  ! Close the file, unless we have more species to read
  if (present(SPC)) then
    if (SPC == n_spec) close(ilun)
  else
    close(ilun)
  end if
    
end subroutine read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Check io_stat for a read then abort if any non-zero values are found.
!----------------------------------------------------------------------------

subroutine namelist_error_check(list_name,io_stat,ispc)

  use mpiinterface, only : root_processor, mpibcast_integer
  use general,       only : gkw_clean_abort

  character (len=*), intent(in) :: list_name
  integer, intent(in) :: io_stat
  integer, optional, intent(in) :: ispc

  integer :: my_io_stat

  my_io_stat = 0

  ! root_processor first reports any error
  if (root_processor) then

    if (io_stat > 0) then
      write(*,*) '* Error in namelist '//list_name//'.'
      if (present(ispc)) write(*,*) '* for species number: ',ispc
    end if
    
    if (io_stat < 0) then
      write(*,*) '* Namelist '//list_name//' MISSING.'
      if (present(ispc)) write(*,*) '* for species number: ',ispc
    end if
    
    my_io_stat = io_stat
    
  endif

  ! broadcast io_stat from root_processor
  call mpibcast_integer(my_io_stat,1)

  ! abort on non-zero io_stat
  if (my_io_stat /= 0) then
    call gkw_clean_abort('namelist_error_check: problem with '//list_name)
  end if

end subroutine namelist_error_check

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  subroutine init_fdisi 
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ! Subroutine that initialises the distribution function with random
    ! numbers
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
    use marsaglia,  only : ran_kiss
    use control,    only : nlapar, nlbpar
    use grid,       only : nmod, nx, ns, nsp, nmu, nvpar, nperiod, parallel_s
    use dist,       only : fdisi, indx, fmaxwl, nsolc, &
         & falpha, iphi, iapar, ibpar
    use geom,       only : bn, sgr, shat,q, eps 
    use mode,       only : mode_box, kxrh, krho, ixzero, lx
    use components, only : signz, de, amp_init, finit, types, pbg
    use velocitygrid, only : vpgr, mugr, intmu, intvp
    use constants, only : ci1, pi 

    ! integers for the loop over all grid points 
    integer imod, ix ,i, j, k, is, p, line

    ! Dummy variables 
    real random1, random2, norm, par, vn,fdum,vperp
    logical alpha_dist

    integer idum 
    
    ! calculate the normalized maxwellian vthi**3/no Fm
    do i = 1, ns 
       do j = 1, nmu 
          do k = 1, nvpar 
             fmaxwl(i,j,k) = exp(-vpgr(i,j,k)**2-2.E0*bn(i)*mugr(j)) &
                  & /(sqrt(pi)**3)
          end do
       end do
    end do

    ! the slowing down distribution 
    alpha_dist = .false. 
    do is = 1, nsp 
       if (types(is).eq.'alpha') then 
          alpha_dist = .true.
          par        = pbg(is)
       endif
    end do
    if (alpha_dist) then 
       !WARNING this part works only for the vpar_max = 3. 
       ! specify as temperature E_alpha / (9 T_ref) 
       ! the parameter is pbg = 9 E_c / E_alpha 
       do i = 1, ns 
          do j = 1, nmu 
             do k = 1, nvpar 
                vn = sqrt(vpgr(i,j,k)**2+2.E0*bn(i)*mugr(j)) 
                if (vn.lt.3) then 
                   falpha(i,j,k) = 3.E0 / (4.*pi*log(1.E0 + 27.E0*par**(-1.5))* & 
                        & (par**1.5 + vn**3))
                endif
             end do
          end do
       end do
    end if

    ! Renormalize the Maxwell to ensure the correct density
    ! Switched off at the moment since this does not work 
    ! when parallizing over the magnetic moment
    do i = 1, ns 
       norm = 0.E0
       do j = 1, nmu 
          do k = 1, nvpar 
             norm = norm + bn(i)*intmu(j)*intvp(i,j,k)*fmaxwl(i,j,k)
          end do
       end do
       !  do j = 1, nmu 
       !    do k = 1, nvpar 
       !      fmaxwl(i,j,k) = fmaxwl(i,j,k)/norm 
       !    end do 
       !  end do 
       ! write(*,*)i,norm 
    end do
    !top 

    ! set the potential to zero (and if necessary the vector potential and the magnetic field) 
    do imod = 1, nmod 
       do ix = 1, nx
          do i = 1, ns 
             ! the potential 
             fdisi(indx(imod,ix,i,iphi)) = 0.
             ! the vector potential 
             if (nlapar) then 
                fdisi(indx(imod,ix,i,iapar)) = 0.
             endif
             ! the parallel magnetic field perturbation 
             if (nlbpar) then 
                fdisi(indx(imod,ix,i,ibpar)) = 0
             endif
          end do
       end do
    end do

    select case(finit)

    case('noise')

       do imod = 1, nmod
          do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp 
                      do ix = 1, nx 

                         ! No initialisation of the zonal flow 
                         if ((imod.eq.1).and.mode_box.and.(nmod.ne.1)) then
                            fdisi(indx(imod,ix,i,j,k,is)) = 0.  
                         else 
                            random1 = ran_kiss() 
                            random2 = ran_kiss() 
                            fdisi(indx(imod,ix,i,j,k,is)) = amp_init *         &
                                 &   de(is) * ((1-2*random1) + (1-2*random2)*ci1)

                         endif

                      end do
                   end do
                end do
             end do
          end do
       end do

    case('cosine')

       do imod = 1, nmod
          do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp 
                      do ix = 1, nx 

                         ! No initialisation of the zonal flow 
                         if ((imod.eq.1).and.mode_box.and.(nmod.ne.1)) then
                            fdisi(indx(imod,ix,i,j,k,is)) = 0.  
                         else 
                            fdisi(indx(imod,ix,i,j,k,is)) = amp_init *      &
                                 &    de(is) * cos(2*pi*sgr(ix,i)) 
                         endif

                      end do
                   end do
                end do
             end do
          end do
       end do

    case('sine') 

       do imod = 1, nmod
          do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp
                      do ix = 1, nx

                         ! No initialisation of the zonal flow
                         if ((imod.eq.1).and.mode_box.and.(nmod.ne.1)) then
                            fdisi(indx(imod,ix,i,j,k,is)) = 0.
                         else
                            fdisi(indx(imod,ix,i,j,k,is)) = amp_init *      &
                                 &    de(is) * sin(2*pi*sgr(ix,i))
                         endif

                      end do
                   end do
                end do
             end do
          end do
       end do

    case('gauss')

       do imod = 1, nmod
          do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp 
                      do ix = 1, nx 

                         ! No initialisation of the zonal flow 
                         if ((imod.eq.1).and.mode_box.and.(nmod.ne.1)) then
                            fdisi(indx(imod,ix,i,j,k,is)) = 0.  
                         else 
                            fdisi(indx(imod,ix,i,j,k,is)) = amp_init *     & 
                                 & exp(-(real(imod)/real(2*nmod))**2)*         &
                                 & exp(-(real(ix)/real(2*nx))**2)
                         endif

                      end do
                   end do
                end do
             end do
          end do
       end do

    case('zonal')

       do imod = 1, nmod
          do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp 

                      idum = 0
                      do ix = 1, nx 
                         ! all non zonal components are zero 
                         if (abs(krho(imod)).gt.1e-10) then 
                            fdisi(indx(imod,ix,i,j,k,is)) = (0.E0,0.E0)
                         endif
                         ! find the central mode 
                         if (abs(kxrh(ix)).lt.1e-10) then
                            !use ixzero here from mode module
                            idum = ix 
                         endif
                      end do
                      if (idum.eq.0) then 
                         write(*,*)'No kx = 0 mode found in init'
                         stop
                      endif
                      ix = idum 
                      if ((ix.le.1).or.(ix.ge.nx)) then 
                         write(*,*)'No finite kx mode found in init'
                         stop 
                      endif
                      fdisi(indx(imod,1,i,j,k,is)) = ci1 * amp_init * &
                           & fmaxwl(i,j,k)
                      fdisi(indx(imod,nx,i,j,k,is)) = -ci1 *amp_init * &
                           & fmaxwl(i,j,k)

                   end do
                end do
             end do
          end do
       end do
       
    case('sgaus')
      write(*,*)'Shifted gaussian chosen'
      do j = 1, nmu
        do k = 1, nvpar
          do i=1,nsp
            ! No initialisation of the zonal flow 
            vperp = sqrt(2.E0*bn(1)*mugr(j))
            fdisi(indx(1,1,1,j,k,i)) = amp_init*exp(-(vpgr(1,j,k)-1.5)**2/(.01*sqrt(pi)**3))*exp(-(vperp-1.)**2/(.01*sqrt(pi)**3))
          end do
        end do
      end do
       
    case('kxzero') !A single mode with kx=0, useful for testing
        if (ixzero.eq.0) call gkw_abort("No kx = 0 mode for odd nx")
        fdisi=0.E0
           do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp 
                      fdisi(indx(3,ixzero,i,j,k,is))=amp_init
                   end do
                end do
              end do
          end do

    case('kyzero') !A single mode with ky=0, useful for testing
       fdisi=0.E0
           do i = 1, ns
             do j = 1, nmu
                do k = 1, nvpar
                   do is = 1, nsp 
                      fdisi(indx(1,ixzero+3,i,j,k,is))=amp_init
                   end do
                end do
              end do
          end do

    case('line') !A line of gaussians (in real space)
       fdisi=0.E0
           do i = 1, ns; do j = 1, nmu; do k = 1, nvpar; do is = 1, nsp 
              do ix = 1, nx; do imod = 1, nmod
                 do line=-49,50 !100 Gaussians shifted by 0.01
                  fdisi(indx(imod,ix,i,j,k,is))=fdisi(indx(imod,ix,i,j,k,is)) + &
                  amp_init*( exp(-(real(ix-ixzero)/real(2*nx))**2) +            &
                             exp(-(real(imod)/(2*nmod)**2))           )         &
                           * exp(ci1*kxrh(ix)*line*0.01*lx) 
                 end do
              end do; end do
           end do; end do; end do; end do

    case default

       write(*,*)'Unknown initilization switch finit'
       stop 

    end select

  end subroutine init_fdisi

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


  subroutine allocate_everything 
    !--------------------------------------------------------------------
    ! This routine calls all the routines that allocate the arrays of 
    ! the different modules 
    !--------------------------------------------------------------------
    use control,          only : non_linear, method 
    use matdat,           only : matdat_allocate 
    use components,       only : components_allocate 
    use geom,             only : geom_allocate 
    use mode,             only : mode_allocate, mode_box 
    use diagnostic,       only : diagnostic_allocate
    use non_linear_terms, only : nonlinear_allocate
    use rotation,         only : rotation_allocate

    implicit none 

    integer :: i

    ! Allocate the arrays of mat.f90 
    call matdat_allocate

    ! Allocate the arrays of components.f90 
    call components_allocate

    ! Allocate the arrays of geom.f90 
    call geom_allocate

    ! Allocate the arrays of mode.f90  
    call mode_allocate  

    ! Allocate the arrays of non_linear_terms.f90 Note 
    ! that this arrays are also used for the recon-
    ! struction of linear modes if mode_box is true 
    !if ((non_linear).or.(mode_box).or.(shear_real)) then 
    call nonlinear_allocate 
    !endif 
    !switched off because mode_box is not yet read from 
    !input??

    call rotation_allocate


    ! Allocate the arrays of diagnostic.f90 
    call diagnostic_allocate 
  
  end subroutine allocate_everything

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine deallocate_runtime_arrays

  use exp_integration, only : exp_integration_deallocate
  
  call exp_integration_deallocate

end subroutine deallocate_runtime_arrays

!****************************************************************************

end module init
