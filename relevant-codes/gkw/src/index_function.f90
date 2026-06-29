!****************************************************************************
!>
!! 
!<
!****************************************************************************

module index_function

  implicit none

  private
  
  public :: index_init,index_set_ghostpoints,indx_,dummy_index,max_dims,register_offset
  public :: index_invert_,index_reorder 
 
  integer, parameter :: max_dims = 6
  integer, parameter :: max_fields = 12
  integer, parameter :: max_offsets = max_dims*8
  integer, dimension(max_dims) :: index_order, gridsize, gridsize_fields,n_ghostpoints_hack
  integer, dimension(max_dims,max_dims) :: n_ghostpoints
  integer, dimension(max_offsets,max_dims+1) :: known_map
  integer, dimension(max_offsets) :: offset
  integer, dimension(max_fields)  :: field_ref, field_offset
  integer, dimension(max_fields)  :: multi_species_field
  
  integer :: n_offsets, n_fields,indmax
  integer, parameter :: dummy_index = -5423

  interface indx_
    module procedure index_main
    module procedure index_other
  end interface

  interface index_invert_
    module procedure index_invert
  end interface
 
  interface index_reorder
    module procedure index_reorder_field
    module procedure index_reorder_main
  end interface
  logical :: register_ready = .false.
contains

!****************************************************************************
!> This routine should be called before any other routine in this module. It
!! sets the order of the dimensions in the fdisi array, calculates a
!! corresponding gridsize array to that order, sets the maximum value that
!! the index can be (based on the calculation in dist) and initialises various
!! quantities used in this module. The first 3 args for this subroutine are
!! not used but presently remain here to make the code restartable from old
!! restart files. It may be useful to allow the order to be set either via
!! input or to vary with the grid configuration. The selected index_order may
!< need to be changed after parts of the code are optimised further.
!-----------------------------------------------------------------------------

subroutine index_init(lsendrecv_mu, lindex_vpar_last, parallel_s, imax)
  
  use global,  only : id_x, id_vpar, id_mu, id_x, id_mod, id_sp, id_s
  use grid,    only : nx, nmod, nvpar, nsp, ns, nmu
  use general, only : gkw_abort

  integer, intent(in) :: imax
  logical, intent(in) :: lsendrecv_mu,lindex_vpar_last,parallel_s
  integer :: i,j
  
  ! The maximum index
  indmax = imax

  ! This bit does nothing unless the line below it is commented out.
  if (lsendrecv_mu) then
    index_order = (/ id_s, id_vpar, id_sp, id_x, id_mod ,id_mu /) 
  else if (lindex_vpar_last) then
    index_order = (/ id_s, id_mu, id_sp, id_x, id_mod, id_vpar /)
  else if(parallel_s) then
    index_order = (/ id_vpar, id_mu, id_sp, id_x, id_mod, id_s /)
  else
    index_order = (/ id_vpar, id_mu, id_s, id_x, id_mod, id_sp /)
  endif

  ! This is the default order and needs to be commented out to read old
  ! restart files into memory correctly.
  index_order = (/ id_x,id_mod,id_s,id_vpar,id_mu,id_sp/)
   
  ! build a correctly ordered array of grid sizes
  do i=1, max_dims
  
    ! check for duplicates in the ordering
    do j=1, max_dims
      if (i /= j .and. index_order(i) == index_order(j)) then
        call gkw_abort('index_init: duplicate index in init')
      end if
    end do
     
    ! Insert the correct dimension size into the current gridsize position.
    select case(index_order(i))
      case(id_vpar) ; gridsize(i) = nvpar
      case(id_s)    ; gridsize(i) = ns
      case(id_sp)   ; gridsize(i) = nsp
      case(id_mu)   ; gridsize(i) = nmu
      case(id_x)    ; gridsize(i) = nx
      case(id_mod)  ; gridsize(i) = nmod
      case default  ; call gkw_abort('index_init: bad index id')
    end select

  end do

  ! initialise other quantities used in this module
  n_offsets = 0
  n_fields  = 0
  n_ghostpoints(:,:) = 0
  multi_species_field(:) = 0
 
  ! ready to use
  register_ready = .true.
  
end subroutine index_init

!****************************************************************************
!> Set up the number of ghost points required in each direction. The array
!! n_ghostpoints(i,j) contains the number of ghost points in the combined
!! directions i and j i.e. only for combinations of 2. If i corresponds to
!! s-direction and j corresponds to the vpar-direction, then 
!! n_ghostpoints(i,i) is the number of s-points which must be communicated in
!! the s-direction and n_ghostpoints(i,j) ( == n_ghostpoints(j,i) ) is the
!! product of the number of points in s and vpar-directions that should be
!! communicated between the local processor and the processor shifted 1 in
!< both of those directions.
!----------------------------------------------------------------------------

subroutine index_set_ghostpoints(gp_mu, gp_s, gp_vpar, gp_vpar_mu, gp_vpar_s)

  use global,  only : id_x, id_vpar, id_mu, id_x, id_mod, id_sp, id_s
  use general, only : gkw_abort

  integer, intent(in), optional :: gp_s, gp_mu, gp_vpar, gp_vpar_mu, gp_vpar_s
  integer :: i,j
  
  do j = 1, max_dims
    do i = 1, max_dims
    
      select case(index_order(i))
      
        case (id_mu)

          mu_outer : if (present(gp_mu)) then
            n_ghostpoints(i,i) = gp_mu
            select case (index_order(j))
              case (id_vpar)
                vpar_mu_inner : if (present(gp_vpar_mu)) then
                  n_ghostpoints(i,j) = gp_vpar_mu
                end if vpar_mu_inner
            end select
          end if mu_outer
          
        case (id_s)
          
          s_outer : if (present(gp_s)) then
            n_ghostpoints(i,i) = gp_s
            select case (index_order(j))
              case (id_vpar)
                vpar_s_inner : if (present(gp_vpar_s)) then
                  n_ghostpoints(i,j) = gp_vpar_s
                end if vpar_s_inner
            end select
          end if s_outer
          
        case (id_vpar)
          
          vpar_outer : if (present(gp_vpar)) then
            n_ghostpoints(i,i) = gp_vpar
            select case (index_order(j))
              case (id_s)
                s_inner : if (present(gp_vpar_s)) then
                  n_ghostpoints(i,j) = gp_vpar_s
                end if s_inner
              case (id_mu)
                mu_inner : if (present(gp_vpar_mu)) then
                  n_ghostpoints(i,j) = gp_vpar_mu
                end if mu_inner
            end select
          end if vpar_outer
            
        case (id_x,id_mod,id_sp)
          
          ! do nothing in these cases; presently no need to index them
        
        case default
          call gkw_abort('index_set_ghostpoints: bad')
      end select
      
    end do
  end do
  
end subroutine index_set_ghostpoints

!****************************************************************************
!> Returns an index for fdisi by calling get_index(). This routine just
!< re-orders the inputs beforehand as appropriate.
!----------------------------------------------------------------------------

function index_main(i_mod, i_x, i_s, i_mu, i_vpar, i_sp)

  use global,  only : id_sp, id_s, id_vpar, id_mu, id_mod, id_x
  use grid,    only : ns, nx, nmod, nvpar, nmu, nsp
  use general, only : gkw_abort
  
  integer, intent(in) :: i_x, i_s, i_mu, i_vpar, i_sp, i_mod
  integer, dimension(max_dims) :: ind_array
  integer :: index_main, indx, i

  ! insert dummies to start with, although unnecessary
  ind_array(:) = dummy_index
  
  ! order the inputs
  do i = 1, max_dims
    select case(index_order(i))
      case (id_sp)   ; ind_array(i) = i_sp
      case (id_s)    ; ind_array(i) = i_s
      case (id_x)    ; ind_array(i) = i_x
      case (id_mod)  ; ind_array(i) = i_mod
      case (id_mu)   ; ind_array(i) = i_mu
      case (id_vpar) ; ind_array(i) = i_vpar
      case default   ; call gkw_abort('bad index_main') ! will never happen
    end select
  end do

  ! call the general index function
  call get_index(ind_array, indx)
  index_main = indx

end function index_main

!****************************************************************************
!> Return an index for the fields by calling get_index(). This routine just
!! re-orders the inputs and checks the switches beforehand. The switch
!! provided should have been recorded by calling the register_offset routine,
!< along with the location in the fdisi array.
!----------------------------------------------------------------------------

function index_other(isw, i_mod, i_x, i_s, i_sp)

  use global,  only : id_sp, id_s, id_vpar, id_mu, id_mod, id_x
  use grid,    only : ns, nx, nmod, nsp
  use general, only : gkw_abort

  integer, intent(in) :: i_x,i_s,i_mod,isw
  integer, optional, intent(in) :: i_sp
  integer, dimension(max_dims) :: ind_array
  integer :: index_other,indx,i,iswitch
  logical :: good_switch

  iswitch = isw
 
  ! Order the inputs; dummies are put into the array where entries are
  ! not present.
  ind_array(:) = dummy_index
  do i = 1,max_dims
    select case(index_order(i))
      case (id_sp)   ; if (present(i_sp)) ind_array(i) = i_sp
      case (id_s)    ; ind_array(i) = i_s
      case (id_x)    ; ind_array(i) = i_x
      case (id_mod)  ; ind_array(i) = i_mod
      case (id_mu,id_vpar) ; ! do nothing
      case default   ; call gkw_abort('bad index_other') ! will never happen
    end select
  end do
  
  ! check the switch to see if it is known to the index function
  good_switch = .false.
  do i=1,n_fields
    if (isw == field_ref(i)) then
      good_switch = .true.
      if (multi_species_field(i) > 0 .and. (.not. present(i_sp))) then
        call gkw_abort('index_other: species number required, but not present')
      end if
    end if
  end do
  ! abort if the switch is unknown - should have been registered
  if (.not. good_switch) call gkw_abort('index_other: unregistered switch')

  ! call the general index function
  call get_index(ind_array, indx, iswitch)
  index_other = indx
  
end function index_other

!****************************************************************************
!> Routine that works out the index. ind_array contains 6 integers
!! corresponding to the grid points in the various directions. When all the
!! integers are in the normal range, from 1 to the number of local grid points
!! in the direction, the index is 
!!  ind_array(1) + (ind_array(2) - 1)*gridsize(1) + ...
!!               + (ind_array(6) - 1)*gridsize(1)*gridsize(2)*...*gridsize(5),
!! where gridsize is the corresponding array of local grid sizes. This is the
!! case for the full local part of the solution, fdisi(1:nf). 
!!
!! If the ind_array values are outside the normal range by a few, the call may
!! refer to a point of the solution on another processor. If the input refers
!! to the solution on the next processor in some direction, then it will be
!! larger than the maximum gridsize value in that direction. When the size of
!! the grid in that direction is subtracted from that value, it should then be
!! between 1 and the maximum number of `ghost' points that we would wish to
!! reference. If the input value is too small, it can be shifted up into this
!! same range. To generate an index for the bit of the array that contains
!! the points from a particular adjacent processor, the gridsize entry for
!! that direction can be reduced to the number of ghost points in that
!! direction, then the above can be used with the shifted ind_array value.
!! The block containing the solution from another processor is offset from the
!! start of fdisi by some amount which is registered from dist. The pattern
!! of shifts in ind_array is stored in the remap array, then used to look up
!! the shift provided by dist.
!!
!! In the case of fields, which only span 3 or so of the dimensions, a dummy
!! index is located in unused dimensions when this routine is called. The
!! shifts and offsets can be obtained in the same way as the distribution
!! function. The dummy dimensions are ignored for the purpose of calculating
!< an index value. The switch is stored in an extra slot in the remap array.
!----------------------------------------------------------------------------

subroutine get_index(ind_array, indx, switch)

  use global,  only : id_sp, id_s, id_vpar, id_mu, id_mod, id_x
  use general, only : gkw_abort

  integer, dimension(max_dims), intent(inout)  :: ind_array
  integer, optional, intent(inout) :: switch
  integer, intent(out) :: indx

  integer, dimension(max_dims) :: fact, length
  integer, dimension(max_dims+1) :: remap
  integer :: i, constant, block, block_len, ioffset, j, outside_range_count
  integer :: jloc
  
  ! initialise the length array to the array of gridsizes
  length = gridsize
  ! initialise the remap and factor arrays
  remap(:) = 0 ; fact(:) = 0
  
  !
  ! perform a check and shift (if necessary) on every input dimension
  !
  
  ! First count how many indices are out of range. The maximum is 2.
  outside_range_count = 0
  do i=1, max_dims
    if ((ind_array(i) < 1 .or. ind_array(i) > length(i)) .and.               &
        &  ind_array(i) /= dummy_index) then
      outside_range_count = outside_range_count + 1
    end if
  end do
  if (outside_range_count > 2) then
    call gkw_abort('get_index: there are presently valid calls with more'//  &
        &          'than 2 indices out of range')
  end if
   
  check_all_dims : do i=1,max_dims

    ! Ignore dummy entries; this array may contain dummies if referencing
    ! a field where some entries are meaningless.
    not_dummy : if (ind_array(i) /= dummy_index) then
       
      ! set a default zero for the remap
      remap(i) = 0

      ! Check if the input value is less than 1
      too_small : if (ind_array(i) < 1) then
        ! If there are ghost points in this direction, it could refer to a
        ! point on the previous processor. Shift the value and check.
        processor_below : if (n_ghostpoints(i,i) > 0) then
          if (outside_range_count == 1) then
            ind_array(i) = ind_array(i) + n_ghostpoints(i,i)
            length(i)    = n_ghostpoints(i,i)
            ! Check that the shifted value is now in the acceptable range for
            ! a ghost point i.e. between 1 and n_ghostpoints(i,i).
            if (ind_array(i) < 1 .or. ind_array(i) > n_ghostpoints(i,i)) then
              call gkw_abort('indx main: bad input')
            end if
          else if (outside_range_count == 2) then
            ! find the other index which is shifted
            jloc = 0
            do j=1,max_dims
              if (i /= j) then
                if (n_ghostpoints(i,j) /= 0) then
                  jloc = j
                end if
              end if
            end do
            if (jloc == 0) call gkw_abort('get_index: bad jloc below')
            ind_array(i) = ind_array(i) + n_ghostpoints(i,jloc)
            length(i)    = n_ghostpoints(i,jloc)
            ! Check that the shifted value is now in the acceptable range for a
            ! ghost point i.e. between 1 and n_ghostpoints(i,i).
            if (ind_array(i) < 1 .or. ind_array(i) > n_ghostpoints(i,jloc)) then
              call gkw_abort('indx main: bad input')
            end if
          end if
          ! Make a note that this point is on the previous processor.
          remap(i) = -1
        else ! a bad input value
          call gkw_abort('indx_main: i < 1')
        end if processor_below
        
      end if too_small
      
      ! Check if the input value is greater than the number of points in the
      ! local grid for the present direction.
      too_big : if (ind_array(i) > length(i)) then
        ! If there are ghost points in this direction, the point may be on the
        ! next processor. Shift the value into the right range.
        processor_above : if (n_ghostpoints(i,i) > 0) then
          if (outside_range_count == 1) then
            ind_array(i) = ind_array(i) - length(i)
            length(i)    = n_ghostpoints(i,i)
            ! Check that the shifted value is now in the acceptable range for a
            ! ghost point i.e. between 1 and n_ghostpoints(i,i).
            if (ind_array(i) < 1 .or. ind_array(i) > n_ghostpoints(i,i)) then
              call gkw_abort('indx main: bad input')
            end if
          else if (outside_range_count == 2) then
            ! find the other index which is shifted
            jloc = 0
            do j=1,max_dims
              if (i /= j) then
                if (n_ghostpoints(i,j) /= 0) then
                  jloc = j
                end if
              end if
            end do
            if (jloc == 0) call gkw_abort('get_index: bad jloc below')
            ind_array(i) = ind_array(i) - length(i)
            length(i)    = n_ghostpoints(i,jloc)
            ! Check that the shifted value is now in the acceptable range for a
            ! ghost point i.e. between 1 and n_ghostpoints(i,i).
            if (ind_array(i) < 1 .or. ind_array(i) > n_ghostpoints(i,jloc)) then
              call gkw_abort('indx main: bad input')
            end if
          end if
          ! Make a note that this point is on the next processor
          remap(i) = 1
        else ! a bad input value
          call gkw_abort('indx_main: i > nmax')
        end if processor_above
      end if too_big
 
    else
      
      ! put a dummy in the length array
      length(i) = dummy_index
      
    end if not_dummy
    
  end do check_all_dims
  
  !
  ! compact the arrays e.g. (12,4,dummy,3,dummy,dummy) -> (12,4,3,0,0,0)
  ! This allows the different types of blocks to be dealt with in the same way.
  !
  
  call compact_array(length)
  call compact_array(ind_array)

  !
  ! work out the array (fact) and integer (constant) to calculate indx
  !
  
  ! initial values
  block = 1
  block_len = 0
  constant = 1
  
  set_blocksizes : do i=1,max_dims
   
    ! Exit if the length is zero, otherwise the bit to add on will be wrong. 
    if (length(i) == 0) exit set_blocksizes
    
    ! set the prefactors and update the additional bit
    fact(i)  = block
    constant = constant - block
    
    ! update the block size
    block_len = length(i)
    block=block*block_len
    
  end do set_blocksizes
  
  !
  ! Calculate the index, using the remap array to obtain the pre-registered
  ! block offset.
  !
  
  ! store any switch for a `field' in an additional slot in remap
  if (present(switch)) then
    remap(max_dims+1) = switch
  end if 

  ! get the offset
  ioffset = index_offset(remap)

  ! indx
  indx = sum(fact*ind_array) + constant + ioffset

  ! check for out of bounds
  if (indx < 0 .or. indx > indmax) then
    call gkw_abort('get_index: bad indx value')
  end if
    
end subroutine get_index

!****************************************************************************
!> Obtain the offset in fdisi by making a comparison of the shifts and 
!! switches recorded in the remap array with those in a table of patterns and
!< offsets.
!----------------------------------------------------------------------------

function index_offset(remap)

  use general, only : gkw_abort

  integer, dimension(max_dims+1), intent(in) :: remap
  integer :: index_offset
  logical, dimension(max_dims+1) :: found
  integer :: i,j
  
  ! loop over the known offsets
  known_patterns : do i=1,n_offsets
  
    ! check if the generated pattern matches the list entry
    found(:) = .false.
    check_pattern : do j=1, max_dims+1
      if(remap(j) == known_map(i,j)) found(j) = .true.
    end do check_pattern
    
    ! return the offset if the pattern matches
    good_pattern : if (all(found)) then
      index_offset = offset(i)
      return
    end if good_pattern
    
  end do known_patterns

  ! The pattern is not in the list, so no offset can be returned.
  write (*,*) 'remap:       ',remap(:)
  call gkw_abort('index_offset: unregistered offset')
  
end function index_offset

!****************************************************************************
!> Record the offset together with call pattern. From dist, a call to this
!! routine is required from each block. The offset/pattern combination is
!! then used to get the right offset when the index function is called. The
!! default pattern for any direction is 0 (a zero) if it is not present (all
!< directions are optional). The combination is stored in known_map.
!----------------------------------------------------------------------------

subroutine register_offset(imod, ix, is, imu, ivpar,isp, ioffset, ifield, nsp)

  use global,  only : id_x, id_s, id_sp, id_vpar, id_mu, id_mod
  use general, only : gkw_abort
 
  integer, optional, intent(in) :: is, isp, imu, ivpar, ix, imod, nsp, ifield
  integer, intent(in) :: ioffset
  integer, dimension(max_dims+1) :: map
  integer :: i, i_vpar, i_mu, i_s, i_sp, i_x, i_mod

  if (.not. register_ready) call gkw_abort('register_offset: not ready')
  
  map(:) = 0
  ! all zero defaults
  i_mod = 0 ; i_s = 0 ; i_sp = 0 ; i_vpar = 0 ; i_x = 0 ; i_mu = 0
  
  ! change patterns for items present
  if (present(imod)) i_mod = imod
  if (present(is)) i_s = is
  if (present(isp)) i_sp = isp
  if (present(imu)) i_mu = imu
  if (present(ivpar)) i_vpar = ivpar
  if (present(ix)) i_x = ix

  ! for the fields, make a fields reference list and store the switch.
  if (present(ifield)) then
    n_fields = n_fields + 1
    field_ref(n_fields) = ifield
    if (present(nsp))then
      multi_species_field(n_fields) = nsp
    end if
    ! the last map value contains the switch
    map(max_dims+1) = ifield
  end if
  
  ! create the pattern
  do i = 1, max_dims
    select case(index_order(i))
      case (id_sp)   ; map(i) = i_sp
      case (id_s)    ; map(i) = i_s
      case (id_x)    ; map(i) = i_x
      case (id_mod)  ; map(i) = i_mod
      case (id_mu)   ; map(i) = i_mu
      case (id_vpar) ; map(i) = i_vpar
      case default
        write (*,*) index_order
        write (*,*) map
        call gkw_abort('bad register offset') ! will never happen
    end select
  end do
  
  ! update the list of offsets/patterns
  n_offsets = n_offsets + 1
  known_map(n_offsets,:) = map(:)
  offset(n_offsets) = ioffset

end subroutine register_offset

!****************************************************************************
!> Take out the dummy entries and shift all real entries to the left; put a
!< zero in the remaining postitions.
!----------------------------------------------------------------------------

subroutine compact_array(iarray)

  integer, dimension(max_dims), intent(inout) :: iarray
  integer :: i, j
  
  do i = 1, max_dims
    grab_dummies : do
      if (iarray(i) == dummy_index) then
        do j = i, max_dims - 1
          iarray(j)=iarray(j+1)
        end do
        iarray(max_dims) = 0
      else
        exit grab_dummies
      end if
    end do grab_dummies
  end do

  
end subroutine compact_array

!****************************************************************************
!> not an inverse as such, this routine tells you how to call the index
!< function in some order.
!----------------------------------------------------------------------------

subroutine index_invert(starts,ends,gps_next,gps_prev,gpvpar_next,gpvpar_prev,gpmu_prev,gpmu_next,field)

  use global,  only : id_sp, id_s, id_vpar, id_mu, id_mod, id_x
  use grid,    only : nx, nmod, nvpar, ns, nmu, nsp
  use general, only : gkw_abort

  integer, intent(in), optional :: gps_next,gps_prev,gpvpar_prev,gpvpar_next,gpmu_next,gpmu_prev,field
  integer, intent(out), dimension(max_dims) :: starts, ends

  integer :: i

  starts(:)=1
  ends(:)=1
  
  !N.B. we have not considered communicated mom conserve types
  do i=1,max_dims
    ! build the default length array
    select case(index_order(i))
      case(id_vpar) 
        if (present(gpvpar_prev)) then
          ends(i) = gpvpar_prev
        else
          if (present(field)) then
            ends(i) = 1
            !stop 'wrong'
          else
            ends(i) = nvpar
          end if
        end if
        if (present(gpvpar_next)) then
          starts(i) = nvpar-gpvpar_next+1
        else
          ! do nothing
        end if
      case(id_s)
        if (present(gps_prev)) then
          ends(i) = gps_prev
        else
!          if (present(field)) then
!            ends(i) = 1
!          else
          ends(i) = ns
!          end if
        end if
        if (present(gps_next)) then
          starts(i) = ns-gps_next+1
        else
          ! do nothing
        end if
      case(id_sp) 
        if (present(field)) then
          ends(i) = 1
        else
          ends(i) = nsp
        end if
      case(id_mu)
        if (present(gpmu_prev)) then
          ends(i) = gpmu_prev
        else
          if (present(field)) then
            ends(i) = 1
          else
            ends(i) = nmu
          end if
        end if
        if (present(gpmu_next)) then
          starts(i) = nmu-gpmu_next+1
        else
          ! do nothing
        end if
      case(id_x)    ; ends(i) = nx
      case(id_mod)  ; ends(i) = nmod
      case default  ; call gkw_abort('index_invert_: bad index id')
    end select
  end do
 
end subroutine index_invert

!****************************************************************************

subroutine index_reorder_main(i1,i2,i3,i4,i5,i6,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)

  use global, only : id_sp, id_s, id_vpar, id_mu, id_mod, id_x

  integer, intent(in)  :: i1,i2,i3,i4,i5,i6
  integer, intent(out) :: i_mod,i_x,i_s,i_mu,i_vpar,i_sp

  integer, dimension(max_dims) :: ind_in
  integer :: i

  ind_in(1:6) = (/ i1,i2,i3,i4,i5,i6 /)
  do i=1,max_dims
    select case(index_order(i))
      case(id_vpar) ; i_vpar = ind_in(i)
      case(id_s)    ; i_s    = ind_in(i)
      case(id_mu)   ; i_mu   = ind_in(i)
      case(id_x)    ; i_x    = ind_in(i)
      case(id_sp)   ; i_sp   = ind_in(i)
      case(id_mod)  ; i_mod  = ind_in(i)
    end select
  end do


end subroutine index_reorder_main

!****************************************************************************

subroutine index_reorder_field(i1,i2,i3,i_mod,i_x,i_s)

  use global, only : id_sp, id_s, id_vpar, id_mu, id_mod, id_x

  integer, intent(in)  :: i1,i2,i3
  integer, intent(out) :: i_mod,i_x,i_s

  integer, dimension(max_dims) :: ind_in
  integer :: i

  ind_in(1:3) = (/ i1,i2,i3 /)
  do i=1,max_dims
    select case(index_order(i))
      case(id_vpar) 
      case(id_s)    ; i_s    = ind_in(i)
      case(id_mu)
      case(id_x)    ; i_x    = ind_in(i)
      case(id_sp)   
      case(id_mod)  ; i_mod  = ind_in(i)
    end select
  end do


end subroutine index_reorder_field

!****************************************************************************

end module index_function
