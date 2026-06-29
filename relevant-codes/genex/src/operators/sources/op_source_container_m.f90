module op_source_container_m
    !! Module containing op source container

    use op_source_m, only: op_source_t

    implicit none
    private

    type, public :: op_source_container_t
        !! Type that contains a collection of source operators
        class(op_source_t), allocatable :: op_source
        !! Abstract type of the source operator
    end type

end module
