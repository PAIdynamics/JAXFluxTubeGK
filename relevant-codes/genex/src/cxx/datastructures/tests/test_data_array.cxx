#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include "genex_cxx_env.hxx"
#include "data_array.hxx"
#ifdef ENABLE_OPENACC
#include "data_array_acc.hxx"
#elif defined(ENABLE_OMPX)
#include "data_array_ompx.hxx"
#else
#include "data_array_omp.hxx"
#endif

using ::testing::ElementsAreArray;

TEST(DataArray,Creation) {
  data_array_data_t dad;
  data_array_t<real_t, 2> *da;
  int32_t array_shape[2]={3,5};
  int32_t array_lb[2]={0,0};
  int32_t array_ub[2]={2,4};
  std::vector<real_t> array(15);
  int i=0;
  for (auto val : array) {
    val=i;
    i++;
  }

  dad.array_dim = 2;
  dad.array_size = 15;
  dad.array_size_stripped = 15;
  dad.rank_id = 0;
  dad.is_distributed_array = 0;
  dad.array_shape_ptr=array_shape;
  dad.array_shape_stripped_ptr=array_shape;
  dad.array_lb_ptr=array_lb;
  dad.array_ub_ptr=array_ub;
  dad.array_lb_stripped_ptr=array_lb;
  dad.array_ub_stripped_ptr=array_ub;
  dad.array_ptr=array.data();

  /* we should use a factory method for creation */
#if defined(ENABLE_OPENACC)
  da=new data_array_acc_t<real_t, 2>{&dad};
#elif defined(ENABLE_OPENMPX)
  da=new data_array_ompx_t<real_t, 2>{&dad};
#else
  da=new data_array_omp_t<real_t, 2>{&dad};
#endif
  EXPECT_THAT(std::vector<real_t>(da->get_array_ptr(),
                                  da->get_array_ptr()+da->get_size()),
                                  ElementsAreArray(array));
}
