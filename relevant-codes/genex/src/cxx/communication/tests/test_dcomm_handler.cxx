#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "dcomm_handler.hxx"
#if defined(ENABLE_OPENACC)
#include "dcomm_handler_acc.hxx"
#endif
#if defined(ENABLE_OPENMPX)
#include "dcomm_handler_ompx.hxx"
#endif
#include "dcomm_handler_omp.hxx"
#include "mpi.h"
#include "assert.h"
//using ::testing::ElementsAreArray;

class Environment : public ::testing::Environment {
 public:
  ~Environment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {
    int ierr=MPI_Init(NULL,NULL);
  }

  // Override this to define how to tear down the environment.
  void TearDown() override;
};

void Environment::TearDown() {
  int ierr = MPI_Finalize();
}

template <typename T>
class DCommHandlerTest : public testing::Test {
 protected:
  DCommHandlerTest() {
    dcomm_handler_data_t chd;
    int ierr;

    chd.comm_base=MPI_Comm_c2f(MPI_COMM_WORLD);
    /*chd.comm_cart;
      chd.comm_phi;
      chd.comm_vp;
      chd.comm_mu;
      chd.comm_sp;
      chd.comm_phi_sp;
      chd.comm_vp_mu;
      chd.comm_phi_vp_mu;
      chd.comm_vp_mu_sp;
    */
    chd.n_dims=1;
    ierr=MPI_Comm_size(MPI_COMM_WORLD,&(chd.n_procs_total));
    assert(ierr==MPI_SUCCESS);
    chd.n_procs_RZ=1;
    chd.n_procs_phi=1;
    chd.n_procs_vp=1;
    chd.n_procs_mu=1;
    chd.n_procs_sp=1;
    ierr=MPI_Comm_rank(MPI_COMM_WORLD,&(chd.rank));
    assert(ierr==MPI_SUCCESS);

    /* we should use a factory method for creation */
    ch = new T{&chd};
  }

  T *ch;
};

using testing::Types;
#if defined(ENABLE_OPENACC)
typedef Types<dcomm_handler_acc_t,dcomm_handler_omp_t> Implementations;
#elif defined(ENABLE_OPENMPX)
typedef Types<dcomm_handler_ompx_t,dcomm_handler_omp_t> Implementations;
#else
typedef Types<dcomm_handler_omp_t> Implementations;
#endif

TYPED_TEST_SUITE(DCommHandlerTest, Implementations);

TYPED_TEST(DCommHandlerTest, Creation) {
  int rank,ierr;

  ASSERT_EQ(1,this->ch->get_n_dims());
  ierr=MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  ASSERT_EQ(0,ierr);
  ASSERT_EQ(rank,this->ch->get_rank());
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  testing::Environment *env=
    testing::AddGlobalTestEnvironment(new Environment);
  return RUN_ALL_TESTS();
}
