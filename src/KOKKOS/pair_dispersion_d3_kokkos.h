#ifdef PAIR_CLASS
//clang-format off
PairStyle(dispersion/d3/kk, PairDispersionD3Kokkos<LMPDeviceType>);
PairStyle(dispersion/d3/device, PairDispersionD3Kokkos<LMPDeviceType>);
PairStyle(dispersion/d3/host, PairDispersionD3Kokkos<LMPHostType>);
#else

#ifndef LMP_PAIR_DISPERSION_D3_KOKKOS_H
#define LMP_PAIR_DISPERSION_D3_KOKKOS_H

#include "pair_dispersion_d3.h"
#include "kokkos_type.h"
#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS{

// Communication tags
struct TagPairDD3PackForwardCommCN{};
struct TagPairDD3PackForwardCommDC6{};
struct TagPairDD3UnpackForwardCommCN{};
struct TagPairDD3UnpackForwardCommDC6{};
struct TagPairDD3PackReverseCommCN{};
struct TagPairDD3PackReverseCommDC6{};
struct TagPairDD3UnpackReverseCommCN{};
struct TagPairDD3UnpackReverseCommDC6{};

// Computation tags
struct TagPairDD3KokkosCNDC6Initialise{};
//struct TagPairDD3KokkosCNDC6Calc{};
template<int NEIGHFLAG>
struct TagPairDD3KokkosCNDC6Kernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJOriginalZeroDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJModifiedZeroDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJOriginalBJDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdIJModifiedBJDampKernel{};

template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
struct TagPairDispDD3dEdXYZKernel{};

template<class DeviceType>
class PairDispersionD3Kokkos : public PairDispersionD3, public KokkosBase
{
  public:
    enum {EnabledNeighFlags=HALF};
    typedef ArrayTypes<DeviceType> AT;
    typedef DeviceType device_type;
    //typename AT::t_x_array_const_um x;
    //typename AT::t_f_array_const_um f;
    typedef EV_FLOAT value_type;
   
    PairDispersionD3Kokkos(class LAMMPS *);
    ~PairDispersionD3Kokkos();
    
    void compute(int eflag, int vflag) override; 
    void coeff(int, char**) override;
    void init_style() override;  
    //void settings(int, char**) override;
    //double init_one(int i, int j) override;  
    void sync_arrays_device(); 
     // Main computation operators
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedZeroDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJOriginalBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdIJModifiedBJDampKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int &ii) const; 
    
    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdXYZKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int& ii, EV_FLOAT& ev) const; 

    template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDispDD3dEdXYZKernel<NEIGHFLAG, NEWTON_PAIR, EVFLAG>, const int& ii) const; 
    
    // Coordination number calculation operators
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3KokkosCNDC6Initialise, const int &i) const;
   
    template<int NEIGHFLAG>
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3KokkosCNDC6Kernel<NEIGHFLAG>, const int &ii) const;
  
    // C6 coefficient calculation
    KOKKOS_INLINE_FUNCTION
    void dC6KK(int itype, int jtype,
                   double cni, double cnj, 
                   double &C6, double &dC6_i, double &dC6_j) const;
    
    // Communication operators
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackForwardCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackForwardCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackForwardCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackForwardCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackReverseCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3PackReverseCommDC6, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackReverseCommCN, const int &i) const;
    KOKKOS_INLINE_FUNCTION
    void operator()(TagPairDD3UnpackReverseCommDC6, const int &i) const;

    // Communication interface methods
    int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_kkfloat_1d&, int, int*) override; 
    //int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d&, int, int*) override;
    void unpack_forward_comm_kokkos(int, int, DAT::tdual_kkfloat_1d&) override;
    int pack_reverse_comm_kokkos(int, int, DAT::tdual_kkfloat_1d&) override;
    void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_kkfloat_1d&) override;
    
    int pack_forward_comm(int, int*, double*, int, int*) override;
    void unpack_forward_comm(int, int, double*) override;
    int pack_reverse_comm(int, int, double*) override;
    void unpack_reverse_comm(int, int*, double*) override;

  protected:
    // non-view variables 
    bool initialised;
    //int need_dup, dampingCode;
    int need_dup;
    using PairDispersionD3::dampingCode;
    int communicationStage;
    int first;
    int nlocal, nall, inum, nmax, eflag, vflag;
    int newton_pair;
    // kokkos execution spaces 
    ExecutionSpace execution_space;
    using KKDeviceType = typename KKDevice<DeviceType>::value;
 

    // Kokkos views 
 
    typename AT::t_kkfloat_1d_3_lr x;
    typename AT::t_kkacc_1d_3      f;
    typename AT::t_int_1d          type;
  
    DAT::ttransform_kkacc_1d   k_eatom;
    DAT::ttransform_kkacc_1d_6 k_vatom;
    typename AT::t_kkacc_1d    d_eatom;
    typename AT::t_kkacc_1d_6  d_vatom;
  
    // neighbors views 
    typename AT::t_neighbors_2d d_neighbors; 
    typename AT::t_int_1d       d_ilist;
    typename AT::t_int_1d       d_numneigh;
 
    // Comm views 
    typename AT::t_int_1d d_sendlistV;
    typename AT::t_double_1d_um bufV;
    
    
    template<typename DataType, typename Layout>
    using DupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;
      
    template<typename DataType, typename Layout>
    using NonDupScatterView = KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;
    
    DupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout>              dup_cn;
    DupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout>              dup_dc6;
    DupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout>       dup_f;
    DupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout>            dup_eatom;
    DupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout>       dup_vatom; 
    NonDupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout>           ndup_cn;
    NonDupScatterView<KK_FLOAT*, typename DAT::t_kkfloat_1d::array_layout>           ndup_dc6;
    NonDupScatterView<KK_ACC_FLOAT*[3], typename DAT::t_kkacc_1d_3::array_layout>    ndup_f; 
    NonDupScatterView<KK_ACC_FLOAT*, typename DAT::t_kkacc_1d::array_layout>         ndup_eatom;
    NonDupScatterView<KK_ACC_FLOAT*[6], typename DAT::t_kkacc_1d_6::array_layout>    ndup_vatom;

    DAT::tdual_kkfloat_1d      k_cn_v;
    DAT::tdual_kkfloat_1d     k_dc6_v;
    typename AT::t_kkfloat_1d  d_cn_v;
    typename AT::t_kkfloat_1d d_dc6_v;
     

    typedef Kokkos::DualView<KK_FLOAT**[5][5][3], DeviceType>  tdual_kkfloat_5d; 
    typedef typename tdual_kkfloat_5d::t_dev_const             t_kkfloat_5d;
    typedef typename tdual_kkfloat_5d::t_dev_const_randomread  t_kkfloat_5d_ra;
    typedef typename tdual_kkfloat_5d::t_host_const_randomread t_host_kkfloat_5d_ra;

    tdual_kkfloat_5d          k_c6ab_v;
    t_kkfloat_5d              d_c6ab_v;
    t_kkfloat_5d_ra           d_c6ab_ra;


    DAT::tdual_kkfloat_1d     k_mxci_v;
    DAT::tdual_kkfloat_1d     k_r2r4_v;
    DAT::tdual_kkfloat_1d     k_rcov_v;
    DAT::tdual_kkfloat_2d     k_r0ab_v;
    DAT::tdual_kkfloat_2d     k_cutsq_v;
    typename AT::t_kkfloat_1d d_mxci_v;
    typename AT::t_kkfloat_1d d_r2r4_v;
    typename AT::t_kkfloat_1d d_rcov_v;
    typename AT::t_kkfloat_2d d_r0ab_v;
    typename AT::t_kkfloat_2d d_cutsq_v;

    typename AT::t_kkfloat_1d_randomread d_rcov_ra;
    typename AT::t_kkfloat_1d_randomread d_r2r4_ra;
    typename AT::t_kkfloat_1d_randomread d_mxci_ra;
    typename AT::t_kkfloat_2d_randomread d_r0ab_ra;
    typename AT::t_kkfloat_2d_randomread d_cutsq_ra; 
  
    // Function declarations 

    void calc_coordination_numbersKK();
    friend void pair_virial_fdotr_compute<PairDispersionD3Kokkos>(PairDispersionD3Kokkos*);
    
    template<int NEIGHFLAG, int NEWTON_PAIR>
    KOKKOS_INLINE_FUNCTION
    void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
                  const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

    // Tag structs 
    template<class TAG>
    struct policyInstance;


};
}
#endif
#endif

