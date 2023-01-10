/*
 * Copyright (c) 2010-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @generated d Mon Nov 14 16:23:41 2022
 *
 */

#include "dplasma.h"
#include "dplasma/types.h"
#include "dplasma/types_lapack.h"
#include "dplasmaaux.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#if defined(DPLASMA_HAVE_CUDA)
#include "parsec/mca/device/cuda/device_cuda.h"
#endif
#include "utils/dplasma_info.h"

#include "dgemm_NN.h"
#include "dgemm_NT.h"
#include "dgemm_TN.h"
#include "dgemm_TT.h"

#include "dgemm_NN_summa.h"
#include "dgemm_NT_summa.h"
#include "dgemm_TN_summa.h"
#include "dgemm_TT_summa.h"

#define MAX_SHAPES 3

#include "dgemm_NN_gpu.h"

#include "parsec/utils/mca_param.h"

static parsec_taskpool_t *
dplasma_Zgemm_New_summa(dplasma_enum_t transA, dplasma_enum_t transB,
                        double alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                        double beta,  parsec_tiled_matrix_t* C,
                        dplasma_info_t opt)
{
    int P, Q, IP, JQ, m, n;
    parsec_taskpool_t *dgemm_tp;
    parsec_matrix_block_cyclic_t *Cdist;

    P = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
    Q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;
    IP = ((parsec_matrix_block_cyclic_t*)C)->grid.ip;
    JQ = ((parsec_matrix_block_cyclic_t*)C)->grid.jq;

    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
    dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

    m = dplasma_imax(C->mt, P);
    n = dplasma_imax(C->nt, Q);

    /* Create a copy of the C matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Cdist = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    parsec_matrix_block_cyclic_init(
            Cdist, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            C->super.myrank,
            1, 1, /* Dimensions of the tiles              */
            m, n, /* Dimensions of the matrix             */
            0, 0, /* Starting points (not important here) */
            m, n, /* Dimensions of the submatrix          */
            P, Q, 1, 1, IP, JQ);
    Cdist->super.super.data_of = NULL;
    Cdist->super.super.data_of_key = NULL;

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "dgemm_NN_summa\n");
            parsec_dgemm_NN_summa_taskpool_t* tp;
            tp = parsec_dgemm_NN_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
            dgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "dgemm_NT_summa\n");
            parsec_dgemm_NT_summa_taskpool_t* tp;
            tp = parsec_dgemm_NT_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
            dgemm_tp = (parsec_taskpool_t*)tp;
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "dgemm_TN_summa\n");
            parsec_dgemm_TN_summa_taskpool_t* tp;
            tp = parsec_dgemm_TN_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
            dgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "dgemm_TT_summa\n");
            parsec_dgemm_TT_summa_taskpool_t* tp;
            tp = parsec_dgemm_TT_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C,
                                           (parsec_data_collection_t*)Cdist);
            dgemm_tp = (parsec_taskpool_t*)tp;
        }
    }

    int shape = 0;
    dplasma_setup_adtt_all_loc( ddc_A,
                                parsec_datatype_double_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    dplasma_setup_adtt_all_loc( ddc_B,
                                parsec_datatype_double_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    dplasma_setup_adtt_all_loc( ddc_C,
                                parsec_datatype_double_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    assert(shape == MAX_SHAPES);

    parsec_dgemm_NN_summa_taskpool_t *taskpool = (parsec_dgemm_NN_summa_taskpool_t *)dgemm_tp;
    dplasma_add2arena_tile( &taskpool->arenas_datatypes[PARSEC_dgemm_NN_summa_DEFAULT_ADT_IDX],
                            C->mb*C->nb*sizeof(double),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_t, C->mb );

    (void)opt; //No user-defined options for this algorithm
    return dgemm_tp;

    (gdb) p __parsec_tp->super.arenas_datatypes[0]
$5 = {arena = 0x0, opaque_dtt = 0x0, ht_item = {next_item = 0x0, hash64 = 0, key = 0}}
(gdb) p __parsec_tp->super.arenas_datatypes[1]
$6 = {arena = 0x1, opaque_dtt = 0x5, ht_item = {next_item = 0x0, hash64 = 42949672961, key = 4294967297}}
(gdb) p __parsec_tp->super.arenas_datatypes[2]
}

static parsec_taskpool_t *
dplasma_Zgemm_New_default(dplasma_enum_t transA, dplasma_enum_t transB,
                          double alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                          double beta,  parsec_tiled_matrix_t* C,
                          dplasma_info_t opt)
{
    parsec_taskpool_t* dgemm_tp;

    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
    dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            parsec_dgemm_NN_taskpool_t* tp;
            tp = parsec_dgemm_NN_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            dgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            parsec_dgemm_NT_taskpool_t* tp;
            tp = parsec_dgemm_NT_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            dgemm_tp = (parsec_taskpool_t*)tp;
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            parsec_dgemm_TN_taskpool_t* tp;
            tp = parsec_dgemm_TN_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            dgemm_tp = (parsec_taskpool_t*)tp;
        }
        else {
            parsec_dgemm_TT_taskpool_t* tp;
            tp = parsec_dgemm_TT_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            dgemm_tp = (parsec_taskpool_t*)tp;
        }
    }

    int shape = 0;
    dplasma_setup_adtt_all_loc( ddc_A,
                                parsec_datatype_double_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    dplasma_setup_adtt_all_loc( ddc_B,
                                parsec_datatype_double_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    dplasma_setup_adtt_all_loc( ddc_C,
                                parsec_datatype_double_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);

    assert(shape == MAX_SHAPES);

    (void)opt; //No user-defined options for this algorithm
    return dgemm_tp;
}

#if defined(DPLASMA_HAVE_CUDA)
static parsec_taskpool_t*
dplasma_Zgemm_New_gpu( dplasma_enum_t transA, dplasma_enum_t transB,
                       double alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                       double beta,  parsec_tiled_matrix_t* C,
                       dplasma_info_t opt)
{

    parsec_taskpool_t* dgemm_tp = NULL;
    int64_t gpu_mem_nb_blocks = -1;
    size_t gpu_mem_block_size = 0;
    size_t tile_size;
    int64_t nb_block_per_tile, nb_tile_per_gpu;
    int mt, nt, kt;
    int info_found;
    char info_value[DPLASMA_MAX_INFO_VAL];
    double vd;

    int *dev_index, nbgpu, dev;
    int u, v;
    int M, Mbound, Mlim;
    int N, Nbound, Nlim;
    int K;

    int b, c, d, p, q, look_ahead;

    if( dplasmaNoTrans != transA || dplasmaNoTrans != transB ) {
        dplasma_error("dplasma_Zgemm_New_gpu", "NoTrans for A or B not implemented yet in JDF for GPUs");
        return NULL;
    }

    nbgpu = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)device;
            nbgpu++;
            if( 0 == gpu_mem_block_size )
                gpu_mem_block_size = cuda_device->super.mem_block_size;
            if( -1 == gpu_mem_nb_blocks || cuda_device->super.mem_nb_blocks < gpu_mem_nb_blocks )
                gpu_mem_nb_blocks = cuda_device->super.mem_nb_blocks;
        }
    }
    if(nbgpu == 0) {
        dplasma_error("dplasma_Zgemm_gpu_New", "Trying to instantiate JDF for GPUs on machine without GPUs");
        return NULL;
    }
    dev_index = (int*)malloc(nbgpu * sizeof(int));
    nbgpu= 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[nbgpu++] = device->device_index;
        }
    }

    p = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
    q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;

    vd = 1.0; // Default percentage of available memory dedicated to this GEMM
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:mem_ratio", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        vd = strtod(info_value, NULL);
        if(vd <= 0.0 || vd > 1.0) {
            dplasma_error("dplasma_Zgemm_New_gpu",
                          "Invalid value for DPLASMA:GEMM:GPU:mem_ratio. Mem ratio must be real in ]0, 1]");
            goto cleanup;
        }
    }
    tile_size = A->mb*A->nb*sizeof(double);
    nb_block_per_tile = (tile_size + gpu_mem_block_size -1 ) / gpu_mem_block_size;
    gpu_mem_nb_blocks = vd * gpu_mem_nb_blocks;
    nb_tile_per_gpu = gpu_mem_nb_blocks / nb_block_per_tile;

    mt = A->mt;
    nt = B->nt;
    kt = A->nt;

    // We find (b, c) such that b*c tiles of C fill at most 75% of the GPU memory
    // and b*p divides MT
    // and c*q divides NT
    vd = 0.75; // By default it's up to 75% of the memory to host C
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:c_ratio", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        vd = strtod(info_value, NULL);
        if(vd <= 0.0 || vd >= 1.0) {
            dplasma_error("dplasma_Zgemm_New_gpu",
                          "Invalid value for DPLASMA:GEMM:GPU:c_ratio. Ratio of memory dedicated to hosting tiles of C must be real in ]0, 1[");
            goto cleanup;
        }
    }
    int fact = 1;
    while( fact < mt && fact < nt && ((mt/fact) * (nt/fact)) / (p * q * nbgpu) > nb_tile_per_gpu * vd ) fact++;
    b = mt/(p*fact);
    c = nt/(q*fact);

    // Usually, look ahead is detrimental to performance when fact=1
    // and critical to performance when fact > 1
    look_ahead = 1 + (fact > 1);
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:look_ahead", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        look_ahead = atoi(info_value);
        if(look_ahead <= 0) {
            dplasma_error("dplasma_Zgemm_New_gpu",
                          "Invalid value for DPLASMA:GEMM:GPU:look_ahead. Look ahead must be 1 or more");
            goto cleanup;
        }
    }

    // OK, now we fill up each GPU with data from A and B
    int c_per_gpu = c / nbgpu;
    int maxd = (nb_tile_per_gpu - b*c_per_gpu)/(b+c_per_gpu) - 1;
    d = maxd < kt ? maxd : kt;

    // Now we let the user overwrite the b, c and d parameters
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:b", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        b = atoi(info_value);
        if(b <= 0 || b*p > A->mt) {
            dplasma_error("dplasma_Zgemm_New_gpu",
                          "Invalid value for DPLASMA:GEMM:GPU:b. b must be > 0 and b*P less or equal to A.mt");
            goto cleanup;
        }
    }
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:c", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        c = atoi(info_value);
        if(c <= 0 || c*q > A->nt) {
            dplasma_error("dplasma_Zgemm_New_gpu",
                          "Invalid value for DPLASMA:GEMM:GPU:c. c must be > 0 and c*Q less or equal to A.nt");
            goto cleanup;
        }
    }
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:d", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        d = atoi(info_value);
        if(d <= 0 || d > B->mt) {
            dplasma_error("dplasma_Zgemm_New_gpu",
                          "Invalid value for DPLASMA:GEMM:GPU:d. d must be > 0 and less or equal to B.mt");
            goto cleanup;
        }
    }

    assert(d <= B->mt);
    assert( b*p <= A->mt );
    assert( c*q <= C->nt );

    {
        parsec_dgemm_NN_gpu_taskpool_t *tp;
        tp = parsec_dgemm_NN_gpu_new(transA, transB, alpha, beta,
                                     A, B, C, b, c, d, p, q, look_ahead,
                                     nbgpu, dev_index);

        u = C->super.myrank / q;
        v = C->super.myrank % q;

        M = A->mt;
        Mbound = M / (p * b);
        Mlim = p * b * Mbound + u;
        tp->_g_xMax = Mbound + (Mlim < M) - 1;

        N = C->nt;
        Nbound = N / (c * q);
        Nlim = c * q * Nbound + v;
        tp->_g_yMax = Nbound + (Nlim < N) - 1;

        K = B->mt;
        tp->_g_zMax = (K + d - 1) / d - 1;

        dgemm_tp = (parsec_taskpool_t *) tp;

        return dgemm_tp;
    }

  cleanup:
    if(NULL != dev_index)
        free(dev_index);
    return NULL;
}
#endif /* DPLASMA_HAVE_CUDA */

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  dplasma_dgemm_New - Generates the taskpool that performs one of the following
 *  matrix-matrix operations. WARNING: The computations are not done by this call.
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = g( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or ugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaTrans: A is ugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or ugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaTrans: B is ugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_dgemm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_dgemm
 * @sa dplasma_dgemm_Destruct
 * @sa dplasma_cgemm_New
 * @sa dplasma_dgemm_New
 * @sa dplasma_sgemm_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_dgemm_New_ex( dplasma_enum_t transA, dplasma_enum_t transB,
                      double alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                      double beta,  parsec_tiled_matrix_t* C, dplasma_info_t opt)
{
    parsec_taskpool_t* dgemm_tp = NULL;

    /* Check input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaTrans)) {
        dplasma_error("dplasma_dgemm_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaTrans)) {
        dplasma_error("dplasma_dgemm_New", "illegal value of transB");
        return NULL /*-2*/;
    }

    if ( C->dtype & parsec_matrix_block_cyclic_type ) {
#if defined(DPLASMA_HAVE_CUDA)
        int nb_gpu_devices = 0, devid;
		int p = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
	    int q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;
		int64_t gpu_mem_block_size = 0;
		int64_t gpu_mem_nb_blocks = -1;
        for(devid = 0; devid < (int)parsec_nb_devices; devid++) {
            parsec_device_module_t *device = parsec_mca_device_get(devid);
            if( PARSEC_DEV_CUDA == device->type ) {
				parsec_device_cuda_module_t *cuda_device = (parsec_device_cuda_module_t*)device;
                nb_gpu_devices++;
				if( 0 == gpu_mem_block_size )
				    gpu_mem_block_size = cuda_device->super.mem_block_size;
				if( -1 == gpu_mem_nb_blocks || cuda_device->super.mem_nb_blocks < gpu_mem_nb_blocks )
				    gpu_mem_nb_blocks = cuda_device->super.mem_nb_blocks;
            }
        }
        if(0 < nb_gpu_devices) {
            int64_t tile_size = A->mb*A->nb*sizeof(double);
            int64_t nb_block_per_tile = (tile_size + gpu_mem_block_size -1 ) / gpu_mem_block_size;
            int64_t nb_tile_per_gpu = gpu_mem_nb_blocks / nb_block_per_tile;
            int64_t nb_active_tiles_per_gpu = C->mt * C->nt / (p*q) + dplasma_aux_getGEMMLookahead(C) * A->mt / p + dplasma_aux_getGEMMLookahead(C) * B->nt / q;
            if( (A->dtype & parsec_matrix_block_cyclic_type) &&
                (B->dtype & parsec_matrix_block_cyclic_type) &&
                transA == dplasmaNoTrans &&
                transB == dplasmaNoTrans &&
                (nb_active_tiles_per_gpu > 0.95* nb_tile_per_gpu) ) {
                dgemm_tp = dplasma_Zgemm_New_gpu(transA, transB, alpha, A, B, beta, C, opt);
                return dgemm_tp;
            }
        }
#endif /* DPLASMA_HAVE_CUDA */
        dgemm_tp = dplasma_Zgemm_New_summa(transA, transB, alpha, A, B, beta, C, opt);
        return dgemm_tp;
    }
    dgemm_tp = dplasma_Zgemm_New_default(transA, transB, alpha, A, B, beta, C, opt);
    return dgemm_tp;
}

parsec_taskpool_t*
dplasma_dgemm_New( dplasma_enum_t transA, dplasma_enum_t transB,
                   double alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                   double beta,  parsec_tiled_matrix_t* C)
{
    parsec_taskpool_t *tp;
    dplasma_info_t opt;
    dplasma_info_create(&opt);
    tp = dplasma_dgemm_New_ex(transA, transB, alpha, A, B, beta, C, opt);
    dplasma_info_free(&opt);
    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  dplasma_dgemm_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_dgemm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_dgemm_New
 * @sa dplasma_dgemm
 *
 ******************************************************************************/
void
dplasma_dgemm_Destruct( parsec_taskpool_t *tp )
{
    parsec_dgemm_NN_taskpool_t *dgemm_tp = (parsec_dgemm_NN_taskpool_t *)tp;
    dplasma_data_collection_t *ddc_A = NULL, *ddc_B = NULL, *ddc_C = NULL;

    if( dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_NN_SUMMA ||
        dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_NT_SUMMA ||
        dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_TN_SUMMA ||
        dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_TT_SUMMA) {
        parsec_dgemm_NN_summa_taskpool_t *dgemm_summa_tp = (parsec_dgemm_NN_summa_taskpool_t *)tp;
        parsec_tiled_matrix_t* Cdist = (parsec_tiled_matrix_t*)dgemm_summa_tp->_g_Cdist;
        if ( NULL != Cdist ) {
            parsec_tiled_matrix_destroy( Cdist );
            free( Cdist );
        }
        dplasma_clean_adtt_all_loc(dgemm_summa_tp->_g_ddescA, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(dgemm_summa_tp->_g_ddescB, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(dgemm_summa_tp->_g_ddescC, MAX_SHAPES);

        ddc_A = dgemm_summa_tp->_g_ddescA;
        ddc_B = dgemm_summa_tp->_g_ddescB;
        ddc_C = dgemm_summa_tp->_g_ddescC;
    } else if( dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_NN ||
               dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_NT ||
               dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_TN ||
               dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_TT) {
        dplasma_clean_adtt_all_loc(dgemm_tp->_g_ddescA, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(dgemm_tp->_g_ddescB, MAX_SHAPES);
        dplasma_clean_adtt_all_loc(dgemm_tp->_g_ddescC, MAX_SHAPES);

        ddc_A = dgemm_tp->_g_ddescA;
        ddc_B = dgemm_tp->_g_ddescB;
        ddc_C = dgemm_tp->_g_ddescC;
#if defined(DPLASMA_HAVE_CUDA)
    } else if( dgemm_tp->_g_gemm_type == DPLASMA_DGEMM_NN_GPU ) {
        parsec_dgemm_NN_gpu_taskpool_t *dgemm_gpu_tp = (parsec_dgemm_NN_gpu_taskpool_t *)tp;
        free(dgemm_gpu_tp->_g_cuda_device_index);
#endif /* DPLASMA_HAVE_CUDA */
    }

    parsec_taskpool_free(tp);

    /* free the dplasma_data_collection_t, after the tp stops referring to them */
    if(NULL != ddc_A)
        dplasma_unwrap_data_collection(ddc_A);
    if(NULL != ddc_B)
        dplasma_unwrap_data_collection(ddc_B);
    if(NULL != ddc_C)
        dplasma_unwrap_data_collection(ddc_C);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  dplasma_dgemm - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = g( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or ugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaTrans: A is ugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or ugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaTrans: B is ugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_dgemm_New
 * @sa dplasma_dgemm_Destruct
 * @sa dplasma_cgemm
 * @sa dplasma_dgemm
 * @sa dplasma_sgemm
 *
 ******************************************************************************/
int
dplasma_dgemm( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               double alpha, const parsec_tiled_matrix_t *A,
                                        const parsec_tiled_matrix_t *B,
               double beta,        parsec_tiled_matrix_t *C)
{
    parsec_taskpool_t *parsec_dgemm = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    /* Check input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaTrans)) {
        dplasma_error("dplasma_dgemm", "illegal value of transA");
        return -1;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaTrans)) {
        dplasma_error("dplasma_dgemm", "illegal value of transB");
        return -2;
    }

    if ( transA == dplasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    } else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    if ( transB == dplasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    } else {
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_dgemm", "tile sizes have to match");
        return -101;
    }
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_dgemm", "sizes of matrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_dgemm", "start indexes have to match");
        return -101;
    }

    M = C->m;
    N = C->n;
    K = An;

    /* Quick return */
    if (M == 0 || N == 0 ||
        ((alpha == (double)0.0 || K == 0) && beta == (double)1.0))
        return 0;

    parsec_dgemm = dplasma_dgemm_New(transA, transB,
                                    alpha, A, B,
                                    beta, C);

    if ( parsec_dgemm != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_dgemm);
        dplasma_wait_until_completion(parsec);
        dplasma_dgemm_Destruct( parsec_dgemm );
        return 0;
    }
    return -101;
}
