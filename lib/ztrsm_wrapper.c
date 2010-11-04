/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "dplasma.h"

#include "generated/ztrsm_LLN.h"
#include "generated/ztrsm_LLT.h"
#include "generated/ztrsm_LUN.h"
#include "generated/ztrsm_LUT.h"
#include "generated/ztrsm_RLN.h"
#include "generated/ztrsm_RLT.h"
#include "generated/ztrsm_RUN.h"
#include "generated/ztrsm_RUT.h"

dague_object_t *
dplasma_ztrsm_New(const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
                  const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_trsm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_LLN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_LLT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_LUN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_LUT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_RLN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_RLT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_RUN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_RUT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, 
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        }
    }

    printf("TRSM A:%ux%u (%ux%u) and B:%ux%u (%ux%u) has %u tasks to run.\n",
           A->m, A->n, A->mt, A->nt,
           B->m, B->n, B->mt, B->nt,
           dague_trsm->nb_local_tasks);

    return dague_trsm;
}

void
dplasma_ztrsm_Destruct( dague_object_t *o )
{
    int side  = ((dague_ztrsm_LLN_object_t *)o)->side;
    int uplo  = ((dague_ztrsm_LLN_object_t *)o)->uplo;
    int trans = ((dague_ztrsm_LLN_object_t *)o)->trans;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_LLN_destroy((dague_ztrsm_LLN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_LLT_destroy((dague_ztrsm_LLT_object_t *)o);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_LUN_destroy((dague_ztrsm_LUN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_LUT_destroy((dague_ztrsm_LUT_object_t *)o);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_RLN_destroy((dague_ztrsm_RLN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_RLT_destroy((dague_ztrsm_RLT_object_t *)o);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_RUN_destroy((dague_ztrsm_RUN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_RUT_destroy((dague_ztrsm_RUT_object_t *)o);
            }
        }
    }
}

void
dplasma_ztrsm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrsm = NULL;

    /* two_dim_block_cyclic_t work; */
    /* /\* Create workspace for control *\/ */
    /* two_dim_block_cyclic_init(&work, matrix_Integer, B->super.nodes, B->super.cores, B->super.myrank, */
    /*                           1, 1, B->mt, B->nt, 0, 0, B->mt, B->nt, 1, 1, ((two_dim_block_cyclic_t*)B)->GRIDrows); */

    dague_ztrsm = dplasma_ztrsm_New(side, uplo, trans, diag, alpha,
				    A, B); /*, (tiled_matrix_desc_t *)&work);*/

    dague_enqueue( dague, dague_ztrsm );
    dague_progress( dague );

    dplasma_ztrsm_Destruct( dague_ztrsm );
    /* dague_data_free(&work.mat); */
}
