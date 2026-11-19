/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
extern "C" {
#include "agx_nir.h"
}

static nir_builder
compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static unsigned
count_loops(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type == nir_cf_node_loop &&
             nir_loop_first_block(nir_cf_node_as_loop(parent)) == block)
            n++;
      }
   }
   return n;
}

static nir_def *
buffer_address(nir_builder *b, nir_def *byte_offset)
{
   nir_def *base = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 16), .range = 32);
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

enum loop_shape { TOP_TESTED, BOTTOM_TESTED };

/* for (i = lo; i <= hi; i++) sum += buf[i], optionally with a barrier. */
static nir_shader *
counted_loop(loop_shape shape, nir_def **hi_out, bool barrier, int64_t hi_const)
{
   nir_builder b = compute_shader("runtime-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *lo = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *hi = hi_const == INT64_MAX
                    ? nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32)
                    : nir_imm_int(&b, (uint32_t)hi_const);
   *hi_out = hi;
   nir_store_var(&b, iv, lo, 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      if (shape == TOP_TESTED) {
         nir_break_if(&b, nir_ilt(&b, hi, i));
      }
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, i, 4)),
                                   .align_mul = 4,
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER));
      if (barrier)
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_def *next = nir_iadd_imm(&b, i, 1);
      nir_store_var(&b, iv, next, 1);
      if (shape == BOTTOM_TESTED) {
         nir_break_if(&b, nir_ilt(&b, hi, next));
      }
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);
   return b.shader;
}

TEST(M5RuntimeLoops, TopTestedUnrolls)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, false, INT64_MAX);
   ASSERT_EQ(count_loops(s), 1u);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   nir_validate_shader(s, "top-tested unroll");
   EXPECT_EQ(count_loops(s), 2u);
   /* Both loops are marked; a second run leaves them alone. */
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256));
   ralloc_free(s);
}

TEST(M5RuntimeLoops, BottomTestedUnrolls)
{
   nir_def *hi;
   nir_shader *s = counted_loop(BOTTOM_TESTED, &hi, false, INT64_MAX);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   nir_validate_shader(s, "bottom-tested unroll");
   EXPECT_EQ(count_loops(s), 2u);
   ralloc_free(s);
}

TEST(M5RuntimeLoops, BarrierRefused)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, true, INT64_MAX);
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256));
   EXPECT_EQ(count_loops(s), 1u);
   ralloc_free(s);
}

/* for (i = 0; i < n; i++) acc += op, with op what the M5 subgroup lowering
 * leaves of subgroupElect() (active-lane index == 0), of
 * subgroupBallotBitCount(subgroupBallot(true)) (the active-lane count), or of
 * subgroupQuadAny() (a quad ballot), or, as a control, a plain compare. */
enum lane_op { LANE_NONE, LANE_ELECT, LANE_BALLOT_COUNT, LANE_QUAD_VOTE };

static nir_shader *
lane_state_loop(lane_op op)
{
   nir_builder b = compute_shader("lane-state-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_def *lane_id = nir_load_local_invocation_index(&b);
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge(&b, i, n));
      nir_def *cond = nir_ult(&b, i, lane_id);
      nir_def *v = NULL;
      switch (op) {
      case LANE_NONE:
         v = nir_b2i32(&b, cond);
         break;
      case LANE_ELECT:
         v = nir_b2i32(&b, nir_ieq_imm(&b, nir_load_active_subgroup_invocation_agx(&b, 16), 0));
         break;
      case LANE_BALLOT_COUNT:
         v = nir_load_active_subgroup_count_agx(&b, 32);
         break;
      case LANE_QUAD_VOTE:
         v = nir_b2i32(&b, nir_ine_imm(&b, nir_quad_ballot_agx(&b, 16, cond), 0));
         break;
      }
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc), buffer_address(&b, nir_imm_int(&b, 0)),
                    .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "lane-state loop before");
   return b.shader;
}

/* These intrinsics carry no subgroup flag, but their result depends on which
 * invocations run them together: a main loop and a remainder run by
 * different lanes at once would elect once in each, or count and vote over
 * only part of the lanes. The loop is left alone; the control unrolls. */
TEST(M5RuntimeLoops, LaneStateRefused)
{
   nir_shader *s = lane_state_loop(LANE_NONE);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   nir_validate_shader(s, "lane-state control after");
   EXPECT_EQ(count_loops(s), 2u);
   ralloc_free(s);

   for (lane_op op : {LANE_ELECT, LANE_BALLOT_COUNT, LANE_QUAD_VOTE}) {
      s = lane_state_loop(op);
      EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256)) << "op " << op;
      EXPECT_EQ(count_loops(s), 1u) << "op " << op;
      ralloc_free(s);
   }
}

TEST(M5RuntimeLoops, BudgetRefused)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, false, INT64_MAX);
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 8));
   ralloc_free(s);
}

/* while (++i < n) acc += buf[i]: the back-edge value of i is defined in the
 * header, before the exit test. Every copy of the body must advance it. */
TEST(M5RuntimeLoops, HeaderDefinedBackEdge)
{
   nir_builder b = compute_shader("header-back-edge");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_store_var(&b, iv, nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *next = nir_iadd_imm(&b, nir_load_var(&b, iv), 1);
      nir_store_var(&b, iv, next, 1);
      nir_break_if(&b, nir_ige(&b, next, n));
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, next, 4)),
                                   .align_mul = 4);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   nir_shader *s = b.shader;
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   ASSERT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   NIR_PASS(_, s, nir_opt_algebraic);
   NIR_PASS(_, s, nir_opt_constant_folding);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "header-defined back edge");

   /* In the main (first) loop, i advances by 4 per iteration. */
   bool checked = false;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (checked || parent->type != nir_cf_node_loop ||
             nir_loop_first_block(nir_cf_node_as_loop(parent)) != block)
            continue;
         nir_block *latch = nir_loop_last_block(nir_cf_node_as_loop(parent));
         nir_foreach_phi(phi, block) {
            nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
            nir_scalar u = nir_scalar_resolved(back->src.ssa, 0);
            if (!nir_scalar_is_alu(u) || nir_scalar_alu_op(u) != nir_op_iadd)
               continue;
            for (unsigned k = 0; k < 2; ++k) {
               nir_scalar c = nir_scalar_chase_alu_src(u, k);
               nir_scalar o = nir_scalar_chase_alu_src(u, 1 - k);
               if (o.def == &phi->def && nir_scalar_is_const(c)) {
                  EXPECT_EQ(nir_scalar_as_uint(c), 4u);
                  checked = true;
               }
            }
         }
      }
   }
   EXPECT_TRUE(checked);
   ralloc_free(s);
}

/* The guard of the main loop compares the induction variable against a
 * select of a loop-invariant bound. With a constant bound it folds: check
 * that "i < G" proves i + 3 <= hi without wrapping, and that a bound too
 * close to INT32_MIN disables the main loop. */
static uint32_t
folded_guard(loop_shape shape, int64_t hi_const)
{
   nir_def *hi;
   nir_shader *s = counted_loop(shape, &hi, false, hi_const);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   NIR_PASS(_, s, nir_opt_constant_folding);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "folded guard");
   /* The first loop's first compare is the guard; read its constant. */
   uint32_t g = 0xdeadbeef;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type != nir_cf_node_loop)
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ilt && nir_src_is_const(alu->src[1].src)) {
               g = nir_src_as_uint(alu->src[1].src);
               goto done;
            }
         }
      }
   }
done:
   ralloc_free(s);
   return g;
}

TEST(M5RuntimeLoops, GuardBounds)
{
   /* Top-tested i <= hi: four iterations run while i + 3 <= hi, i < hi - 2. */
   EXPECT_EQ(folded_guard(TOP_TESTED, 100), 98u);
   EXPECT_EQ(folded_guard(TOP_TESTED, 0x7fffffff), 0x7ffffffdu);
   /* hi - 3 would wrap: never. */
   EXPECT_EQ(folded_guard(TOP_TESTED, INT32_MIN + 2), (uint32_t)INT32_MIN);
   EXPECT_EQ(folded_guard(TOP_TESTED, INT32_MIN + 3), (uint32_t)INT32_MIN + 1);
}

static nir_intrinsic_instr *
first_intrinsic(nir_shader *s, nir_intrinsic_op op, unsigned skip)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op && skip-- == 0)
               return nir_instr_as_intrinsic(instr);
         }
      }
   }
   return NULL;
}

/* Four consecutive words read in one block become one vector load. */
TEST(M5MergeWords, FourWords)
{
   nir_builder b = compute_shader("merge-words");
   nir_def *i = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *base = nir_amul_imm(&b, i, 4);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 4; ++k) {
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_iadd_imm(&b, base, 4 * k)),
                                   .align_mul = 4,
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER));
      sum = nir_iadd(&b, sum, v);
   }
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader, true));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge words");
   nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_global, 0);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->def.num_components, 4u);
   EXPECT_EQ(first_intrinsic(b.shader, nir_intrinsic_load_global, 1), nullptr);
   ralloc_free(b.shader);
}

/* Bytes X + 1 .. X + 4 become one u8x4 load at X + 1 when bytes are enabled;
 * a byte at X + 6 stays alone; nothing merges with bytes disabled. */
static nir_builder
byte_reads(void)
{
   nir_builder b = compute_shader("merge-bytes");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *sum = nir_imm_int(&b, 0);
   static const unsigned offs[] = {3, 1, 6, 4, 2};
   for (unsigned k = 0; k < ARRAY_SIZE(offs); ++k) {
      nir_def *v = nir_load_global(&b, 1, 8,
                                   buffer_address(&b, nir_iadd_imm(&b, x, offs[k])),
                                   .align_mul = 1);
      sum = nir_iadd(&b, nir_imul_imm(&b, sum, 3), nir_u2u32(&b, v));
   }
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   return b;
}

TEST(M5MergeWords, FourBytes)
{
   nir_builder b = byte_reads();
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader, true));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge bytes");
   nir_intrinsic_instr *vec = NULL, *lone = NULL;
   for (unsigned k = 0; k < 2; ++k) {
      nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_global, k);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(load->def.bit_size, 8u);
      (load->def.num_components == 4 ? vec : lone) = load;
   }
   EXPECT_EQ(first_intrinsic(b.shader, nir_intrinsic_load_global, 2), nullptr);
   ASSERT_NE(vec, nullptr);
   ASSERT_NE(lone, nullptr);
   EXPECT_EQ(lone->def.num_components, 1u);
   /* The vector starts at X + 1 and sits where X + 3 was read, the first. */
   EXPECT_EQ(first_intrinsic(b.shader, nir_intrinsic_load_global, 0), vec);
   ralloc_free(b.shader);

   b = byte_reads();
   EXPECT_FALSE(agx_nir_merge_global_words(b.shader, false));
   ralloc_free(b.shader);
}

/* Byte reads at X, X + 1 and X + 2 that execute together share X. */
TEST(M5FoldOffsets, ByteSiblings)
{
   nir_builder b = compute_shader("byte-siblings");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 3; ++k) {
      nir_def *v = nir_load_global(&b, 1, 8, buffer_address(&b, nir_iadd_imm(&b, x, k)),
                                   .align_mul = 1);
      sum = nir_iadd(&b, sum, nir_u2u32(&b, v));
   }
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_fold_m5_offsets(b.shader));
   nir_validate_shader(b.shader, "byte siblings");
   nir_def *index = NULL;
   for (unsigned k = 0; k < 3; ++k) {
      nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_agx, k);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(nir_intrinsic_base(load) >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT, k);
      EXPECT_FALSE(nir_intrinsic_sign_extend(load));
      if (index)
         EXPECT_EQ(load->src[1].ssa, index);
      index = load->src[1].ssa;
   }
   ralloc_free(b.shader);
}

/* A lone byte read at X + 1 cannot prove X + 1 does not wrap: unchanged. */
TEST(M5FoldOffsets, LoneByteKept)
{
   nir_builder b = compute_shader("lone-byte");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *v = nir_load_global(&b, 1, 8, buffer_address(&b, nir_iadd_imm(&b, x, 1)),
                                .align_mul = 1);
   nir_store_global(&b, nir_u2u32(&b, v), buffer_address(&b, nir_imm_int(&b, 64)),
                    .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_FALSE(agx_nir_fold_m5_offsets(b.shader));
   ralloc_free(b.shader);
}

/* buf[i + 3] of 32-bit elements indexes with i and a 12-byte offset. */
TEST(M5FoldOffsets, ArrayElement)
{
   nir_builder b = compute_shader("array-element");
   nir_def *i = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *v = nir_load_global(&b, 1, 32,
                                buffer_address(&b, nir_amul_imm(&b, nir_iadd_imm(&b, i, 3), 4)),
                                .align_mul = 4);
   nir_store_global(&b, v, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_fold_m5_offsets(b.shader));
   nir_validate_shader(b.shader, "array element");
   nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_agx, 0);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->src[1].ssa, i);
   EXPECT_EQ(nir_intrinsic_base(load) >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT, 12u);
   EXPECT_TRUE(nir_intrinsic_sign_extend(load));
   ralloc_free(b.shader);
}

/* |a - b| of unsigned bytes in 16-bit arithmetic: evaluate the rewritten
 * expression for every operand pair. */
static uint32_t
eval_scalar(nir_scalar s, uint8_t a, uint8_t b, nir_def *la, nir_def *lb)
{
   if (s.def == la)
      return a;
   if (s.def == lb)
      return b;
   if (nir_scalar_is_const(s))
      return nir_scalar_as_uint(s);
   EXPECT_TRUE(nir_scalar_is_alu(s));
   nir_alu_instr *alu = nir_def_as_alu(s.def);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value src[4][NIR_MAX_VEC_COMPONENTS] = {};
   nir_const_value *srcs[4];
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar in = nir_scalar_chase_alu_src(s, i);
      uint64_t v = eval_scalar(in, a, b, la, lb);
      src[i][0] = nir_const_value_for_uint(v, in.def->bit_size);
      srcs[i] = src[i];
   }
   nir_const_value dest[NIR_MAX_VEC_COMPONENTS] = {};
   uint16_t poison = 0;
   nir_eval_const_opcode(alu->op, dest, &poison, 1, alu->def.bit_size, srcs, 0);
   return nir_const_value_as_uint(dest[0], alu->def.bit_size);
}

TEST(M5U8AbsDiff, Exhaustive)
{
   for (unsigned form = 0; form < 2; ++form) {
      nir_builder b = compute_shader("u8-absdiff");
      nir_def *addr = buffer_address(&b, nir_imm_int(&b, 0));
      nir_def *la = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
      nir_def *lb = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 1), .align_mul = 1);
      nir_def *d = nir_isub(&b, nir_u2u32(&b, la), nir_u2u32(&b, lb));
      nir_def *r = form == 0 ? nir_imax(&b, d, nir_ineg(&b, d)) : nir_iabs(&b, d);
      nir_intrinsic_instr *store =
         nir_store_global(&b, r, nir_iadd_imm(&b, addr, 4), .align_mul = 4);
      NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_opt_u8_absdiff(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8 absdiff");
      nir_scalar out = nir_scalar_resolved(store->src[0].ssa, 0);
      EXPECT_NE(nir_scalar_alu_op(out), nir_op_imax);
      for (unsigned x = 0; x < 256; ++x) {
         for (unsigned y = 0; y < 256; ++y) {
            uint32_t got = eval_scalar(out, x, y, la, lb);
            ASSERT_EQ(got, (uint32_t)abs((int)x - (int)y)) << x << " " << y;
         }
      }
      ralloc_free(b.shader);
   }
}

/* A delta with another use keeps its 32-bit form. */
TEST(M5U8AbsDiff, OtherUseRefused)
{
   nir_builder b = compute_shader("u8-absdiff-uses");
   nir_def *addr = buffer_address(&b, nir_imm_int(&b, 0));
   nir_def *la = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
   nir_def *lb = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 1), .align_mul = 1);
   nir_def *d = nir_isub(&b, nir_u2u32(&b, la), nir_u2u32(&b, lb));
   nir_store_global(&b, nir_imax(&b, d, nir_ineg(&b, d)), addr, .align_mul = 4);
   nir_store_global(&b, d, nir_iadd_imm(&b, addr, 8), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_FALSE(agx_nir_opt_u8_absdiff(b.shader));
   ralloc_free(b.shader);
}

/* Written-out population counts become one native count. */
static nir_def *
pc_level(nir_builder *b, nir_def *v, uint32_t m, unsigned s)
{
   return nir_iadd(b, nir_iand_imm(b, v, m), nir_iand_imm(b, nir_ushr_imm(b, v, s), m));
}

TEST(M5PopcountIdiom, TreeAndMultiplyForms)
{
   for (unsigned form = 0; form < 2; ++form) {
      nir_builder b = compute_shader("popcount-idiom");
      nir_def *addr = buffer_address(&b, nir_imm_int(&b, 0));
      nir_def *x = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_def *r;
      if (form == 0) {
         nir_def *v = pc_level(&b, x, 0x55555555, 1);
         v = pc_level(&b, v, 0x33333333, 2);
         v = pc_level(&b, v, 0x0f0f0f0f, 4);
         v = pc_level(&b, v, 0x00ff00ff, 8);
         r = pc_level(&b, v, 0x0000ffff, 16);
      } else {
         nir_def *v = nir_isub(&b, x, nir_iand_imm(&b, nir_ushr_imm(&b, x, 1), 0x55555555));
         v = pc_level(&b, v, 0x33333333, 2);
         v = nir_iand_imm(&b, nir_iadd(&b, v, nir_ushr_imm(&b, v, 4)), 0x0f0f0f0f);
         r = nir_ushr_imm(&b, nir_imul_imm(&b, v, 0x01010101), 24);
      }
      nir_intrinsic_instr *store =
         nir_store_global(&b, r, nir_iadd_imm(&b, addr, 4), .align_mul = 4);
      EXPECT_TRUE(agx_nir_opt_popcount_idiom(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "popcount idiom");
      nir_scalar out = nir_scalar_resolved(store->src[0].ssa, 0);
      ASSERT_TRUE(nir_scalar_is_alu(out));
      EXPECT_EQ(nir_scalar_alu_op(out), nir_op_bit_count);
      /* The original expression's value equals the count everywhere. */
      uint32_t v = 0x9e3779b9;
      for (unsigned i = 0; i < 100000; ++i) {
         v = v * 1664525u + 1013904223u;
         uint32_t t = v - ((v >> 1) & 0x55555555u);
         t = (t & 0x33333333u) + ((t >> 2) & 0x33333333u);
         t = (((t + (t >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
         ASSERT_EQ(t, (uint32_t)__builtin_popcount(v));
      }
      ralloc_free(b.shader);
   }
}
