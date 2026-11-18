/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* A 32-bit offset as a sum of up to four scaled terms plus a constant, for
 * grouping memory accesses that differ by constants. */

#pragma once

#include "compiler/nir/nir.h"

#define MAX_TERMS 4

struct lin {
   unsigned n;
   nir_scalar term[MAX_TERMS];
   uint32_t scale[MAX_TERMS];
   uint32_t constant;
   bool valid;
};


static inline bool
agx_lin_is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

static inline void
lin_add_term(struct lin *l, nir_scalar s, uint32_t scale)
{
   for (unsigned i = 0; i < l->n; ++i) {
      if (l->term[i].def == s.def && l->term[i].comp == s.comp) {
         l->scale[i] += scale;
         return;
      }
   }
   if (l->n == MAX_TERMS) {
      l->valid = false;
      return;
   }
   l->term[l->n] = s;
   l->scale[l->n++] = scale;
}

static inline void
lin_parse(struct lin *l, nir_scalar s, uint32_t mult, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (!l->valid)
      return;
   if (nir_scalar_is_const(s)) {
      l->constant += mult * (uint32_t)nir_scalar_as_uint(s);
      return;
   }
   if (depth) {
      /* Constant scales: a * #c, a << #c. */
      if (agx_lin_is_op(s, nir_op_amul) || agx_lin_is_op(s, nir_op_imul) ||
          agx_lin_is_op(s, nir_op_ishl)) {
         nir_scalar a = nir_scalar_chase_alu_src(s, 0);
         nir_scalar b = nir_scalar_chase_alu_src(s, 1);
         const bool shl = agx_lin_is_op(s, nir_op_ishl);
         if (!shl && nir_scalar_is_const(a)) {
            nir_scalar t = a;
            a = b;
            b = t;
         }
         if (nir_scalar_is_const(b) && !nir_scalar_is_const(a)) {
            uint32_t k = nir_scalar_as_uint(b);
            if (shl)
               k = 1u << (k & 31);
            lin_parse(l, a, mult * k, depth - 1);
            return;
         }
      }
      if (agx_lin_is_op(s, nir_op_iadd)) {
         lin_parse(l, nir_scalar_chase_alu_src(s, 0), mult, depth - 1);
         lin_parse(l, nir_scalar_chase_alu_src(s, 1), mult, depth - 1);
         return;
      }
      if (agx_lin_is_op(s, nir_op_isub)) {
         lin_parse(l, nir_scalar_chase_alu_src(s, 0), mult, depth - 1);
         lin_parse(l, nir_scalar_chase_alu_src(s, 1), -mult, depth - 1);
         return;
      }
      if (agx_lin_is_op(s, nir_op_ineg)) {
         lin_parse(l, nir_scalar_chase_alu_src(s, 0), -mult, depth - 1);
         return;
      }
   }
   lin_add_term(l, s, mult);
}

static inline void
lin_normalize(struct lin *l)
{
   /* Drop cancelled terms and sort for comparison. */
   unsigned n = 0;
   for (unsigned i = 0; i < l->n; ++i) {
      if (l->scale[i]) {
         l->term[n] = l->term[i];
         l->scale[n++] = l->scale[i];
      }
   }
   l->n = n;
   for (unsigned i = 1; i < n; ++i) {
      for (unsigned j = i; j > 0; --j) {
         nir_scalar *a = &l->term[j - 1], *b = &l->term[j];
         if (a->def->index < b->def->index ||
             (a->def->index == b->def->index && a->comp <= b->comp))
            break;
         nir_scalar t = *a;
         *a = *b;
         *b = t;
         uint32_t u = l->scale[j - 1];
         l->scale[j - 1] = l->scale[j];
         l->scale[j] = u;
      }
   }
}

static inline bool
lin_same_terms(const struct lin *a, const struct lin *b)
{
   if (a->n != b->n)
      return false;
   for (unsigned i = 0; i < a->n; ++i) {
      if (a->term[i].def != b->term[i].def || a->term[i].comp != b->term[i].comp ||
          a->scale[i] != b->scale[i])
         return false;
   }
   return true;
}


static inline int
lin_cmp_terms(const struct lin *a, const struct lin *b)
{
   if (a->n != b->n)
      return a->n < b->n ? -1 : 1;
   for (unsigned i = 0; i < a->n; ++i) {
      if (a->term[i].def->index != b->term[i].def->index)
         return a->term[i].def->index < b->term[i].def->index ? -1 : 1;
      if (a->term[i].comp != b->term[i].comp)
         return a->term[i].comp < b->term[i].comp ? -1 : 1;
      if (a->scale[i] != b->scale[i])
         return a->scale[i] < b->scale[i] ? -1 : 1;
   }
   return 0;
}
