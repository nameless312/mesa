/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
init_cmd(struct hk_device *dev, struct hk_cmd_buffer *cmd, enum agx_chip chip,
         bool robust)
{
   memset(dev, 0, sizeof(*dev));
   memset(cmd, 0, sizeof(*cmd));
   dev->dev.chip = chip;
   dev->vk.enabled_features.robustBufferAccess = robust;
   dev->vk.enabled_features.robustBufferAccess2 = robust;
   cmd->vk.base.device = &dev->vk;
}

int
main(void)
{
   struct hk_device dev;
   struct hk_cmd_buffer cmd;

   /* G17C with robustness enabled: aligned 16/32-bit indirect draws within
    * 32-bit byte bounds use native hardware bounds; oversized, unaligned, and
    * 8-bit indirect draws take the software rewrite path. */
   init_cmd(&dev, &cmd, AGX_CHIP_G17C, true);

   struct agx_draw ind_u32 = agx_draw_indexed_indirect(
      UINT64_C(0x200000000), UINT64_C(0x100000000), UINT32_C(0xfffffffc),
      AGX_INDEX_SIZE_U32, false);
   assert(!hk_needs_index_robustness(&cmd, &ind_u32, NULL));

   struct agx_draw ind_u16 = agx_draw_indexed_indirect(
      UINT64_C(0x200000000), UINT64_C(0x100000000), UINT32_C(0xfffffffc),
      AGX_INDEX_SIZE_U16, true);
   assert(!hk_needs_index_robustness(&cmd, &ind_u16, NULL));

   struct agx_draw ind_u16_unaligned = agx_draw_indexed_indirect(
      UINT64_C(0x200000000), UINT64_C(0x100000000), 6, AGX_INDEX_SIZE_U16,
      false);
   assert(hk_needs_index_robustness(&cmd, &ind_u16_unaligned, NULL));

   struct agx_draw ind_u8 = agx_draw_indexed_indirect(
      UINT64_C(0x200000000), UINT64_C(0x100000000), 16, AGX_INDEX_SIZE_U8,
      false);
   assert(hk_needs_index_robustness(&cmd, &ind_u8, NULL));

   struct agx_draw ind_4gib = agx_draw_indexed_indirect(
      UINT64_C(0x200000000), UINT64_C(0x100000000), UINT64_C(1) << 32,
      AGX_INDEX_SIZE_U32, false);
   assert(hk_needs_index_robustness(&cmd, &ind_4gib, NULL));

   struct agx_draw ind_16gib = agx_draw_indexed_indirect(
      UINT64_C(0x200000000), UINT64_C(0x100000000), UINT64_C(1) << 34,
      AGX_INDEX_SIZE_U32, false);
   assert(hk_needs_index_robustness(&cmd, &ind_16gib, NULL));

   /* G17P keeps software rewrite for robust indirect draws. */
   init_cmd(&dev, &cmd, AGX_CHIP_G17P, true);
   assert(hk_needs_index_robustness(&cmd, &ind_u32, NULL));

   /* Oversized (> UINT32_MAX * 4) indirect and conditionally lowered direct
    * draws must always lower through the software rewrite even when no
    * robustness features are enabled, including a 1-point U32 draw at
    * firstIndex = 0 with a 16 GiB + 4 binding. */
   for (unsigned r = 0; r < 2; ++r) {
      const bool robust = r != 0;
      for (unsigned c = 0; c < 2; ++c) {
         const enum agx_chip chip = c == 0 ? AGX_CHIP_G17C : AGX_CHIP_G17P;
         init_cmd(&dev, &cmd, chip, robust);

         const uint64_t range_16gib_p4 = (UINT64_C(1) << 34) + 4;
         struct agx_draw cond_direct = agx_draw_indexed(
            1, 1, 0, 0, 0, UINT64_C(0x100000000), range_16gib_p4,
            AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), range_16gib_p4,
            AGX_INDEX_SIZE_U32, false);
         assert(hk_needs_index_robustness(&cmd, &cond_indirect, &cond_direct));
         assert(hk_needs_index_robustness(&cmd, &cond_indirect, NULL));

         const uint64_t range_16gib = UINT64_C(1) << 34;
         struct agx_draw cond_direct_16g = agx_draw_indexed(
            3, 1, 0, 0, 0, UINT64_C(0x100000000), range_16gib,
            AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect_16g = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), range_16gib,
            AGX_INDEX_SIZE_U32, false);
         assert(
            hk_needs_index_robustness(&cmd, &cond_indirect_16g, &cond_direct_16g));

         /* In-bounds encodable binding (4 GiB) skips rewrite. */
         struct agx_draw cond_direct_4g = agx_draw_indexed(
            1, 1, 0, 0, 0, UINT64_C(0x100000000), UINT64_C(1) << 32,
            AGX_INDEX_SIZE_U32, false);
         struct agx_draw cond_indirect_4g = agx_draw_indexed_indirect(
            UINT64_C(0x200000000), UINT64_C(0x100000000), UINT64_C(1) << 32,
            AGX_INDEX_SIZE_U32, false);
         assert(
            !hk_needs_index_robustness(&cmd, &cond_indirect_4g, &cond_direct_4g));
      }
   }

   puts("hk_index_robustness: ok");
   return 0;
}
