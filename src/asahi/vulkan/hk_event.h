/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"

#include "vk_object.h"

struct hk_event {
   struct vk_object_base base;
   struct agx_bo *bo;

   uint64_t addr;
   VkResult *status;

   /* Host sets, under hk_device::host_event.lock. */
   uint32_t host_sets;
};

struct hk_device;

bool hk_event_host_unset(struct hk_device *dev, struct hk_event *event,
                         uint32_t *sets);
bool hk_event_wait_host_set(struct hk_device *dev, struct hk_event *event,
                            uint32_t sets);

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)
