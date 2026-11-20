/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P loads do not see a host write made while their command runs, so the
 * poll of a vkCmdWaitEvents2 on an event the host sets later must start in an
 * ioctl sent after vkSetEvent. Under the no-op DRM shim, which logs every
 * submit ioctl (ASAHI_SHIM_SUBMIT_LOG), a fill, a wait and a fill go as:
 *  - host-only source, event reset: one ioctl before vkSetEvent, and one
 *    after it that starts with the poll;
 *  - host-only source, event already set: one ioctl;
 *  - a device-set event (vkCmdSetEvent2, compute source): one ioctl;
 *  - a fill command buffer, then one that opens with the host-only wait:
 *    the same two ioctls, the poll does not continue the fill's command;
 *  - vkDestroyDevice with a host-only wait whose set never comes returns.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "FAIL: %s: %d\n", #call, r_);                         \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static char log_path[256];

/* Submit ioctls logged so far; *first_cmd: the first hardware command of the
 * last one. */
static unsigned
submits(char first_cmd[16])
{
   FILE *f = fopen(log_path, "r");
   if (!f)
      return 0;
   unsigned n = 0;
   bool want = false;
   char line[512];
   while (fgets(line, sizeof(line), f)) {
      if (!strncmp(line, "submit ", 7)) {
         n++;
         want = true;
      } else if (want && (!strncmp(line, "  cmd compute", 13) ||
                          !strncmp(line, "  cmd render", 12))) {
         if (first_cmd)
            snprintf(first_cmd, 16, "%s", line[6] == 'c' ? "compute" : "render");
         want = false;
      }
   }
   fclose(f);
   return n;
}

static void
sleep_ms(unsigned ms)
{
   const struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
   nanosleep(&t, NULL);
}

static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkBuffer buffer;

static VkCommandBuffer
record(VkEvent event, bool host, bool lead)
{
   const VkCommandBufferAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &alloc, &cb));
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkBeginCommandBuffer(cb, &begin));
   if (lead)
      vkCmdFillBuffer(cb, buffer, 0, 256, 1);
   if (event == VK_NULL_HANDLE) {
      /* Flushed, the fill's command could take the next command buffer's
       * first stream (hk_cs_cdm_may_continue()). */
      const VkMemoryBarrier2 flush = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
      };
      const VkDependencyInfo flush_dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &flush,
      };
      vkCmdPipelineBarrier2(cb, &flush_dep);
      CHECK(vkEndCommandBuffer(cb));
      return cb;
   }

   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = host ? VK_PIPELINE_STAGE_2_HOST_BIT
                           : VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      .srcAccessMask = host ? VK_ACCESS_2_HOST_WRITE_BIT
                            : VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };
   if (!host)
      vkCmdSetEvent2(cb, event, &dep);
   vkCmdWaitEvents2(cb, 1, &event, &dep);
   vkCmdFillBuffer(cb, buffer, 256, 256, 2);
   CHECK(vkEndCommandBuffer(cb));
   return cb;
}

/* The ioctls since before, once count are there or 5 s passed, and 50 ms
 * more. The submit thread sends them; the shim's syncobjs do not wait, so
 * vkQueueWaitIdle() would not wait for that thread. */
static unsigned
settle(unsigned before, unsigned count)
{
   unsigned n = 0;
   for (unsigned t = 0; t < 5000 && (n = submits(NULL) - before) < count; ++t)
      sleep_ms(1);
   sleep_ms(50);
   return submits(NULL) - before;
}

static void
submit_n(const VkCommandBuffer *cbs, uint32_t count)
{
   const VkSubmitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = count,
      .pCommandBuffers = cbs,
   };
   CHECK(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
}

static void
submit(VkCommandBuffer cb)
{
   submit_n(&cb, 1);
}

/* A host-only wait set after the submission: one ioctl before the set,
 * a second one starting with the poll after it. */
static int
check_held_back(const char *what, const VkCommandBuffer *cbs, uint32_t count,
                VkEvent event)
{
   int fails = 0;
   CHECK(vkResetEvent(dev, event));
   unsigned before = submits(NULL);
   submit_n(cbs, count);
   unsigned n = settle(before, 1);
   if (n != 1) {
      fprintf(stderr, "FAIL: %s: %u ioctls before vkSetEvent, expected 1\n",
              what, n);
      fails++;
   }
   CHECK(vkSetEvent(dev, event));
   n = settle(before, 2);
   char first[16] = "";
   submits(first);
   if (n != 2 || strcmp(first, "compute")) {
      fprintf(stderr, "FAIL: %s: %u ioctls after vkSetEvent, the last "
                      "starting with %s, expected 2 starting with compute\n",
              what, n, first[0] ? first : "nothing");
      fails++;
   }
   return fails;
}

int
main(void)
{
   snprintf(log_path, sizeof(log_path), "/tmp/hk-host-event-wait-%d.log",
            (int)getpid());
   unlink(log_path);
   setenv("ASAHI_SHIM_SUBMIT_LOG", log_path, 1);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice pdev;
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count == 0) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = 0,
   };
   CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 4096,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   CHECK(vkCreateBuffer(dev, &buffer_info, NULL, &buffer));
   VkMemoryRequirements reqs;
   vkGetBufferMemoryRequirements(dev, buffer, &reqs);
   uint32_t type = 0;
   while (type < 32 && !(reqs.memoryTypeBits & (1u << type)))
      type++;
   const VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = type,
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc_info, NULL, &memory));
   CHECK(vkBindBufferMemory(dev, buffer, memory, 0));

   const VkEventCreateInfo event_info = {
      .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
   };
   VkEvent event;
   CHECK(vkCreateEvent(dev, &event_info, NULL, &event));
   int fails = 0;

   /* Host-only source, the host sets the event after the submission. */
   VkCommandBuffer host_cb = record(event, true, true);
   fails += check_held_back("host wait", &host_cb, 1, event);

   /* The same with the event already set: nothing to hold back. */
   unsigned before = submits(NULL);
   submit(host_cb);
   unsigned n = settle(before, 1);
   if (n != 1) {
      fprintf(stderr, "FAIL: host wait on a set event: %u ioctls, "
                      "expected 1\n", n);
      fails++;
   }

   /* A device-set event is not held back. */
   CHECK(vkResetEvent(dev, event));
   VkCommandBuffer device_cb = record(event, false, true);
   before = submits(NULL);
   submit(device_cb);
   n = settle(before, 1);
   if (n != 1) {
      fprintf(stderr, "FAIL: device-set event wait: %u ioctls, expected 1\n",
              n);
      fails++;
   }

   /* A poll opening a command buffer does not continue the command of the
    * one before. */
   const VkCommandBuffer pair[2] = {record(VK_NULL_HANDLE, false, true),
                                    record(event, true, false)};
   fails += check_held_back("host wait after a fill", pair, 2, event);

   /* A wait whose set never comes ends with the device. */
   CHECK(vkResetEvent(dev, event));
   before = submits(NULL);
   submit(host_cb);
   settle(before, 1);
   /* The command buffer is still pending: keep what it uses. */
   alarm(20);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   unlink(log_path);

   if (!fails)
      printf("PASS: a host-only event wait starts after vkSetEvent\n");
   return fails ? 1 : 0;
}
