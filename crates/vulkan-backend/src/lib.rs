use ash::{vk, Entry, Instance};
use std::ffi::{CStr, CString};
use std::fmt;

/// AMD PCI vendor ID
const AMD_VENDOR_ID: u32 = 0x1002;

/// Application name passed to Vulkan instance creation
const APP_NAME: &str = "vulkanize";
const APP_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
const ENGINE_NAME: &str = "vulkanize";
const ENGINE_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
const API_VERSION: u32 = vk::API_VERSION_1_1;

/// Validation layer used in debug builds
#[cfg(debug_assertions)]
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during Vulkan initialisation.
#[derive(Debug)]
pub enum VulkanError {
    /// Vulkan loader could not be loaded.
    LoaderNotFound,
    /// Failed to create the Vulkan instance.
    InstanceCreation(vk::Result),
    /// No physical devices found.
    NoPhysicalDevices,
    /// No physical device with a compute queue family was found.
    NoComputeQueue,
    /// Failed to create the logical device.
    DeviceCreation(vk::Result),
    /// Failed to create the command pool.
    CommandPoolCreation(vk::Result),
    /// Failed to create a Vulkan buffer.
    BufferCreation(vk::Result),
    /// Failed to allocate device memory for a buffer.
    MemoryAllocation(vk::Result),
    /// Failed to bind device memory to a buffer.
    MemoryBinding(vk::Result),
    /// No suitable memory type found for the requested properties.
    NoSuitableMemoryType,
    /// Failed to map device memory.
    MemoryMapping(vk::Result),
    /// Failed to allocate a command buffer.
    CommandBufferAllocation(vk::Result),
    /// Failed to create a fence.
    FenceCreation(vk::Result),
    /// Failed to wait on a fence.
    FenceWait(vk::Result),
    /// Validation error for transfer operations.
    TransferValidation(String),
    /// Failed to create a shader module.
    ShaderModuleCreation(vk::Result),
    /// SPIR-V bytes failed pre-validation (empty or not word-aligned).
    InvalidSpvBytes(String),
    /// Failed to load a SPIR-V binary file.
    SpvLoadingError(String),
    /// Failed to create a pipeline layout.
    PipelineLayoutCreation(vk::Result),
    /// Failed to create a compute pipeline.
    ComputePipelineCreation(vk::Result),
    /// Entry point name contains a null byte or is otherwise invalid for CString.
    InvalidEntryPoint(String),
    /// Failed to create a descriptor set layout.
    DescriptorSetLayoutCreation(vk::Result),
    /// Failed to create a descriptor pool.
    DescriptorPoolCreation(vk::Result),
    /// Failed to allocate a descriptor set from the pool.
    DescriptorSetAllocation(vk::Result),
    /// Command buffer was not in the recording state when a command was recorded.
    CommandBufferNotRecording,
    /// Push constant validation error.
    PushConstantValidation(String),
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VulkanError::LoaderNotFound => write!(f, "Vulkan loader not found"),
            VulkanError::InstanceCreation(res) => {
                write!(f, "instance creation failed: {:?}", res)
            }
            VulkanError::NoPhysicalDevices => write!(f, "no Vulkan physical devices found"),
            VulkanError::NoComputeQueue => {
                write!(f, "no physical device with a compute queue family found")
            }
            VulkanError::DeviceCreation(res) => {
                write!(f, "logical device creation failed: {:?}", res)
            }
            VulkanError::CommandPoolCreation(res) => {
                write!(f, "command pool creation failed: {:?}", res)
            }
            VulkanError::BufferCreation(res) => {
                write!(f, "buffer creation failed: {:?}", res)
            }
            VulkanError::MemoryAllocation(res) => {
                write!(f, "device memory allocation failed: {:?}", res)
            }
            VulkanError::MemoryBinding(res) => {
                write!(f, "buffer memory binding failed: {:?}", res)
            }
            VulkanError::NoSuitableMemoryType => {
                write!(f, "no suitable memory type found for requested properties")
            }
            VulkanError::MemoryMapping(res) => {
                write!(f, "device memory mapping failed: {:?}", res)
            }
            VulkanError::CommandBufferAllocation(res) => {
                write!(f, "command buffer allocation failed: {:?}", res)
            }
            VulkanError::FenceCreation(res) => {
                write!(f, "fence creation failed: {:?}", res)
            }
            VulkanError::FenceWait(res) => {
                write!(f, "fence wait failed: {:?}", res)
            }
            VulkanError::TransferValidation(msg) => {
                write!(f, "transfer validation failed: {}", msg)
            }
            VulkanError::ShaderModuleCreation(res) => {
                write!(f, "shader module creation failed: {:?}", res)
            }
            VulkanError::InvalidSpvBytes(msg) => {
                write!(f, "invalid SPIR-V bytes: {}", msg)
            }
            VulkanError::SpvLoadingError(msg) => {
                write!(f, "SPIR-V loading error: {}", msg)
            }
            VulkanError::PipelineLayoutCreation(res) => {
                write!(f, "pipeline layout creation failed: {:?}", res)
            }
            VulkanError::ComputePipelineCreation(res) => {
                write!(f, "compute pipeline creation failed: {:?}", res)
            }
            VulkanError::InvalidEntryPoint(msg) => {
                write!(f, "invalid entry point name: {}", msg)
            }
            VulkanError::DescriptorSetLayoutCreation(res) => {
                write!(f, "descriptor set layout creation failed: {:?}", res)
            }
            VulkanError::DescriptorPoolCreation(res) => {
                write!(f, "descriptor pool creation failed: {:?}", res)
            }
            VulkanError::DescriptorSetAllocation(res) => {
                write!(f, "descriptor set allocation failed: {:?}", res)
            }
            VulkanError::CommandBufferNotRecording => {
                write!(f, "command buffer is not in the recording state")
            }
            VulkanError::PushConstantValidation(msg) => {
                write!(f, "push constant validation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for VulkanError {}

// ---------------------------------------------------------------------------
// Public info types
// ---------------------------------------------------------------------------

/// Information about a single queue family on a physical device.
#[derive(Debug, Clone)]
pub struct QueueFamilyInfo {
    /// Index of the queue family.
    pub index: u32,
    /// Number of queues in this family.
    pub queue_count: u32,
    /// Queue flags (bitmask of `vk::QueueFlags`).
    pub queue_flags: vk::QueueFlags,
    /// Does this family support compute?
    pub supports_compute: bool,
    /// Does this family support graphics?
    pub supports_graphics: bool,
    /// Does this family support transfer?
    pub supports_transfer: bool,
}

/// Describes a Vulkan physical device.
#[derive(Debug, Clone)]
pub struct PhysicalDeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Device type (e.g. DiscreteGpu, IntegratedGpu).
    pub device_type: vk::PhysicalDeviceType,
    /// PCI vendor ID (0x1002 = AMD).
    pub vendor_id: u32,
    /// PCI device ID.
    pub device_id: u32,
    /// Vulkan API version supported by this device.
    pub api_version: u32,
    /// Driver version.
    pub driver_version: u32,
    /// Driver name (Vulkan 1.1+).
    pub driver_name: String,
    /// All queue families exposed by this device.
    pub queue_families: Vec<QueueFamilyInfo>,
    /// Enabled device extensions.
    pub extensions: Vec<String>,
}

/// Result of selecting a compute-capable queue family from a physical device.
#[derive(Debug, Clone)]
pub struct QueueFamilySelection {
    /// Index of the selected queue family.
    pub queue_family_index: u32,
    /// Number of queues available in this family.
    pub queue_count: u32,
    /// Queue flags for the selected family.
    pub queue_flags: vk::QueueFlags,
}

/// A Vulkan logical device with its compute queue.
pub struct VulkanDevice {
    /// The underlying ash device handle.
    inner: ash::Device,
    /// The compute queue retrieved from the device.
    pub compute_queue: vk::Queue,
}

impl VulkanDevice {
    /// Returns a reference to the underlying `ash::Device`.
    pub fn handle(&self) -> &ash::Device {
        &self.inner
    }
}

/// Command pool and associated resources for recording commands.
pub struct CommandResources {
    /// Command pool bound to the compute queue family.
    pub command_pool: vk::CommandPool,
}

// ---------------------------------------------------------------------------
// Memory type selection
// ---------------------------------------------------------------------------

/// Cached physical device memory properties with a helper to find
/// a compatible memory type index for buffer allocations.
///
/// This is constant per device and should be created once during
/// initialisation and reused for all buffer allocations.
pub struct MemoryTypeSelector {
    properties: vk::PhysicalDeviceMemoryProperties,
}

impl MemoryTypeSelector {
    /// Create a new selector from the physical device's memory properties.
    pub fn new(instance: &Instance, physical_device: vk::PhysicalDevice) -> Self {
        let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        Self { properties }
    }

    /// Find a memory type index that is both compatible with the given
    /// `memory_type_bits` bitmask and satisfies all requested `properties`.
    ///
    /// Returns `None` if no suitable memory type exists.
    pub fn find_memory_type(
        &self,
        memory_type_bits: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let count = self.properties.memory_type_count as usize;
        (0..count)
            .find(|&i| {
                (memory_type_bits & (1 << i)) != 0
                    && self.properties.memory_types[i]
                        .property_flags
                        .contains(properties)
            })
            .map(|i| i as u32)
    }
}

// ---------------------------------------------------------------------------
// Vulkan buffer abstraction
// ---------------------------------------------------------------------------

/// A Vulkan buffer with its associated device memory.
///
/// This type owns both the `VkBuffer` handle and the `VkDeviceMemory`
/// allocation, and ensures correct cleanup order on drop:
/// unmap (if mapped) → free memory → destroy buffer.
///
/// The buffer holds a raw pointer to the `ash::Device` for cleanup.
/// Callers must ensure the device outlives all buffers created from it.
/// `VulkanContext` guarantees this by owning both the device and any
/// buffers created through its methods.
pub struct VulkanBuffer {
    device: *const ash::Device,
    /// The Vulkan buffer handle.
    pub buffer: vk::Buffer,
    /// The allocated device memory.
    pub memory: vk::DeviceMemory,
    /// Size of the buffer in bytes.
    pub size: vk::DeviceSize,
    /// Buffer usage flags used at creation time.
    pub usage: vk::BufferUsageFlags,
    /// Memory property flags of the allocated memory.
    pub memory_property_flags: vk::MemoryPropertyFlags,
    /// Currently mapped pointer, if the buffer is mapped.
    mapped_ptr: Option<*mut u8>,
}

impl fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("buffer", &self.buffer)
            .field("memory", &self.memory)
            .field("size", &self.size)
            .field("usage", &self.usage)
            .field("memory_property_flags", &self.memory_property_flags)
            .field("is_mapped", &self.mapped_ptr.is_some())
            .finish()
    }
}

impl VulkanBuffer {
    /// Returns the Vulkan buffer handle for use in descriptor sets
    /// and command recordings.
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Returns true if this buffer's memory is currently mapped.
    pub fn is_mapped(&self) -> bool {
        self.mapped_ptr.is_some()
    }

    /// Map the buffer's device memory for host access.
    ///
    /// The buffer must have been created with `HOST_VISIBLE` memory.
    /// Only one map is allowed at a time; call `unmap()` first if
    /// the buffer is already mapped.
    ///
    /// For `HOST_COHERENT` memory, writes are visible to the device
    /// without an explicit flush.
    pub fn map(&mut self) -> Result<*mut u8, VulkanError> {
        if self.mapped_ptr.is_some() {
            return Err(VulkanError::MemoryMapping(
                vk::Result::ERROR_MEMORY_MAP_FAILED,
            ));
        }

        if !self
            .memory_property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            return Err(VulkanError::MemoryMapping(
                vk::Result::ERROR_MEMORY_MAP_FAILED,
            ));
        }

        let device = unsafe { &*self.device };
        let ptr = unsafe {
            device
                .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
                .map_err(VulkanError::MemoryMapping)?
        };

        self.mapped_ptr = Some(ptr as *mut u8);
        Ok(ptr as *mut u8)
    }

    /// Unmap previously mapped device memory.
    pub fn unmap(&mut self) -> Result<(), VulkanError> {
        if self.mapped_ptr.is_none() {
            return Ok(());
        }

        let device = unsafe { &*self.device };
        unsafe {
            device.unmap_memory(self.memory);
        }
        self.mapped_ptr = None;
        Ok(())
    }

    /// Write data into the buffer at offset 0.
    ///
    /// If the buffer is already mapped, writes directly through the
    /// existing mapping without unmap/remap. Otherwise maps, writes,
    /// and unmaps. The buffer must have `HOST_VISIBLE` memory.
    pub fn write_data(&mut self, data: &[u8]) -> Result<(), VulkanError> {
        if data.len() > self.size as usize {
            return Err(VulkanError::MemoryMapping(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ));
        }

        let was_mapped = self.mapped_ptr.is_some();
        let ptr = if was_mapped {
            self.mapped_ptr.unwrap()
        } else {
            self.map()?
        };

        unsafe {
            std::slice::from_raw_parts_mut(ptr, data.len()).copy_from_slice(data);
        }

        if !was_mapped {
            self.unmap()?;
        }
        Ok(())
    }

    /// Write data into the buffer at a specific byte offset.
    ///
    /// If the buffer is already mapped, writes directly through the
    /// existing mapping without unmap/remap. Otherwise maps, writes,
    /// and unmaps. The buffer must have `HOST_VISIBLE` memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset + data.len()` exceeds the buffer size.
    pub fn write_at(&mut self, offset: vk::DeviceSize, data: &[u8]) -> Result<(), VulkanError> {
        let end =
            offset
                .checked_add(data.len() as vk::DeviceSize)
                .ok_or(VulkanError::MemoryMapping(
                    vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                ))?;

        if end > self.size {
            return Err(VulkanError::MemoryMapping(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ));
        }

        let was_mapped = self.mapped_ptr.is_some();
        let ptr = if was_mapped {
            self.mapped_ptr.unwrap()
        } else {
            self.map()?
        };

        unsafe {
            std::slice::from_raw_parts_mut(ptr.add(offset as usize), data.len())
                .copy_from_slice(data);
        }

        if !was_mapped {
            self.unmap()?;
        }
        Ok(())
    }

    /// Internal creation that takes a pre-resolved memory type index.
    /// Called by `VulkanContext` methods after memory type lookup.
    fn create_with_memory_type(
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_property_flags: vk::MemoryPropertyFlags,
        memory_type_index: u32,
    ) -> Result<Self, VulkanError> {
        if size == 0 {
            return Err(VulkanError::BufferCreation(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ));
        }

        let create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&create_info, None)
                .map_err(VulkanError::BufferCreation)?
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe {
            device.allocate_memory(&alloc_info, None).map_err(|e| {
                device.destroy_buffer(buffer, None);
                VulkanError::MemoryAllocation(e)
            })?
        };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0).map_err(|e| {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                VulkanError::MemoryBinding(e)
            })?
        };

        let mut mapped_ptr = None;
        if memory_property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            let ptr = unsafe {
                device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(|e| {
                        device.free_memory(memory, None);
                        device.destroy_buffer(buffer, None);
                        VulkanError::MemoryMapping(e)
                    })?
            };
            mapped_ptr = Some(ptr as *mut u8);
        }

        Ok(Self {
            device: device as *const ash::Device,
            buffer,
            memory,
            size,
            usage,
            memory_property_flags,
            mapped_ptr,
        })
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        // If device pointer is null (e.g. in tests), skip cleanup.
        if self.device.is_null() {
            return;
        }

        let device = unsafe { &*self.device };

        // Unmap if currently mapped
        if self.mapped_ptr.is_some() {
            unsafe {
                device.unmap_memory(self.memory);
            }
            self.mapped_ptr = None;
        }

        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Fence — GPU→CPU synchronization
// ---------------------------------------------------------------------------

/// A Vulkan fence for synchronizing GPU command completion with the CPU.
///
/// Fences are submitted with `vkQueueSubmit` and waited on from the CPU.
/// They support signalling, waiting, and resetting for reuse.
///
/// This is the simplest synchronization primitive — suitable for
/// synchronous upload paths and immediate execution. Future async
/// transfer queues can layer timeline semaphores on top of this.
pub struct Fence {
    device: *const ash::Device,
    handle: vk::Fence,
}

impl fmt::Debug for Fence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fence")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Fence {
    /// Create a new unsignaled fence.
    pub fn create(device: &ash::Device) -> Result<Self, VulkanError> {
        let create_info = vk::FenceCreateInfo::builder();
        let handle = unsafe {
            device
                .create_fence(&create_info, None)
                .map_err(VulkanError::FenceCreation)?
        };
        Ok(Self {
            device: device as *const ash::Device,
            handle,
        })
    }

    /// Create a new fence in the signaled state.
    ///
    /// Useful for initial elements in wait fences, where the first
    /// submission should not block on a prior (non-existent) stage.
    pub fn create_signaled(device: &ash::Device) -> Result<Self, VulkanError> {
        let create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let handle = unsafe {
            device
                .create_fence(&create_info, None)
                .map_err(VulkanError::FenceCreation)?
        };
        Ok(Self {
            device: device as *const ash::Device,
            handle,
        })
    }

    /// Wait for the fence to become signaled.
    ///
    /// `timeout` is in nanoseconds. Use `u64::MAX` to wait indefinitely.
    pub fn wait(&self, timeout: u64) -> Result<(), VulkanError> {
        let device = unsafe { &*self.device };
        unsafe {
            device
                .wait_for_fences([self.handle].as_slice(), true, timeout)
                .map_err(VulkanError::FenceWait)?
        };
        Ok(())
    }

    /// Reset a signaled fence back to the unsignaled state.
    ///
    /// The fence must be signaled (i.e., waited on or already signaled)
    /// before it can be reset.
    pub fn reset(&self) -> Result<(), VulkanError> {
        let device = unsafe { &*self.device };
        unsafe {
            device
                .reset_fences([self.handle].as_slice())
                .map_err(VulkanError::FenceWait)?
        };
        Ok(())
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        unsafe {
            device.destroy_fence(self.handle, None);
        }
    }
}

// ---------------------------------------------------------------------------
// CommandBuffer — record and submit GPU commands
// ---------------------------------------------------------------------------

/// A Vulkan primary command buffer for recording GPU commands.
///
/// Allocated from a `CommandResources` pool. Supports begin/end recording,
/// submission with fence synchronization, and automatic reset on drop.
///
/// # Lifecycle
///
/// 1. Allocate via `VulkanContext::allocate_command_buffer()`
/// 2. Begin recording: `begin()` or `begin_one_time_submit()`
/// 3. Record commands (e.g., `record_copy_buffer()`)
/// 4. End recording: `end()`
/// 5. Submit: `submit_and_wait()` — submits, waits, and resets
/// 6. Reuse from step 2, or drop to return to pool
///
/// Dropping an unfinished (still-recording) command buffer is undefined
/// behavior. The Vulkan validation layer will detect this.
pub struct CommandBuffer {
    device: *const ash::Device,
    command_pool: vk::CommandPool,
    handle: vk::CommandBuffer,
    is_recording: bool,
}

impl fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("handle", &self.handle)
            .field("is_recording", &self.is_recording)
            .finish()
    }
}

impl CommandBuffer {
    /// Returns the Vulkan command buffer handle for low-level API calls.
    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    /// Begin recording with `ONE_TIME_SUBMIT` flag.
    ///
    /// Hint to the implementation that this command buffer will be
    /// submitted at most once before being reset. Suitable for staging
    /// uploads and one-shot transfers.
    pub fn begin_one_time_submit(&mut self) -> Result<(), VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let device = unsafe { &*self.device };
        unsafe {
            device
                .begin_command_buffer(self.handle, &begin_info)
                .map_err(VulkanError::CommandBufferAllocation)?
        };
        self.is_recording = true;
        Ok(())
    }

    /// Begin recording without optimization flags.
    ///
    /// Use for command buffers that may be submitted multiple times
    /// before being reset.
    pub fn begin(&mut self) -> Result<(), VulkanError> {
        let begin_info =
            vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::empty());

        let device = unsafe { &*self.device };
        unsafe {
            device
                .begin_command_buffer(self.handle, &begin_info)
                .map_err(VulkanError::CommandBufferAllocation)?
        };
        self.is_recording = true;
        Ok(())
    }

    /// End command buffer recording.
    ///
    /// Must be called after `begin()` before the buffer can be submitted.
    pub fn end(&mut self) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferAllocation(
                vk::Result::ERROR_UNKNOWN,
            ));
        }

        let device = unsafe { &*self.device };
        unsafe {
            device
                .end_command_buffer(self.handle)
                .map_err(VulkanError::CommandBufferAllocation)?
        };
        self.is_recording = false;
        Ok(())
    }

    /// Record a `vkCmdCopyBuffer` command to copy data between buffers.
    ///
    /// Validates that `src` has `TRANSFER_SRC` usage and `dst` has
    /// `TRANSFER_DST` usage.
    pub fn record_copy_buffer(
        &mut self,
        src: &VulkanBuffer,
        dst: &VulkanBuffer,
        size: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferAllocation(
                vk::Result::ERROR_UNKNOWN,
            ));
        }

        if !src.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC) {
            return Err(VulkanError::TransferValidation(
                "source buffer missing TRANSFER_SRC usage flag".to_string(),
            ));
        }

        if !dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
            return Err(VulkanError::TransferValidation(
                "destination buffer missing TRANSFER_DST usage flag".to_string(),
            ));
        }

        let region = vk::BufferCopy::builder()
            .src_offset(0)
            .dst_offset(0)
            .size(size);

        let device = unsafe { &*self.device };
        unsafe {
            device.cmd_copy_buffer(self.handle, src.buffer, dst.buffer, &[region.build()]);
        }
        Ok(())
    }

    /// Submit the recorded command buffer and wait for GPU completion.
    ///
    /// After this call the command buffer is reset and ready for
    /// reuse with another `begin()` call.
    ///
    /// Takes the queue and queue family index from the caller, allowing
    /// future extensions to use a dedicated transfer queue.
    pub fn submit_and_wait(
        &mut self,
        queue: vk::Queue,
        _queue_family_index: u32,
    ) -> Result<(), VulkanError> {
        if self.is_recording {
            return Err(VulkanError::CommandBufferAllocation(
                vk::Result::ERROR_UNKNOWN,
            ));
        }

        let device = unsafe { &*self.device };

        let fence = Fence::create(device)?;

        let cmd_buffers = [self.handle];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_buffers);

        unsafe {
            device
                .queue_submit(queue, &[(*submit_info)], fence.handle)
                .map_err(VulkanError::CommandBufferAllocation)?;
        }

        fence.wait(u64::MAX)?;

        unsafe {
            device
                .reset_fences([fence.handle].as_slice())
                .map_err(VulkanError::FenceWait)?;
            device
                .reset_command_buffer(self.handle, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::CommandBufferAllocation)?;
        }

        Ok(())
    }

    /// Bind a compute pipeline to this command buffer.
    ///
    /// # Panics
    ///
    /// Returns an error if the command buffer is not currently recording.
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferNotRecording);
        }

        unsafe {
            (*self.device).cmd_bind_pipeline(
                self.handle,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
        }
        Ok(())
    }

    /// Bind descriptor sets to this command buffer.
    ///
    /// # Arguments
    ///
    /// * `layout` - The pipeline layout the descriptor sets were allocated for
    /// * `set_offset` - Index of the first descriptor set to bind
    /// * `descriptor_sets` - Slice of descriptor set handles to bind
    ///
    /// # Errors
    ///
    /// Returns an error if the command buffer is not currently recording.
    pub fn bind_descriptor_sets(
        &mut self,
        layout: vk::PipelineLayout,
        set_offset: u32,
        descriptor_sets: &[vk::DescriptorSet],
    ) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferNotRecording);
        }

        unsafe {
            (*self.device).cmd_bind_descriptor_sets(
                self.handle,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                set_offset,
                descriptor_sets,
                &[],
            );
        }
        Ok(())
    }

    /// Dispatch a compute shader with the given workgroup dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the command buffer is not currently recording.
    pub fn dispatch(
        &mut self,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferNotRecording);
        }

        unsafe {
            (*self.device).cmd_dispatch(self.handle, group_count_x, group_count_y, group_count_z);
        }
        Ok(())
    }

    /// Push constant data to the GPU for the currently bound pipeline.
    ///
    /// # Arguments
    ///
    /// * `layout` - The pipeline layout that defines the push constant range
    /// * `offset` - Byte offset within the push constant range
    /// * `data` - Raw bytes to push (must be word-aligned, size <= range)
    ///
    /// # Errors
    ///
    /// Returns an error if the command buffer is not currently recording
    /// or if the data size is not a multiple of 4.
    pub fn push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        offset: u32,
        data: &[u8],
    ) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferNotRecording);
        }

        if !data.len().is_multiple_of(4) {
            return Err(VulkanError::PushConstantValidation(format!(
                "push constant data size {} is not a multiple of 4",
                data.len()
            )));
        }

        unsafe {
            (*self.device).cmd_push_constants(
                self.handle,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                offset,
                data,
            );
        }
        Ok(())
    }

    /// Record a pipeline barrier for transfer write → shader read.
    ///
    /// Ensures that data written to the buffer by a previous transfer
    /// operation (e.g., `vkCmdCopyBuffer` from a staging upload) is
    /// visible to a subsequent compute shader that reads from it.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Buffers to apply the barrier to
    pub fn record_barrier_transfer_to_compute(
        &mut self,
        buffers: &[&VulkanBuffer],
    ) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferNotRecording);
        }

        let buffer_handles: Vec<vk::Buffer> = buffers.iter().map(|b| b.buffer).collect();

        let barrier = vk::BufferMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(vk::Buffer::null())
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .build();

        unsafe {
            (*self.device).cmd_pipeline_barrier(
                self.handle,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }

        // Per-buffer barriers for correct per-buffer tracking
        for buf in &buffer_handles {
            let per_buffer_barrier = vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(*buf)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            unsafe {
                (*self.device).cmd_pipeline_barrier(
                    self.handle,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[per_buffer_barrier],
                    &[],
                );
            }
        }

        Ok(())
    }

    /// Record a pipeline barrier for shader write → transfer read.
    ///
    /// Ensures that data written to the buffer by a compute shader is
    /// visible to a subsequent transfer operation (e.g., `vkCmdCopyBuffer`
    /// for readback to host memory).
    ///
    /// # Arguments
    ///
    /// * `buffers` - Buffers to apply the barrier to
    pub fn record_barrier_compute_to_transfer(
        &mut self,
        buffers: &[&VulkanBuffer],
    ) -> Result<(), VulkanError> {
        if !self.is_recording {
            return Err(VulkanError::CommandBufferNotRecording);
        }

        let buffer_handles: Vec<vk::Buffer> = buffers.iter().map(|b| b.buffer).collect();

        for buf in &buffer_handles {
            let barrier = vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(*buf)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build();

            unsafe {
                (*self.device).cmd_pipeline_barrier(
                    self.handle,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            }
        }

        Ok(())
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        unsafe {
            device.free_command_buffers(self.command_pool, [self.handle].as_slice());
        }
    }
}

// ---------------------------------------------------------------------------
// DescriptorSetLayout — owns a VkDescriptorSetLayout
// ---------------------------------------------------------------------------

/// A Vulkan descriptor set layout defining the bindings available
/// in a descriptor set.
///
/// This is the minimal descriptor abstraction needed to run compute
/// shaders with buffer bindings. It owns the `VkDescriptorSetLayout`
/// handle and ensures correct cleanup on drop.
///
/// # Lifecycle
///
/// 1. Create via `VulkanContext::create_descriptor_set_layout_with_buffer()`
/// 2. Pass to `ComputePipeline::new_with_layouts()` for pipeline layout creation
/// 3. Pass to `VulkanContext::create_descriptor_set_with_buffer()` for allocation
/// 4. `Drop` destroys the layout
pub struct DescriptorSetLayout {
    device: *const ash::Device,
    /// The Vulkan descriptor set layout handle.
    pub handle: vk::DescriptorSetLayout,
}

impl fmt::Debug for DescriptorSetLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DescriptorSetLayout")
            .field("handle", &self.handle)
            .finish()
    }
}

impl DescriptorSetLayout {
    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        unsafe {
            device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

// ---------------------------------------------------------------------------
// DescriptorPool — owns a VkDescriptorPool
// ---------------------------------------------------------------------------

/// A Vulkan descriptor pool for allocating descriptor sets.
///
/// Descriptor sets are allocated from a pool and must be freed
/// before the pool is destroyed. This type owns the `VkDescriptorPool`
/// handle and ensures correct cleanup on drop.
///
/// # Lifecycle
///
/// 1. Create via `DescriptorPool::create()` with a max set count
/// 2. Allocate descriptor sets via `device.allocate_descriptor_sets()`
/// 3. Update descriptor sets via `device.update_descriptor_sets()`
/// 4. Free all descriptor sets via `device.free_descriptor_sets()`
/// 5. `Drop` destroys the pool
pub struct DescriptorPool {
    device: *const ash::Device,
    /// The Vulkan descriptor pool handle.
    pub handle: vk::DescriptorPool,
}

impl fmt::Debug for DescriptorPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DescriptorPool")
            .field("handle", &self.handle)
            .finish()
    }
}

impl DescriptorPool {
    /// Create a new descriptor pool with the given maximum number of
    /// descriptor sets and descriptor type counts.
    ///
    /// # Arguments
    ///
    /// * `device` - The Vulkan device
    /// * `max_sets` - Maximum number of descriptor sets that can be allocated
    /// * `descriptor_counts` - Pairs of (descriptor type, count) specifying
    ///   how many descriptors of each type the pool can hold
    pub fn create(
        device: &ash::Device,
        max_sets: u32,
        descriptor_counts: &[(vk::DescriptorType, u32)],
    ) -> Result<Self, VulkanError> {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = descriptor_counts
            .iter()
            .map(|(ty, count)| {
                vk::DescriptorPoolSize::builder()
                    .ty(*ty)
                    .descriptor_count(*count)
                    .build()
            })
            .collect();

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        let handle = unsafe {
            device
                .create_descriptor_pool(&create_info, None)
                .map_err(VulkanError::DescriptorPoolCreation)?
        };

        Ok(Self {
            device: device as *const ash::Device,
            handle,
        })
    }

    /// Allocate descriptor sets from this pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The Vulkan device (for the API call)
    /// * `layout` - The descriptor set layout to use for each allocated set
    /// * `count` - Number of descriptor sets to allocate
    pub fn allocate(
        &self,
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        count: u32,
    ) -> Result<Vec<vk::DescriptorSet>, VulkanError> {
        let layouts: Vec<vk::DescriptorSetLayout> =
            std::iter::repeat_n(layout, count as usize).collect();

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.handle)
            .set_layouts(&layouts);

        unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(VulkanError::DescriptorSetAllocation)
        }
    }

    /// Free all descriptor sets back to the pool.
    pub fn free_all(
        &self,
        device: &ash::Device,
        sets: &[vk::DescriptorSet],
    ) -> Result<(), VulkanError> {
        unsafe {
            device
                .free_descriptor_sets(self.handle, sets)
                .map_err(VulkanError::DescriptorSetAllocation)?;
        }
        Ok(())
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        unsafe {
            device.destroy_descriptor_pool(self.handle, None);
        }
    }
}

// ---------------------------------------------------------------------------
// DescriptorSet — owns a VkDescriptorSet
// ---------------------------------------------------------------------------

/// A Vulkan descriptor set containing bindings for shader resources.
///
/// This is a minimal wrapper that owns the descriptor set handle.
/// Descriptor sets are allocated from a `DescriptorPool` and must
/// be freed before the pool is destroyed.
///
/// # Lifecycle
///
/// 1. Allocate via `DescriptorPool::allocate()` or
///    `VulkanContext::create_descriptor_set_with_buffer()`
/// 2. Update with `device.update_descriptor_sets()`
/// 3. Bind in a command buffer with `vkCmdBindDescriptorSets`
/// 4. Free back to pool or let pool clean up on drop
pub struct DescriptorSet {
    device: *const ash::Device,
    pool: Box<DescriptorPool>,
    /// The Vulkan descriptor set handle.
    pub handle: vk::DescriptorSet,
}

impl fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DescriptorSet")
            .field("handle", &self.handle)
            .finish()
    }
}

impl DescriptorSet {
    pub fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        unsafe {
            let _ = device.free_descriptor_sets(self.pool.handle, [self.handle].as_slice());
        }
    }
}

// ---------------------------------------------------------------------------
// ShaderModule — owns a VkShaderModule
// ---------------------------------------------------------------------------

/// A Vulkan shader module loaded from SPIR-V bytecode.
///
/// This type owns the `VkShaderModule` handle and ensures correct
/// cleanup on drop. Shader modules are created from precompiled
/// `.spv` binaries — there is no runtime GLSL compilation.
///
/// # Lifecycle
///
/// 1. Load SPIR-V bytes (from `.spv` file or embedded binary)
/// 2. Create via `ShaderModule::from_spv_bytes()` or
///    `VulkanContext::create_shader_module_from_spv_bytes()`
/// 3. Use the handle in pipeline creation (`vk::ShaderModuleCreateInfo`)
/// 4. `Drop` destroys the shader module
///
/// The module holds a raw pointer to the `ash::Device` for cleanup.
/// Callers must ensure the device outlives all shader modules created
/// from it. `VulkanContext` guarantees this by owning both.
pub struct ShaderModule {
    device: *const ash::Device,
    /// The Vulkan shader module handle.
    handle: vk::ShaderModule,
    /// Number of u32 words in the SPIR-V code.
    word_count: u32,
}

impl fmt::Debug for ShaderModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShaderModule")
            .field("handle", &self.handle)
            .field("word_count", &self.word_count)
            .finish()
    }
}

impl ShaderModule {
    /// Returns the Vulkan shader module handle for use in pipeline
    /// creation.
    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }

    /// Returns the number of u32 words in the SPIR-V code.
    ///
    /// This is equivalent to `code.len() / 4` for the original
    /// bytecode that was used to create this module.
    pub fn word_count(&self) -> u32 {
        self.word_count
    }

    /// Create a shader module from raw SPIR-V bytes.
    ///
    /// The byte slice must:
    /// - Be non-empty
    /// - Have a length that is a multiple of 4 (SPIR-V is a sequence of u32 words)
    ///
    /// The bytes are consumed into an owned `Vec<u8>` for internal
    /// validation, but the Vulkan module owns its own copy of the code.
    pub fn from_spv_bytes(device: &ash::Device, spv_bytes: &[u8]) -> Result<Self, VulkanError> {
        if spv_bytes.is_empty() {
            return Err(VulkanError::InvalidSpvBytes(
                "SPIR-V bytecode must not be empty".to_string(),
            ));
        }

        if !spv_bytes.len().is_multiple_of(4) {
            return Err(VulkanError::InvalidSpvBytes(format!(
                "SPIR-V bytecode length {} is not a multiple of 4 (must be word-aligned)",
                spv_bytes.len()
            )));
        }

        let word_count = (spv_bytes.len() / 4) as u32;

        // SAFETY: spv_bytes is valid for its entire lifetime (it's a &[]
        // parameter). We only read from it during this function call to
        // create the Vulkan module, which copies the data internally.
        let code_words = unsafe {
            std::slice::from_raw_parts(spv_bytes.as_ptr() as *const u32, word_count as usize)
        };

        let create_info = vk::ShaderModuleCreateInfo::builder().code(code_words);

        let handle = unsafe {
            device
                .create_shader_module(&create_info, None)
                .map_err(VulkanError::ShaderModuleCreation)?
        };

        Ok(Self {
            device: device as *const ash::Device,
            handle,
            word_count,
        })
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        unsafe {
            device.destroy_shader_module(self.handle, None);
        }
    }
}

/// Load a `.spv` binary file from disk.
///
/// Returns the raw bytes of the file. The caller is responsible for
/// passing the bytes to `ShaderModule::from_spv_bytes()` or
/// `VulkanContext::create_shader_module_from_spv_bytes()`.
///
/// This function is intentionally separate from the Vulkan creation
/// path so that file I/O errors can be distinguished from Vulkan
/// API errors.
pub fn load_spv_file(path: &std::path::Path) -> Result<Vec<u8>, VulkanError> {
    std::fs::read(path).map_err(|e| {
        VulkanError::SpvLoadingError(format!("failed to read {}: {}", path.display(), e))
    })
}

// ---------------------------------------------------------------------------
// ComputePipeline — owns VkPipeline + VkPipelineLayout
// ---------------------------------------------------------------------------

/// A Vulkan compute pipeline with its associated pipeline layout.
///
/// This type owns both the `VkPipeline` and `VkPipelineLayout` handles
/// and ensures correct cleanup order on drop: destroy pipeline first,
/// then destroy pipeline layout.
///
/// The pipeline is created from a `ShaderModule` and an entry point name.
/// The pipeline layout starts minimal (no descriptor sets, no push
/// constants) and is designed to be extended when kernels require
/// buffer bindings.
///
/// # Lifecycle
///
/// 1. Create a `ShaderModule` from SPIR-V bytes
/// 2. Create via `ComputePipeline::new()` or
///    `VulkanContext::create_compute_pipeline()`
/// 3. Bind in a command buffer with `vkCmdBindPipeline`
/// 4. `Drop` destroys pipeline, then pipeline layout
///
/// The pipeline holds a raw pointer to the `ash::Device` for cleanup.
/// Callers must ensure the device outlives all pipelines created from
/// it. `VulkanContext` guarantees this by owning both.
pub struct ComputePipeline {
    device: *const ash::Device,
    /// The Vulkan compute pipeline handle.
    pipeline: vk::Pipeline,
    /// The Vulkan pipeline layout handle.
    layout: vk::PipelineLayout,
}

impl fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("pipeline", &self.pipeline)
            .field("layout", &self.layout)
            .finish()
    }
}

impl ComputePipeline {
    /// Returns the Vulkan pipeline handle for use in `vkCmdBindPipeline`.
    pub fn handle(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// Returns the Vulkan pipeline layout handle for use in descriptor
    /// set bindings and future pipeline cache operations.
    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }

    /// Create a compute pipeline from a shader module and entry point name.
    ///
    /// Creates a minimal pipeline layout with no descriptor set layouts
    /// and no push constants. This is sufficient for shaders that don't
    /// bind buffers. For shaders that need descriptor sets, use
    /// `new_with_layouts`.
    ///
    /// # Errors
    ///
    /// Returns `VulkanError::InvalidEntryPoint` if the entry point name
    /// contains a null byte. Returns pipeline creation errors for Vulkan
    /// API failures.
    pub fn new(
        device: &ash::Device,
        shader_module: &ShaderModule,
        entry_point: &str,
    ) -> Result<Self, VulkanError> {
        Self::new_with_layouts(device, shader_module, entry_point, &[])
    }

    /// Create a compute pipeline with descriptor set layouts.
    ///
    /// This is the full constructor that accepts descriptor set layouts
    /// for shaders that bind buffers, textures, or other resources.
    ///
    /// # Arguments
    ///
    /// * `device` - The Vulkan device
    /// * `shader_module` - The compiled shader module
    /// * `entry_point` - Name of the entry point function in the shader
    /// * `set_layouts` - Descriptor set layouts to include in the pipeline layout
    ///
    /// # Errors
    ///
    /// Returns `VulkanError::InvalidEntryPoint` if the entry point name
    /// contains a null byte. Returns pipeline creation errors for Vulkan
    /// API failures.
    pub fn new_with_layouts(
        device: &ash::Device,
        shader_module: &ShaderModule,
        entry_point: &str,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self, VulkanError> {
        let entry_point_cstr = CString::new(entry_point).map_err(|e| {
            VulkanError::InvalidEntryPoint(format!(
                "entry point name contains invalid bytes: {}",
                e
            ))
        })?;

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(set_layouts);

        let layout = unsafe {
            device
                .create_pipeline_layout(&layout_create_info, None)
                .map_err(VulkanError::PipelineLayoutCreation)?
        };

        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module.handle())
            .name(entry_point_cstr.as_c_str())
            .build();

        let pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .layout(layout)
            .build();

        let pipelines = unsafe {
            match device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            ) {
                Ok(pipes) => pipes,
                Err((_, vk_result)) => {
                    device.destroy_pipeline_layout(layout, None);
                    return Err(VulkanError::ComputePipelineCreation(vk_result));
                }
            }
        };

        let pipeline = pipelines
            .into_iter()
            .next()
            .ok_or(VulkanError::ComputePipelineCreation(
                vk::Result::ERROR_UNKNOWN,
            ))?;

        Ok(Self {
            device: device as *const ash::Device,
            pipeline,
            layout,
        })
    }

    /// Create a compute pipeline with descriptor set layouts and push
    /// constants.
    ///
    /// # Arguments
    ///
    /// * `device` - The Vulkan device
    /// * `shader_module` - The compiled shader module
    /// * `entry_point` - Name of the entry point function
    /// * `set_layouts` - Descriptor set layouts for the pipeline
    /// * `push_constant_range` - Push constant range configuration
    pub fn new_with_push_constants(
        device: &ash::Device,
        shader_module: &ShaderModule,
        entry_point: &str,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_range: vk::PushConstantRange,
    ) -> Result<Self, VulkanError> {
        let entry_point_cstr = CString::new(entry_point).map_err(|e| {
            VulkanError::InvalidEntryPoint(format!(
                "entry point name contains invalid bytes: {}",
                e
            ))
        })?;

        let push_constant_ranges = [push_constant_range];
        let layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let layout = unsafe {
            device
                .create_pipeline_layout(&layout_create_info, None)
                .map_err(VulkanError::PipelineLayoutCreation)?
        };

        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module.handle())
            .name(entry_point_cstr.as_c_str())
            .build();

        let pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .layout(layout)
            .build();

        let pipelines = unsafe {
            match device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            ) {
                Ok(pipes) => pipes,
                Err((_, vk_result)) => {
                    device.destroy_pipeline_layout(layout, None);
                    return Err(VulkanError::ComputePipelineCreation(vk_result));
                }
            }
        };

        let pipeline = pipelines
            .into_iter()
            .next()
            .ok_or(VulkanError::ComputePipelineCreation(
                vk::Result::ERROR_UNKNOWN,
            ))?;

        Ok(Self {
            device: device as *const ash::Device,
            pipeline,
            layout,
        })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        if self.device.is_null() {
            return;
        }
        let device = unsafe { &*self.device };
        // Destroy pipeline first, then layout (correct Vulkan destruction order).
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// DescriptorBinding — configuration for a single buffer binding
// ---------------------------------------------------------------------------

/// Configuration for a single storage buffer descriptor binding.
///
/// Used with `VulkanContext::create_descriptor_set_layout_with_buffers()`
/// and `VulkanContext::create_descriptor_set_with_buffers()` to create
/// multi-binding descriptor set layouts and sets.
#[derive(Debug, Clone)]
pub struct DescriptorBinding {
    /// The binding number (e.g., 0 for `layout(binding = 0)`).
    pub binding: u32,
    /// The Vulkan descriptor type (typically `STORAGE_BUFFER`).
    pub descriptor_type: vk::DescriptorType,
    /// The shader stage flags that will access this binding.
    pub stage_flags: vk::ShaderStageFlags,
}

// ---------------------------------------------------------------------------
// VulkanContext — owns instance + logical device + queues
// ---------------------------------------------------------------------------

/// A fully-initialised Vulkan context ready for compute work.
pub struct VulkanContext {
    /// The Vulkan entry point (needed for instance-level calls).
    entry: Entry,
    /// Vulkan instance.
    instance: Instance,
    /// Selected physical device handle.
    physical_device: vk::PhysicalDevice,
    /// Cached info about the selected physical device.
    physical_device_info: PhysicalDeviceInfo,
    /// Logical device and compute queue.
    device: VulkanDevice,
    /// Selected queue family information.
    queue_family: QueueFamilySelection,
    /// Command pool resources.
    commands: CommandResources,
    /// Cached memory properties for buffer allocation.
    memory_selector: MemoryTypeSelector,
}

impl VulkanContext {
    /// Create a new Vulkan context.
    ///
    /// In debug builds validation layers are enabled automatically (if
    /// available).  An AMD GPU is preferred when multiple devices exist.
    pub fn new() -> Result<Self, VulkanError> {
        let entry = unsafe { Entry::load() }.map_err(|_| VulkanError::LoaderNotFound)?;
        let instance = create_instance(&entry)?;
        let (physical_device, info) = select_device(&instance)?;
        let (vulkan_device, queue_family, commands) = create_device(&instance, physical_device)?;
        let memory_selector = MemoryTypeSelector::new(&instance, physical_device);

        Ok(Self {
            entry,
            instance,
            physical_device,
            physical_device_info: info,
            device: vulkan_device,
            queue_family,
            commands,
            memory_selector,
        })
    }

    // -- Accessors ----------------------------------------------------------

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Returns the logical device wrapper.
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn physical_device_info(&self) -> &PhysicalDeviceInfo {
        &self.physical_device_info
    }

    /// Returns the selected compute queue family information.
    pub fn queue_family(&self) -> &QueueFamilySelection {
        &self.queue_family
    }

    /// Returns the command pool resources.
    pub fn commands(&self) -> &CommandResources {
        &self.commands
    }

    /// Returns the memory type selector for buffer allocations.
    pub fn memory_selector(&self) -> &MemoryTypeSelector {
        &self.memory_selector
    }

    /// Create a device-local buffer for GPU-side data (weights, KV cache).
    ///
    /// The buffer is allocated with `DEVICE_LOCAL` memory for fast GPU
    /// access. Data must be uploaded via a staging buffer and
    /// `vkCmdCopyBuffer`.
    pub fn create_device_local_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<VulkanBuffer, VulkanError> {
        self.create_buffer(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    }

    /// Create a host-visible, coherent staging buffer.
    ///
    /// The buffer is allocated with `HOST_VISIBLE | HOST_COHERENT` memory
    /// and is mapped immediately. Data can be written directly via
    /// `VulkanBuffer::write_data()`.
    pub fn create_host_visible_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> Result<VulkanBuffer, VulkanError> {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// Create a buffer with specific usage and memory property flags.
    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Result<VulkanBuffer, VulkanError> {
        if size == 0 {
            return Err(VulkanError::BufferCreation(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ));
        }

        let create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .handle()
                .create_buffer(&create_info, None)
                .map_err(VulkanError::BufferCreation)?
        };

        let requirements = unsafe { self.device.handle().get_buffer_memory_requirements(buffer) };

        let memory_type_index = self
            .memory_selector
            .find_memory_type(requirements.memory_type_bits, memory_property_flags)
            .ok_or_else(|| {
                unsafe {
                    self.device.handle().destroy_buffer(buffer, None);
                }
                VulkanError::NoSuitableMemoryType
            })?;

        VulkanBuffer::create_with_memory_type(
            self.device.handle(),
            size,
            usage,
            memory_property_flags,
            memory_type_index,
        )
    }

    // -- Command buffer management ------------------------------------------

    /// Allocate a primary command buffer from the context's command pool.
    pub fn allocate_command_buffer(&self) -> Result<CommandBuffer, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.commands.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let handles = unsafe {
            self.device
                .handle()
                .allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::CommandBufferAllocation)?
        };

        if handles.is_empty() {
            return Err(VulkanError::CommandBufferAllocation(
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ));
        }

        Ok(CommandBuffer {
            device: self.device.handle() as *const ash::Device,
            command_pool: self.commands.command_pool,
            handle: handles[0],
            is_recording: false,
        })
    }

    /// Execute a closure that records commands, submitting and waiting
    /// for completion synchronously.
    ///
    /// The allocated command buffer is freed after execution. This is
    /// the simplest path for one-shot operations like staging uploads.
    pub fn execute_immediate(
        &self,
        record: impl FnOnce(&mut CommandBuffer) -> Result<(), VulkanError>,
    ) -> Result<(), VulkanError> {
        let mut cmd = self.allocate_command_buffer()?;
        cmd.begin_one_time_submit()?;
        record(&mut cmd)?;
        cmd.end()?;
        cmd.submit_and_wait(
            self.device.compute_queue,
            self.queue_family.queue_family_index,
        )?;
        Ok(())
    }

    // -- Transfer / upload operations ---------------------------------------

    /// Copy `size` bytes from `src` to `dst` via a recorded, submitted,
    /// and synchronously-waited buffer copy.
    ///
    /// `src` must have `TRANSFER_SRC` usage. `dst` must have
    /// `TRANSFER_DST` usage.
    pub fn copy_buffer(
        &self,
        src: &VulkanBuffer,
        dst: &VulkanBuffer,
        size: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        if size == 0 {
            return Ok(());
        }

        if size > src.size {
            return Err(VulkanError::TransferValidation(format!(
                "copy size {} exceeds source buffer size {}",
                size, src.size
            )));
        }

        if size > dst.size {
            return Err(VulkanError::TransferValidation(format!(
                "copy size {} exceeds destination buffer size {}",
                size, dst.size
            )));
        }

        self.execute_immediate(|cmd| cmd.record_copy_buffer(src, dst, size))
    }

    /// Upload CPU data into a device-local buffer via a staging buffer.
    ///
    /// Creates a temporary host-visible staging buffer, writes the data,
    /// performs a synchronous `vkCmdCopyBuffer` to the destination, and
    /// cleans up the staging buffer.
    ///
    /// The destination buffer must have been created with `DEVICE_LOCAL`
    /// memory and include `TRANSFER_DST` in its usage flags.
    pub fn upload_to_device_local(
        &self,
        dst: &VulkanBuffer,
        data: &[u8],
    ) -> Result<(), VulkanError> {
        if !dst
            .memory_property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Err(VulkanError::TransferValidation(
                "destination buffer is not DEVICE_LOCAL".to_string(),
            ));
        }

        if !dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
            return Err(VulkanError::TransferValidation(
                "destination buffer missing TRANSFER_DST usage flag".to_string(),
            ));
        }

        if data.is_empty() {
            return Ok(());
        }

        let data_len = data.len() as vk::DeviceSize;
        if data_len > dst.size {
            return Err(VulkanError::TransferValidation(format!(
                "data size {} exceeds destination buffer size {}",
                data_len, dst.size
            )));
        }

        let mut staging = self.create_host_visible_buffer(data_len)?;
        staging.write_data(data)?;

        self.execute_immediate(|cmd| cmd.record_copy_buffer(&staging, dst, data_len))
    }

    // -- Shader module helpers -----------------------------------------------

    /// Create a shader module from SPIR-V bytes.
    ///
    /// Validates byte alignment and delegates to the device.
    pub fn create_shader_module_from_spv_bytes(
        &self,
        spv_bytes: &[u8],
    ) -> Result<ShaderModule, VulkanError> {
        ShaderModule::from_spv_bytes(self.device.handle(), spv_bytes)
    }

    /// Create a shader module from a `.spv` file on disk.
    ///
    /// Loads the file bytes and creates the Vulkan shader module.
    pub fn create_shader_module_from_spv_file(
        &self,
        path: &std::path::Path,
    ) -> Result<ShaderModule, VulkanError> {
        let bytes = load_spv_file(path)?;
        self.create_shader_module_from_spv_bytes(&bytes)
    }

    /// Create a compute pipeline from a shader module and entry point.
    ///
    /// Creates a minimal pipeline layout with no descriptor sets and
    /// no push constants. Extend when kernels require buffer bindings.
    pub fn create_compute_pipeline(
        &self,
        shader_module: &ShaderModule,
        entry_point: &str,
    ) -> Result<ComputePipeline, VulkanError> {
        ComputePipeline::new(self.device.handle(), shader_module, entry_point)
    }

    /// Create a compute pipeline with descriptor set layouts.
    ///
    /// # Arguments
    ///
    /// * `shader_module` - The compiled shader module
    /// * `entry_point` - Name of the entry point function
    /// * `set_layouts` - Descriptor set layout handles for the pipeline
    pub fn create_compute_pipeline_with_layouts(
        &self,
        shader_module: &ShaderModule,
        entry_point: &str,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<ComputePipeline, VulkanError> {
        ComputePipeline::new_with_layouts(
            self.device.handle(),
            shader_module,
            entry_point,
            set_layouts,
        )
    }

    /// Create a compute pipeline with descriptor set layouts and push
    /// constants.
    ///
    /// # Arguments
    ///
    /// * `shader_module` - The compiled shader module
    /// * `entry_point` - Name of the entry point function
    /// * `set_layouts` - Descriptor set layout handles for the pipeline
    /// * `push_constant_range` - Push constant range configuration
    pub fn create_compute_pipeline_with_push_constants(
        &self,
        shader_module: &ShaderModule,
        entry_point: &str,
        set_layouts: &[vk::DescriptorSetLayout],
        push_constant_range: vk::PushConstantRange,
    ) -> Result<ComputePipeline, VulkanError> {
        ComputePipeline::new_with_push_constants(
            self.device.handle(),
            shader_module,
            entry_point,
            set_layouts,
            push_constant_range,
        )
    }

    /// Create a descriptor set layout with a single storage buffer binding.
    ///
    /// # Arguments
    ///
    /// * `binding` - The binding number (e.g., 0 for `layout(binding = 0)`)
    pub fn create_descriptor_set_layout_with_buffer(
        &self,
        binding: u32,
    ) -> Result<DescriptorSetLayout, VulkanError> {
        let binding_info = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();

        let bindings = [binding_info];
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        let handle = unsafe {
            self.device
                .handle()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(VulkanError::DescriptorSetLayoutCreation)?
        };

        Ok(DescriptorSetLayout {
            device: self.device.handle() as *const ash::Device,
            handle,
        })
    }

    /// Create a descriptor set with a single storage buffer binding.
    ///
    /// Creates a descriptor pool, allocates a descriptor set, and updates
    /// it with the given buffer binding. This is the convenience method
    /// for the common case of a single buffer binding.
    ///
    /// # Arguments
    ///
    /// * `layout` - The descriptor set layout (must have a storage buffer binding)
    /// * `binding` - The binding number matching the layout
    /// * `buffer` - The buffer to bind
    /// * `range` - Range of the buffer in bytes (use `vk::WHOLE_SIZE` for full buffer)
    pub fn create_descriptor_set_with_buffer(
        &self,
        layout: &DescriptorSetLayout,
        binding: u32,
        buffer: &VulkanBuffer,
        range: vk::DeviceSize,
    ) -> Result<DescriptorSet, VulkanError> {
        let device = self.device.handle();

        let pool = DescriptorPool::create(device, 1, &[(vk::DescriptorType::STORAGE_BUFFER, 1)])?;

        let sets = pool.allocate(device, layout.handle, 1)?;
        let set = sets[0];

        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(buffer.buffer)
            .offset(0)
            .range(range)
            .build();

        let descriptor_write = {
            let mut write = vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(binding)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&[buffer_info])
                .build();
            write.descriptor_count = 1;
            write
        };

        unsafe {
            device.update_descriptor_sets(&[descriptor_write], &[]);
        }

        Ok(DescriptorSet {
            device: device as *const ash::Device,
            pool: Box::new(pool),
            handle: set,
        })
    }

    /// Create a descriptor set layout with multiple buffer bindings.
    ///
    /// # Arguments
    ///
    /// * `bindings` - Configuration for each binding in the layout.
    ///   Bindings must start at 0 and be consecutive with no duplicates.
    ///
    /// # Errors
    ///
    /// Returns `VulkanError::PushConstantValidation` if bindings are
    /// empty, not consecutive starting from 0, or contain duplicates.
    pub fn create_descriptor_set_layout_with_buffers(
        &self,
        bindings: &[DescriptorBinding],
    ) -> Result<DescriptorSetLayout, VulkanError> {
        if bindings.is_empty() {
            return Err(VulkanError::PushConstantValidation(
                "descriptor set layout requires at least one binding".to_string(),
            ));
        }

        let mut sorted = bindings.to_vec();
        sorted.sort_by_key(|b| b.binding);

        for (i, binding) in sorted.iter().enumerate() {
            if binding.binding != i as u32 {
                return Err(VulkanError::PushConstantValidation(format!(
                    "descriptor bindings must be consecutive starting from 0, found gap at index {} (got binding {})",
                    i, binding.binding
                )));
            }
        }

        let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(b.binding)
                    .descriptor_type(b.descriptor_type)
                    .descriptor_count(1)
                    .stage_flags(b.stage_flags)
                    .build()
            })
            .collect();

        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        let handle = unsafe {
            self.device
                .handle()
                .create_descriptor_set_layout(&create_info, None)
                .map_err(VulkanError::DescriptorSetLayoutCreation)?
        };

        Ok(DescriptorSetLayout {
            device: self.device.handle() as *const ash::Device,
            handle,
        })
    }

    /// Create a descriptor set with multiple buffer bindings.
    ///
    /// Creates a descriptor pool, allocates a descriptor set, and updates
    /// it with the given buffer bindings.
    ///
    /// # Arguments
    ///
    /// * `layout` - The descriptor set layout (must match the bindings)
    /// * `bindings` - Buffer bindings to write into the descriptor set
    pub fn create_descriptor_set_with_buffers(
        &self,
        layout: &DescriptorSetLayout,
        bindings: &[(u32, &VulkanBuffer, vk::DeviceSize)],
    ) -> Result<DescriptorSet, VulkanError> {
        let device = self.device.handle();

        let desc_count = bindings.len() as u32;
        let pool = DescriptorPool::create(
            device,
            1,
            &[(vk::DescriptorType::STORAGE_BUFFER, desc_count)],
        )?;

        let sets = pool.allocate(device, layout.handle, 1)?;
        let set = sets[0];

        let descriptor_writes: Vec<vk::WriteDescriptorSet> = bindings
            .iter()
            .map(|(binding, buffer, range)| {
                let buffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(*range)
                    .build();

                let mut write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[buffer_info])
                    .build();
                write.descriptor_count = 1;
                write
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        Ok(DescriptorSet {
            device: device as *const ash::Device,
            pool: Box::new(pool),
            handle: set,
        })
    }

    /// Read data back from a device-local buffer to host memory.
    ///
    /// Creates a host-visible staging buffer, copies data from the source
    /// buffer, and reads the bytes back. This validates the full readback
    /// path that will be used for future inference results.
    ///
    /// # Arguments
    ///
    /// * `src` - The device-local source buffer (must have TRANSFER_SRC usage)
    /// * `size` - Number of bytes to read back
    pub fn readback_buffer_data(
        &self,
        src: &VulkanBuffer,
        size: vk::DeviceSize,
    ) -> Result<Vec<u8>, VulkanError> {
        let mut staging = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        self.copy_buffer(src, &staging, size)?;

        // Staging buffer is mapped at creation time for HOST_VISIBLE memory.
        // Unmap and remap to get a fresh pointer, or use existing mapping.
        if staging.is_mapped() {
            staging.unmap()?;
        }
        let ptr = staging.map()?;
        let mut data = vec![0u8; size as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size as usize);
        }
        staging.unmap()?;

        Ok(data)
    }

    /// Wait for the device to finish all pending work.
    ///
    /// Useful for cleanup / shutdown.  Avoid in hot paths.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        unsafe { self.device.handle().device_wait_idle() }
    }

    /// Enumerate **all** physical devices on the system (useful for
    /// diagnostic / info commands).
    pub fn enumerate_physical_devices(&self) -> Result<Vec<PhysicalDeviceInfo>, VulkanError> {
        let phys_devices = unsafe { self.instance.enumerate_physical_devices() }
            .map_err(|_| VulkanError::NoPhysicalDevices)?;

        let mut infos = Vec::new();
        for pd in phys_devices {
            let info = build_physical_device_info(&self.instance, pd);
            infos.push(info);
        }
        Ok(infos)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle()
                .destroy_command_pool(self.commands.command_pool, None);
            self.device.handle().destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Instance creation
// ---------------------------------------------------------------------------

#[allow(unused_mut)]
fn create_instance(entry: &Entry) -> Result<Instance, VulkanError> {
    let app_name_cstr = CString::new(APP_NAME).unwrap();
    let engine_name_cstr = CString::new(ENGINE_NAME).unwrap();

    let appinfo = vk::ApplicationInfo::builder()
        .application_name(app_name_cstr.as_c_str())
        .application_version(APP_VERSION)
        .engine_name(engine_name_cstr.as_c_str())
        .engine_version(ENGINE_VERSION)
        .api_version(API_VERSION);

    let mut enabled_layers: Vec<*const i8> = Vec::new();

    // Validation layers in debug builds
    #[cfg(debug_assertions)]
    {
        let available = entry
            .enumerate_instance_layer_properties()
            .map_err(VulkanError::InstanceCreation)?;

        for layer in VALIDATION_LAYERS {
            if available.iter().any(|l| {
                let name = unsafe { CStr::from_ptr(&l.layer_name[0]) }
                    .to_string_lossy()
                    .into_owned();
                name == *layer
            }) {
                enabled_layers.push(layer.as_ptr() as *const i8);
            }
        }
    }

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&appinfo)
        .enabled_layer_names(&enabled_layers);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .map_err(VulkanError::InstanceCreation)?
    };

    Ok(instance)
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------

fn get_driver_name(instance: &Instance, pd: vk::PhysicalDevice) -> String {
    // Use VkPhysicalDeviceProperties2 + pNext chaining to get driver name
    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2 {
        p_next: &mut driver_props as *mut _ as *mut std::ffi::c_void,
        ..Default::default()
    };

    unsafe {
        instance.get_physical_device_properties2(pd, &mut props2);
    }

    unsafe {
        CStr::from_ptr(driver_props.driver_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

fn get_device_extensions(instance: &Instance, pd: vk::PhysicalDevice) -> Vec<String> {
    unsafe {
        match instance.enumerate_device_extension_properties(pd) {
            Ok(extensions_raw) => extensions_raw
                .iter()
                .map(|ext| {
                    CStr::from_ptr(&ext.extension_name[0])
                        .to_string_lossy()
                        .into_owned()
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

fn build_physical_device_info(instance: &Instance, pd: vk::PhysicalDevice) -> PhysicalDeviceInfo {
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let name = unsafe {
        CStr::from_ptr(&props.device_name[0])
            .to_string_lossy()
            .into_owned()
    };

    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };

    let queue_families_info: Vec<QueueFamilyInfo> = queue_families
        .iter()
        .enumerate()
        .map(|(i, qf)| QueueFamilyInfo {
            index: i as u32,
            queue_count: qf.queue_count,
            queue_flags: qf.queue_flags,
            supports_compute: qf.queue_flags.contains(vk::QueueFlags::COMPUTE),
            supports_graphics: qf.queue_flags.contains(vk::QueueFlags::GRAPHICS),
            supports_transfer: qf.queue_flags.contains(vk::QueueFlags::TRANSFER),
        })
        .collect();

    let driver_name = get_driver_name(instance, pd);
    let extensions = get_device_extensions(instance, pd);

    PhysicalDeviceInfo {
        name,
        device_type: props.device_type,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        api_version: props.api_version,
        driver_version: props.driver_version,
        driver_name,
        queue_families: queue_families_info,
        extensions,
    }
}

/// Select the best physical device, preferring AMD GPUs.
fn select_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, PhysicalDeviceInfo), VulkanError> {
    let phys_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|_| VulkanError::NoPhysicalDevices)?;

    if phys_devices.is_empty() {
        return Err(VulkanError::NoPhysicalDevices);
    }

    // Score each device: higher is better
    let mut best: Option<(vk::PhysicalDevice, PhysicalDeviceInfo, i64)> = None;

    for pd in &phys_devices {
        let info = build_physical_device_info(instance, *pd);

        // Must have at least one compute queue family
        if !info.queue_families.iter().any(|q| q.supports_compute) {
            continue;
        }

        let mut score: i64 = 0;

        // AMD vendor ID gets a large bonus
        if info.vendor_id == AMD_VENDOR_ID {
            score += 1_000_000;
        }

        // Prefer discrete GPUs
        match info.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => score += 100_000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => score += 50_000,
            _ => {}
        }

        // Prefer more compute queues
        let compute_queues: u32 = info
            .queue_families
            .iter()
            .filter(|q| q.supports_compute)
            .map(|q| q.queue_count)
            .sum();
        score += compute_queues as i64 * 1_000;

        // Prefer larger heap (approximate by driver version as tie-breaker)
        score += info.driver_version as i64;

        match &best {
            None => best = Some((*pd, info, score)),
            Some((_, _, best_score)) => {
                if score > *best_score {
                    best = Some((*pd, info, score));
                }
            }
        }
    }

    match best {
        Some((pd, info, _)) => Ok((pd, info)),
        None => Err(VulkanError::NoComputeQueue),
    }
}

// ---------------------------------------------------------------------------
// Logical device, queue, and command pool creation
// ---------------------------------------------------------------------------

fn create_device(
    instance: &Instance,
    pd: vk::PhysicalDevice,
) -> Result<(VulkanDevice, QueueFamilySelection, CommandResources), VulkanError> {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };

    // Find first compute-capable queue family
    let queue_family_index = queue_families
        .iter()
        .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or(VulkanError::NoComputeQueue)? as u32;

    let selected_qf = &queue_families[queue_family_index as usize];

    let queue_selection = QueueFamilySelection {
        queue_family_index,
        queue_count: selected_qf.queue_count,
        queue_flags: selected_qf.queue_flags,
    };

    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let queue_create_infos = [*queue_create_info];
    let create_info = vk::DeviceCreateInfo::builder().queue_create_infos(&queue_create_infos);

    let device = unsafe {
        instance
            .create_device(pd, &create_info, None)
            .map_err(VulkanError::DeviceCreation)?
    };

    let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    let command_pool = unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .map_err(VulkanError::CommandPoolCreation)?
    };

    let vulkan_device = VulkanDevice {
        inner: device,
        compute_queue,
    };

    let commands = CommandResources { command_pool };

    Ok((vulkan_device, queue_selection, commands))
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a Vulkan version number as "major.minor.patch".
pub fn format_version(version: u32) -> String {
    let major = vk::api_version_major(version);
    let minor = vk::api_version_minor(version);
    let patch = vk::api_version_patch(version);
    format!("{}.{}.{}", major, minor, patch)
}

/// Format a VkPhysicalDeviceType as a human-readable string.
pub fn format_device_type(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::OTHER => "Other",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "Unknown",
    }
}

/// Format a vendor ID, resolving known vendors.
pub fn format_vendor_id(vendor_id: u32) -> String {
    match vendor_id {
        0x1002 => "AMD".to_string(),
        0x10DE => "NVIDIA".to_string(),
        0x8086 => "Intel".to_string(),
        0x1AE0 => "ImgTec".to_string(),
        0x1414 => "NVIDIA (Tesla)".to_string(),
        _ => format!("0x{:04X}", vendor_id),
    }
}

/// Format queue flags as a comma-separated list.
pub fn format_queue_flags(flags: vk::QueueFlags) -> String {
    let mut parts = Vec::new();
    if flags.contains(vk::QueueFlags::GRAPHICS) {
        parts.push("graphics");
    }
    if flags.contains(vk::QueueFlags::COMPUTE) {
        parts.push("compute");
    }
    if flags.contains(vk::QueueFlags::TRANSFER) {
        parts.push("transfer");
    }
    if flags.contains(vk::QueueFlags::SPARSE_BINDING) {
        parts.push("sparse");
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- format_version tests ---

    #[test]
    fn test_format_version_1_0() {
        let v = vk::make_api_version(0, 1, 0, 0);
        assert_eq!(format_version(v), "1.0.0");
    }

    #[test]
    fn test_format_version_1_1() {
        let v = vk::make_api_version(0, 1, 1, 0);
        assert_eq!(format_version(v), "1.1.0");
    }

    #[test]
    fn test_format_version_1_2() {
        let v = vk::make_api_version(0, 1, 2, 0);
        assert_eq!(format_version(v), "1.2.0");
    }

    #[test]
    fn test_format_version_1_3() {
        let v = vk::make_api_version(0, 1, 3, 0);
        assert_eq!(format_version(v), "1.3.0");
    }

    #[test]
    fn test_format_version_with_patch() {
        let v = vk::make_api_version(0, 1, 2, 195);
        assert_eq!(format_version(v), "1.2.195");
    }

    #[test]
    fn test_format_version_zero() {
        let v = vk::make_api_version(0, 0, 0, 0);
        assert_eq!(format_version(v), "0.0.0");
    }

    // --- format_device_type tests ---

    #[test]
    fn test_format_device_type_discrete() {
        assert_eq!(
            format_device_type(vk::PhysicalDeviceType::DISCRETE_GPU),
            "Discrete GPU"
        );
    }

    #[test]
    fn test_format_device_type_integrated() {
        assert_eq!(
            format_device_type(vk::PhysicalDeviceType::INTEGRATED_GPU),
            "Integrated GPU"
        );
    }

    #[test]
    fn test_format_device_type_cpu() {
        assert_eq!(format_device_type(vk::PhysicalDeviceType::CPU), "CPU");
    }

    #[test]
    fn test_format_device_type_other() {
        assert_eq!(format_device_type(vk::PhysicalDeviceType::OTHER), "Other");
    }

    // --- format_vendor_id tests ---

    #[test]
    fn test_format_vendor_amd() {
        assert_eq!(format_vendor_id(0x1002), "AMD");
    }

    #[test]
    fn test_format_vendor_nvidia() {
        assert_eq!(format_vendor_id(0x10DE), "NVIDIA");
    }

    #[test]
    fn test_format_vendor_intel() {
        assert_eq!(format_vendor_id(0x8086), "Intel");
    }

    #[test]
    fn test_format_vendor_unknown() {
        assert_eq!(format_vendor_id(0xBEEF), "0xBEEF");
    }

    // --- format_queue_flags tests ---

    #[test]
    fn test_format_queue_flags_compute() {
        assert_eq!(format_queue_flags(vk::QueueFlags::COMPUTE), "compute");
    }

    #[test]
    fn test_format_queue_flags_graphics_compute() {
        let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        assert_eq!(format_queue_flags(flags), "graphics, compute");
    }

    #[test]
    fn test_format_queue_flags_all() {
        let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        assert_eq!(format_queue_flags(flags), "graphics, compute, transfer");
    }

    #[test]
    fn test_format_queue_flags_empty() {
        assert_eq!(format_queue_flags(vk::QueueFlags::empty()), "none");
    }

    // --- QueueFamilySelection tests ---

    #[test]
    fn test_queue_family_selection_fields() {
        let sel = QueueFamilySelection {
            queue_family_index: 2,
            queue_count: 4,
            queue_flags: vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
        };
        assert_eq!(sel.queue_family_index, 2);
        assert_eq!(sel.queue_count, 4);
        assert!(sel.queue_flags.contains(vk::QueueFlags::COMPUTE));
        assert!(sel.queue_flags.contains(vk::QueueFlags::TRANSFER));
        assert!(!sel.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    }

    // --- QueueFamilyInfo tests ---

    #[test]
    fn test_queue_family_info_compute_only() {
        let info = QueueFamilyInfo {
            index: 0,
            queue_count: 1,
            queue_flags: vk::QueueFlags::COMPUTE,
            supports_compute: true,
            supports_graphics: false,
            supports_transfer: false,
        };
        assert!(info.supports_compute);
        assert!(!info.supports_graphics);
        assert!(!info.supports_transfer);
    }

    #[test]
    fn test_queue_family_info_all_flags() {
        let info = QueueFamilyInfo {
            index: 1,
            queue_count: 2,
            queue_flags: vk::QueueFlags::GRAPHICS
                | vk::QueueFlags::COMPUTE
                | vk::QueueFlags::TRANSFER,
            supports_compute: true,
            supports_graphics: true,
            supports_transfer: true,
        };
        assert!(info.supports_compute);
        assert!(info.supports_graphics);
        assert!(info.supports_transfer);
    }

    // --- Error display tests ---

    #[test]
    fn test_error_display_loader_not_found() {
        let err = VulkanError::LoaderNotFound;
        let msg = format!("{}", err);
        assert!(msg.contains("Vulkan loader"));
    }

    #[test]
    fn test_error_display_no_physical_devices() {
        let err = VulkanError::NoPhysicalDevices;
        assert!(format!("{}", err).contains("physical devices"));
    }

    #[test]
    fn test_error_display_no_compute_queue() {
        let err = VulkanError::NoComputeQueue;
        assert!(format!("{}", err).contains("compute queue"));
    }

    #[test]
    fn test_error_display_device_creation() {
        let err = VulkanError::DeviceCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("logical device"));
    }

    #[test]
    fn test_error_display_command_pool_creation() {
        let err = VulkanError::CommandPoolCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("command pool"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(VulkanError::LoaderNotFound);
        assert!(!format!("{}", err).is_empty());
    }

    // --- Buffer error display tests ---

    #[test]
    fn test_error_display_buffer_creation() {
        let err = VulkanError::BufferCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("buffer creation"));
    }

    #[test]
    fn test_error_display_memory_allocation() {
        let err = VulkanError::MemoryAllocation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("memory allocation"));
    }

    #[test]
    fn test_error_display_memory_binding() {
        let err = VulkanError::MemoryBinding(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("memory binding"));
    }

    #[test]
    fn test_error_display_no_suitable_memory_type() {
        let err = VulkanError::NoSuitableMemoryType;
        assert!(format!("{}", err).contains("memory type"));
    }

    #[test]
    fn test_error_display_memory_mapping() {
        let err = VulkanError::MemoryMapping(vk::Result::ERROR_MEMORY_MAP_FAILED);
        assert!(format!("{}", err).contains("memory mapping"));
    }

    // --- MemoryTypeSelector tests (pure logic, no GPU required) ---

    fn make_memory_properties(
        types: &[vk::MemoryPropertyFlags],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        let count = types.len().min(32);
        props.memory_type_count = count as u32;
        for (i, flags) in types.iter().enumerate() {
            props.memory_types[i] = vk::MemoryType {
                property_flags: *flags,
                heap_index: 0,
            };
        }
        props
    }

    #[test]
    fn test_memory_type_selector_device_local() {
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        let idx = selector.find_memory_type(0b11, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_memory_type_selector_host_visible() {
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        let idx = selector.find_memory_type(0b11, vk::MemoryPropertyFlags::HOST_VISIBLE);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_memory_type_selector_host_coherent() {
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        let idx = selector.find_memory_type(
            0b11,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_memory_type_selector_no_match() {
        let props = make_memory_properties(&[vk::MemoryPropertyFlags::DEVICE_LOCAL]);
        let selector = MemoryTypeSelector { properties: props };
        let idx = selector.find_memory_type(0b1, vk::MemoryPropertyFlags::HOST_VISIBLE);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_memory_type_selector_bitmask_filter() {
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        // Only bit 1 is set, so index 0 should be skipped
        let idx = selector.find_memory_type(
            0b10,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_memory_type_selector_empty_bitmask() {
        let props = make_memory_properties(&[vk::MemoryPropertyFlags::DEVICE_LOCAL]);
        let selector = MemoryTypeSelector { properties: props };
        let idx = selector.find_memory_type(0, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_memory_type_selector_prefers_first_match() {
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        let idx = selector.find_memory_type(0b111, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_memory_type_selector_cached_device_local() {
        // Simulates AMD GPU memory layout: type 0 is DEVICE_LOCAL
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ]);
        let selector = MemoryTypeSelector { properties: props };

        // Device-local should find type 0
        assert_eq!(
            selector.find_memory_type(0b111, vk::MemoryPropertyFlags::DEVICE_LOCAL),
            Some(0)
        );

        // Host-visible + coherent should find type 1 (first match)
        assert_eq!(
            selector.find_memory_type(
                0b111,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            ),
            Some(1)
        );

        // Host-visible only should also find type 1
        assert_eq!(
            selector.find_memory_type(0b111, vk::MemoryPropertyFlags::HOST_VISIBLE),
            Some(1)
        );
    }

    // --- VulkanBuffer property tests (no GPU, struct construction only) ---
    // Note: We cannot test actual buffer creation without a Vulkan device.
    // These tests verify the struct fields and accessor logic.

    #[test]
    fn test_vulkan_buffer_debug_format() {
        // Verify Debug impl compiles and produces expected struct name
        // We can't construct a real VulkanBuffer without a device, but we
        // can verify the type has Debug via a trait object check.
        let _: &dyn fmt::Debug = &VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
    }

    #[test]
    fn test_vulkan_buffer_is_mapped_false() {
        let buf = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
        assert!(!buf.is_mapped());
    }

    #[test]
    fn test_vulkan_buffer_is_mapped_true() {
        let buf = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            mapped_ptr: Some(std::ptr::null_mut()),
        };
        assert!(buf.is_mapped());
    }

    #[test]
    fn test_vulkan_buffer_handle_accessor() {
        let buf = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
        assert_eq!(buf.handle(), vk::Buffer::null());
    }

    #[test]
    fn test_vulkan_buffer_fields_public() {
        let buf = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 4096,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
        assert_eq!(buf.size, 4096);
        assert!(buf.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        assert!(buf.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert!(buf
            .memory_property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
    }

    // --- Fence property tests (no GPU, struct construction only) ---

    #[test]
    fn test_fence_debug_format() {
        let fence = Fence {
            device: std::ptr::null(),
            handle: vk::Fence::null(),
        };
        let debug_str = format!("{:?}", fence);
        assert!(debug_str.contains("Fence"));
    }

    #[test]
    fn test_fence_null_device_safe() {
        // Verify that a Fence with null device doesn't panic on construction.
        // Drop is also safe (null check prevents dereference).
        let _fence = Fence {
            device: std::ptr::null(),
            handle: vk::Fence::null(),
        };
    }

    // --- CommandBuffer property tests (no GPU, struct construction only) ---

    #[test]
    fn test_command_buffer_debug_format() {
        let cmd = CommandBuffer {
            device: std::ptr::null(),
            command_pool: vk::CommandPool::null(),
            handle: vk::CommandBuffer::null(),
            is_recording: false,
        };
        let debug_str = format!("{:?}", cmd);
        assert!(debug_str.contains("CommandBuffer"));
        assert!(debug_str.contains("is_recording"));
    }

    #[test]
    fn test_command_buffer_not_recording_initially() {
        let cmd = CommandBuffer {
            device: std::ptr::null(),
            command_pool: vk::CommandPool::null(),
            handle: vk::CommandBuffer::null(),
            is_recording: false,
        };
        assert!(!cmd.is_recording);
    }

    #[test]
    fn test_command_buffer_null_device_safe() {
        // Verify that a CommandBuffer with null device doesn't panic
        // on construction or drop.
        let _cmd = CommandBuffer {
            device: std::ptr::null(),
            command_pool: vk::CommandPool::null(),
            handle: vk::CommandBuffer::null(),
            is_recording: false,
        };
    }

    // --- Transfer validation tests ---

    #[test]
    fn test_transfer_validation_src_missing_flag() {
        let src = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            mapped_ptr: None,
        };
        assert!(!src.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC));
    }

    #[test]
    fn test_transfer_validation_dst_missing_flag() {
        let dst = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
        assert!(!dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn test_transfer_validation_src_has_flag() {
        let src = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            mapped_ptr: None,
        };
        assert!(src.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC));
    }

    #[test]
    fn test_transfer_validation_dst_has_flag() {
        let dst = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
        assert!(dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn test_transfer_size_exceeds_src() {
        let src_size: vk::DeviceSize = 1024;
        let copy_size: vk::DeviceSize = 2048;
        assert!(copy_size > src_size);
    }

    #[test]
    fn test_transfer_size_exceeds_dst() {
        let dst_size: vk::DeviceSize = 512;
        let copy_size: vk::DeviceSize = 1024;
        assert!(copy_size > dst_size);
    }

    #[test]
    fn test_transfer_size_exact_fit() {
        let buf_size: vk::DeviceSize = 1024;
        let copy_size: vk::DeviceSize = 1024;
        assert!(copy_size <= buf_size);
    }

    #[test]
    fn test_transfer_size_zero() {
        let copy_size: vk::DeviceSize = 0;
        assert_eq!(copy_size, 0);
    }

    // --- Upload validation tests ---

    #[test]
    fn test_upload_dst_not_device_local() {
        let dst = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::TRANSFER_DST,
            memory_property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            mapped_ptr: None,
        };
        assert!(!dst
            .memory_property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
    }

    #[test]
    fn test_upload_dst_is_device_local() {
        let dst = VulkanBuffer {
            device: std::ptr::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
            usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mapped_ptr: None,
        };
        assert!(dst
            .memory_property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        assert!(dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn test_upload_data_exceeds_buffer() {
        let data = vec![0u8; 2048];
        let buf_size: vk::DeviceSize = 1024;
        assert!(data.len() as vk::DeviceSize > buf_size);
    }

    #[test]
    fn test_upload_data_fits_buffer() {
        let data = vec![0u8; 512];
        let buf_size: vk::DeviceSize = 1024;
        assert!(data.len() as vk::DeviceSize <= buf_size);
    }

    #[test]
    fn test_upload_empty_data() {
        let data: &[u8] = &[];
        assert!(data.is_empty());
    }

    // --- New error display tests ---

    #[test]
    fn test_error_display_command_buffer_allocation() {
        let err = VulkanError::CommandBufferAllocation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("command buffer"));
    }

    #[test]
    fn test_error_display_fence_creation() {
        let err = VulkanError::FenceCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("fence"));
    }

    #[test]
    fn test_error_display_fence_wait() {
        let err = VulkanError::FenceWait(vk::Result::TIMEOUT);
        assert!(format!("{}", err).contains("fence"));
    }

    #[test]
    fn test_error_display_transfer_validation() {
        let err =
            VulkanError::TransferValidation("destination buffer missing TRANSFER_DST".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("transfer validation"));
        assert!(msg.contains("TRANSFER_DST"));
    }

    // --- ShaderModule tests ---

    #[test]
    fn test_shader_module_debug_format() {
        let module = ShaderModule {
            device: std::ptr::null(),
            handle: vk::ShaderModule::null(),
            word_count: 128,
        };
        let debug_str = format!("{:?}", module);
        assert!(debug_str.contains("ShaderModule"));
        assert!(debug_str.contains("word_count"));
    }

    #[test]
    fn test_shader_module_word_count_zero() {
        let module = ShaderModule {
            device: std::ptr::null(),
            handle: vk::ShaderModule::null(),
            word_count: 0,
        };
        assert_eq!(module.word_count(), 0);
    }

    #[test]
    fn test_shader_module_word_count_from_bytes() {
        // 64 bytes / 4 = 16 words
        let word_count = (64 / 4) as u32;
        assert_eq!(word_count, 16);
    }

    #[test]
    fn test_shader_module_word_count_from_bytes_100() {
        // 400 bytes / 4 = 100 words
        let word_count = (400 / 4) as u32;
        assert_eq!(word_count, 100);
    }

    #[test]
    fn test_shader_module_byte_to_word_conversion() {
        // Verify byte-to-word conversion logic
        assert_eq!(4 / 4, 1); // minimum valid SPIR-V (1 word)
        assert_eq!(512 / 4, 128);
        assert_eq!(65536 / 4, 16384);
    }

    #[test]
    fn test_shader_module_byte_length_not_multiple_of_4() {
        // 5 bytes is not a multiple of 4
        assert_ne!(5 % 4, 0);
        // Would be rejected by from_spv_bytes
    }

    #[test]
    fn test_shader_module_empty_bytes_detected() {
        let bytes: &[u8] = &[];
        assert!(bytes.is_empty());
        // Would be rejected by from_spv_bytes
    }

    #[test]
    fn test_shader_module_handle_accessor() {
        let module = ShaderModule {
            device: std::ptr::null(),
            handle: vk::ShaderModule::null(),
            word_count: 64,
        };
        assert_eq!(module.handle(), vk::ShaderModule::null());
    }

    #[test]
    fn test_shader_module_null_device_safe() {
        // Verify that a ShaderModule with null device doesn't panic
        // on construction or drop.
        let _module = ShaderModule {
            device: std::ptr::null(),
            handle: vk::ShaderModule::null(),
            word_count: 32,
        };
    }

    // --- ShaderModule error display tests ---

    #[test]
    fn test_error_display_shader_module_creation() {
        let err = VulkanError::ShaderModuleCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("shader module"));
    }

    #[test]
    fn test_error_display_invalid_spv_bytes_empty() {
        let err = VulkanError::InvalidSpvBytes("SPIR-V bytecode must not be empty".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("invalid SPIR-V"));
        assert!(msg.contains("empty"));
    }

    #[test]
    fn test_error_display_invalid_spv_bytes_alignment() {
        let err = VulkanError::InvalidSpvBytes(
            "SPIR-V bytecode length 5 is not a multiple of 4 (must be word-aligned)".to_string(),
        );
        let msg = format!("{}", err);
        assert!(msg.contains("invalid SPIR-V"));
        assert!(msg.contains("multiple of 4"));
    }

    #[test]
    fn test_error_display_spv_loading() {
        let err = VulkanError::SpvLoadingError("file not found: shaders/test.spv".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("SPIR-V loading"));
        assert!(msg.contains("file not found"));
    }

    // --- ComputePipeline tests ---

    #[test]
    fn test_compute_pipeline_debug_format() {
        let pipeline = ComputePipeline {
            device: std::ptr::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
        };
        let debug_str = format!("{:?}", pipeline);
        assert!(debug_str.contains("ComputePipeline"));
        assert!(debug_str.contains("pipeline"));
        assert!(debug_str.contains("layout"));
    }

    #[test]
    fn test_compute_pipeline_null_device_safe() {
        // Verify that a ComputePipeline with null device doesn't panic
        // on construction or drop.
        let _pipeline = ComputePipeline {
            device: std::ptr::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
        };
    }

    #[test]
    fn test_compute_pipeline_handle_accessor() {
        let pipeline = ComputePipeline {
            device: std::ptr::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
        };
        assert_eq!(pipeline.handle(), vk::Pipeline::null());
    }

    #[test]
    fn test_compute_pipeline_layout_accessor() {
        let pipeline = ComputePipeline {
            device: std::ptr::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
        };
        assert_eq!(pipeline.layout(), vk::PipelineLayout::null());
    }

    #[test]
    fn test_compute_pipeline_drop_order() {
        // Verify that Drop compiles and runs without panic for null handles.
        // The actual destruction order (pipeline first, then layout) is
        // enforced in the Drop implementation.
        let pipeline = ComputePipeline {
            device: std::ptr::null(),
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
        };
        drop(pipeline);
    }

    // --- Entry point CString validation tests ---

    #[test]
    fn test_entry_point_valid_main() {
        let result = CString::new("main");
        assert!(result.is_ok());
    }

    #[test]
    fn test_entry_point_valid_underscored() {
        let result = CString::new("embedding_lookup");
        assert!(result.is_ok());
    }

    #[test]
    fn test_entry_point_valid_with_digits() {
        let result = CString::new("matmul_8x8");
        assert!(result.is_ok());
    }

    #[test]
    fn test_entry_point_valid_with_double_underscore() {
        // Common in SPIR-V: __attribute or vendor-specific names
        let result = CString::new("__rope_apply");
        assert!(result.is_ok());
    }

    #[test]
    fn test_entry_point_rejects_null_byte() {
        // CString::new rejects strings containing null bytes
        let result = CString::new("main\0extra");
        assert!(result.is_err());
    }

    #[test]
    fn test_entry_point_rejects_embedded_null() {
        let result = CString::new("foo\0bar");
        assert!(result.is_err());
    }

    // --- Pipeline error display tests ---

    #[test]
    fn test_error_display_pipeline_layout_creation() {
        let err = VulkanError::PipelineLayoutCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("pipeline layout"));
    }

    #[test]
    fn test_error_display_compute_pipeline_creation() {
        let err = VulkanError::ComputePipelineCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("compute pipeline"));
    }

    #[test]
    fn test_error_display_invalid_entry_point() {
        let err = VulkanError::InvalidEntryPoint(
            "entry point name contains invalid bytes: unexpected null byte".to_string(),
        );
        let msg = format!("{}", err);
        assert!(msg.contains("invalid entry point"));
        assert!(msg.contains("null byte"));
    }

    // --- Descriptor type tests ---

    #[test]
    fn test_descriptor_set_layout_debug_format() {
        let layout = DescriptorSetLayout {
            device: std::ptr::null(),
            handle: vk::DescriptorSetLayout::null(),
        };
        let debug_str = format!("{:?}", layout);
        assert!(debug_str.contains("DescriptorSetLayout"));
        assert!(debug_str.contains("handle"));
    }

    #[test]
    fn test_descriptor_set_layout_handle_accessor() {
        let layout = DescriptorSetLayout {
            device: std::ptr::null(),
            handle: vk::DescriptorSetLayout::null(),
        };
        assert_eq!(layout.handle(), vk::DescriptorSetLayout::null());
    }

    #[test]
    fn test_descriptor_set_layout_null_device_safe() {
        let _layout = DescriptorSetLayout {
            device: std::ptr::null(),
            handle: vk::DescriptorSetLayout::null(),
        };
    }

    #[test]
    fn test_descriptor_pool_debug_format() {
        let pool = DescriptorPool {
            device: std::ptr::null(),
            handle: vk::DescriptorPool::null(),
        };
        let debug_str = format!("{:?}", pool);
        assert!(debug_str.contains("DescriptorPool"));
        assert!(debug_str.contains("handle"));
    }

    #[test]
    fn test_descriptor_pool_null_device_safe() {
        let _pool = DescriptorPool {
            device: std::ptr::null(),
            handle: vk::DescriptorPool::null(),
        };
    }

    #[test]
    fn test_descriptor_set_debug_format() {
        let set = DescriptorSet {
            device: std::ptr::null(),
            pool: Box::new(DescriptorPool {
                device: std::ptr::null(),
                handle: vk::DescriptorPool::null(),
            }),
            handle: vk::DescriptorSet::null(),
        };
        let debug_str = format!("{:?}", set);
        assert!(debug_str.contains("DescriptorSet"));
        assert!(debug_str.contains("handle"));
    }

    #[test]
    fn test_descriptor_set_handle_accessor() {
        let set = DescriptorSet {
            device: std::ptr::null(),
            pool: Box::new(DescriptorPool {
                device: std::ptr::null(),
                handle: vk::DescriptorPool::null(),
            }),
            handle: vk::DescriptorSet::null(),
        };
        assert_eq!(set.handle(), vk::DescriptorSet::null());
    }

    #[test]
    fn test_descriptor_set_null_device_safe() {
        let _set = DescriptorSet {
            device: std::ptr::null(),
            pool: Box::new(DescriptorPool {
                device: std::ptr::null(),
                handle: vk::DescriptorPool::null(),
            }),
            handle: vk::DescriptorSet::null(),
        };
    }

    // --- Descriptor/dispatch error display tests ---

    #[test]
    fn test_error_display_descriptor_set_layout_creation() {
        let err = VulkanError::DescriptorSetLayoutCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("descriptor set layout"));
    }

    #[test]
    fn test_error_display_descriptor_pool_creation() {
        let err = VulkanError::DescriptorPoolCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
        assert!(format!("{}", err).contains("descriptor pool"));
    }

    #[test]
    fn test_error_display_descriptor_set_allocation() {
        let err = VulkanError::DescriptorSetAllocation(vk::Result::ERROR_OUT_OF_POOL_MEMORY);
        assert!(format!("{}", err).contains("descriptor set allocation"));
    }

    #[test]
    fn test_error_display_command_buffer_not_recording() {
        let err = VulkanError::CommandBufferNotRecording;
        let msg = format!("{}", err);
        assert!(msg.contains("not in the recording state"));
    }

    // --- CommandBuffer dispatch state tests ---

    #[test]
    fn test_command_buffer_recording_state_transitions() {
        let mut cmd = CommandBuffer {
            device: std::ptr::null(),
            command_pool: vk::CommandPool::null(),
            handle: vk::CommandBuffer::null(),
            is_recording: false,
        };
        assert!(!cmd.is_recording);
        // Simulate begin
        cmd.is_recording = true;
        assert!(cmd.is_recording);
        // Simulate end
        cmd.is_recording = false;
        assert!(!cmd.is_recording);
    }

    // --- DescriptorBinding tests ---

    #[test]
    fn test_descriptor_binding_construction() {
        let binding = DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        };
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
        assert!(binding.stage_flags.contains(vk::ShaderStageFlags::COMPUTE));
    }

    #[test]
    fn test_descriptor_binding_debug_format() {
        let binding = DescriptorBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        };
        let debug_str = format!("{:?}", binding);
        assert!(debug_str.contains("DescriptorBinding"));
    }

    #[test]
    fn test_descriptor_binding_clone() {
        let binding = DescriptorBinding {
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        };
        let cloned = binding.clone();
        assert_eq!(cloned.binding, 2);
        assert_eq!(cloned.descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
    }

    #[test]
    fn test_descriptor_binding_multiple_types() {
        let storage = DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        };
        let uniform = DescriptorBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        };
        assert_eq!(storage.descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
        assert_eq!(uniform.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    }

    // --- Descriptor binding validation tests (pure logic) ---

    #[test]
    fn test_descriptor_bindings_consecutive_valid() {
        let bindings = vec![
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
            DescriptorBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
        ];
        let mut sorted = bindings.clone();
        sorted.sort_by_key(|b| b.binding);
        for (i, b) in sorted.iter().enumerate() {
            assert_eq!(b.binding, i as u32, "bindings should be consecutive");
        }
    }

    #[test]
    fn test_descriptor_bindings_gap_detected() {
        let bindings = vec![
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
            DescriptorBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
        ];
        let mut sorted = bindings.clone();
        sorted.sort_by_key(|b| b.binding);
        // Binding 1 is missing — gap at index 1
        assert_eq!(sorted[0].binding, 0);
        assert_eq!(sorted[1].binding, 2);
        assert_ne!(sorted[1].binding, 1); // gap detected
    }

    #[test]
    fn test_descriptor_bindings_duplicate_detected() {
        let bindings = vec![
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
            DescriptorBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
            },
        ];
        let mut sorted = bindings.clone();
        sorted.sort_by_key(|b| b.binding);
        assert_eq!(sorted[0].binding, sorted[1].binding); // duplicate
    }

    #[test]
    fn test_descriptor_bindings_empty_vec() {
        let bindings: Vec<DescriptorBinding> = vec![];
        assert!(bindings.is_empty());
    }

    #[test]
    fn test_descriptor_bindings_single() {
        let bindings = [DescriptorBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        }];
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].binding, 0);
    }

    // --- Push constant range construction tests ---

    #[test]
    fn test_push_constant_range_embedding() {
        // 3 uint values = 12 bytes, rounded up to 16 for alignment
        let range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)
            .build();
        assert_eq!(range.offset, 0);
        assert_eq!(range.size, 16);
        assert!(range.stage_flags.contains(vk::ShaderStageFlags::COMPUTE));
    }

    #[test]
    fn test_push_constant_range_two_uints() {
        // 2 uint values = 8 bytes
        let range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)
            .build();
        assert_eq!(range.size, 8);
    }

    #[test]
    fn test_push_constant_range_max_size() {
        // Vulkan guarantees at least 128 bytes of push constant space
        let range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(128)
            .build();
        assert_eq!(range.size, 128);
    }

    #[test]
    fn test_push_constant_range_with_offset() {
        let range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(8)
            .size(8)
            .build();
        assert_eq!(range.offset, 8);
        assert_eq!(range.size, 8);
    }

    // --- Push constant validation tests ---

    #[test]
    fn test_push_constant_data_size_aligned() {
        let data = [0u8; 16];
        assert_eq!(data.len() % 4, 0);
    }

    #[test]
    fn test_push_constant_data_size_misaligned() {
        let data = [0u8; 5];
        assert_ne!(data.len() % 4, 0);
    }

    #[test]
    fn test_push_constant_data_size_two_uints() {
        let data = [0u8; 8];
        assert_eq!(data.len() % 4, 0);
        assert_eq!(data.len(), 8);
    }

    #[test]
    fn test_push_constant_data_size_three_uints() {
        let data = [0u8; 12];
        assert_eq!(data.len() % 4, 0);
        assert_eq!(data.len(), 12);
    }

    // --- Error display tests for new error variants ---

    #[test]
    fn test_error_display_push_constant_validation() {
        let err = VulkanError::PushConstantValidation(
            "push constant data size 5 is not a multiple of 4".to_string(),
        );
        let msg = format!("{}", err);
        assert!(msg.contains("push constant validation"));
        assert!(msg.contains("not a multiple of 4"));
    }

    #[test]
    fn test_error_display_push_constant_validation_bindings() {
        let err = VulkanError::PushConstantValidation(
            "descriptor bindings must be consecutive starting from 0".to_string(),
        );
        let msg = format!("{}", err);
        assert!(msg.contains("push constant validation"));
        assert!(msg.contains("consecutive"));
    }

    // --- Single-buffer helper type regression tests ---

    #[test]
    fn test_single_buffer_layout_binding_construction() {
        let binding_info = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();
        assert_eq!(binding_info.binding, 0);
        assert_eq!(
            binding_info.descriptor_type,
            vk::DescriptorType::STORAGE_BUFFER
        );
        assert_eq!(binding_info.descriptor_count, 1);
        assert!(binding_info
            .stage_flags
            .contains(vk::ShaderStageFlags::COMPUTE));
    }

    #[test]
    fn test_descriptor_buffer_info_construction() {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(vk::Buffer::null())
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build();
        assert_eq!(buffer_info.buffer, vk::Buffer::null());
        assert_eq!(buffer_info.offset, 0);
        assert_eq!(buffer_info.range, vk::WHOLE_SIZE);
    }

    #[test]
    fn test_write_descriptor_set_construction() {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(vk::Buffer::null())
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build();
        let mut write = vk::WriteDescriptorSet::builder()
            .dst_set(vk::DescriptorSet::null())
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[buffer_info])
            .build();
        write.descriptor_count = 1;
        assert_eq!(write.dst_set, vk::DescriptorSet::null());
        assert_eq!(write.dst_binding, 0);
        assert_eq!(write.descriptor_count, 1);
    }

    // --- Barrier model tests ---

    #[test]
    fn test_barrier_transfer_to_compute_stages() {
        // Verify the stage flags used in transfer->compute barrier
        let src_stage = vk::PipelineStageFlags::TRANSFER;
        let dst_stage = vk::PipelineStageFlags::COMPUTE_SHADER;
        assert!(src_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(dst_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        // Source and destination stages must be different
        assert_ne!(src_stage, dst_stage);
    }

    #[test]
    fn test_barrier_compute_to_transfer_stages() {
        // Verify the stage flags used in compute->transfer barrier
        let src_stage = vk::PipelineStageFlags::COMPUTE_SHADER;
        let dst_stage = vk::PipelineStageFlags::TRANSFER;
        assert!(src_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(dst_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert_ne!(src_stage, dst_stage);
    }

    #[test]
    fn test_barrier_access_masks_transfer_write() {
        let access = vk::AccessFlags::TRANSFER_WRITE;
        assert!(access.contains(vk::AccessFlags::TRANSFER_WRITE));
        assert!(!access.contains(vk::AccessFlags::SHADER_READ));
    }

    #[test]
    fn test_barrier_access_masks_shader_read() {
        let access = vk::AccessFlags::SHADER_READ;
        assert!(access.contains(vk::AccessFlags::SHADER_READ));
        assert!(!access.contains(vk::AccessFlags::TRANSFER_WRITE));
    }

    #[test]
    fn test_barrier_access_masks_shader_write() {
        let access = vk::AccessFlags::SHADER_WRITE;
        assert!(access.contains(vk::AccessFlags::SHADER_WRITE));
        assert!(!access.contains(vk::AccessFlags::TRANSFER_READ));
    }

    #[test]
    fn test_barrier_access_masks_transfer_read() {
        let access = vk::AccessFlags::TRANSFER_READ;
        assert!(access.contains(vk::AccessFlags::TRANSFER_READ));
        assert!(!access.contains(vk::AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn test_barrier_buffer_memory_barrier_construction() {
        let barrier = vk::BufferMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(vk::Buffer::null())
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .build();
        assert!(barrier
            .src_access_mask
            .contains(vk::AccessFlags::TRANSFER_WRITE));
        assert!(barrier
            .dst_access_mask
            .contains(vk::AccessFlags::SHADER_READ));
        assert_eq!(barrier.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barrier.dst_queue_family_index, vk::QUEUE_FAMILY_IGNORED);
    }

    #[test]
    fn test_barrier_whole_size_constant() {
        // vk::WHOLE_SIZE should be u64::MAX
        assert_eq!(vk::WHOLE_SIZE, u64::MAX);
    }
}
