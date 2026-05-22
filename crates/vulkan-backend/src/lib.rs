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
        let properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
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
            .field(
                "memory_property_flags",
                &self.memory_property_flags,
            )
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
    /// Maps the buffer, copies the data, and unmaps it.
    /// The buffer must have `HOST_VISIBLE` memory.
    pub fn write_data(&mut self, data: &[u8]) -> Result<(), VulkanError> {
        if data.len() > self.size as usize {
            return Err(VulkanError::MemoryMapping(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        }

        let ptr = self.map()?;
        unsafe {
            std::slice::from_raw_parts_mut(ptr, data.len()).copy_from_slice(data);
        }
        self.unmap()?;
        Ok(())
    }

    /// Write data into the buffer at a specific byte offset.
    ///
    /// Maps the buffer, copies the data at the given offset, and unmaps it.
    /// The buffer must have `HOST_VISIBLE` memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `offset + data.len()` exceeds the buffer size.
    pub fn write_at(
        &mut self,
        offset: vk::DeviceSize,
        data: &[u8],
    ) -> Result<(), VulkanError> {
        let end = offset
            .checked_add(data.len() as vk::DeviceSize)
            .ok_or(VulkanError::MemoryMapping(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;

        if end > self.size {
            return Err(VulkanError::MemoryMapping(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        }

        let ptr = self.map()?;
        unsafe {
            std::slice::from_raw_parts_mut(ptr.add(offset as usize), data.len())
                .copy_from_slice(data);
        }
        self.unmap()?;
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
            return Err(VulkanError::BufferCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
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
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| {
                    device.destroy_buffer(buffer, None);
                    VulkanError::MemoryAllocation(e)
                })?
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| {
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
        let (vulkan_device, queue_family, commands) =
            create_device(&instance, physical_device)?;
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
            return Err(VulkanError::BufferCreation(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
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

        let requirements =
            unsafe { self.device.handle().get_buffer_memory_requirements(buffer) };

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

    /// Wait for the device to finish all pending work.
    ///
    /// Useful for cleanup / shutdown.  Avoid in hot paths.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        unsafe { self.device.handle().device_wait_idle() }
    }

    /// Enumerate **all** physical devices on the system (useful for
    /// diagnostic / info commands).
    pub fn enumerate_physical_devices(&self) -> Result<Vec<PhysicalDeviceInfo>, VulkanError> {
        let phys_devices = unsafe {
            self.instance
                .enumerate_physical_devices()
        }.map_err(|_| VulkanError::NoPhysicalDevices)?;

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
            self.device.handle().destroy_command_pool(
                self.commands.command_pool,
                None,
            );
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

fn get_device_extensions(
    instance: &Instance,
    pd: vk::PhysicalDevice,
) -> Vec<String> {
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

fn build_physical_device_info(
    instance: &Instance,
    pd: vk::PhysicalDevice,
) -> PhysicalDeviceInfo {
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let name = unsafe {
        CStr::from_ptr(&props.device_name[0])
            .to_string_lossy()
            .into_owned()
    };

    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(pd)
    };

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
    let phys_devices = unsafe {
        instance
            .enumerate_physical_devices()
    }.map_err(|_| VulkanError::NoPhysicalDevices)?;

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
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(pd)
    };

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
    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos);

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
        assert_eq!(
            format_device_type(vk::PhysicalDeviceType::CPU),
            "CPU"
        );
    }

    #[test]
    fn test_format_device_type_other() {
        assert_eq!(
            format_device_type(vk::PhysicalDeviceType::OTHER),
            "Other"
        );
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
        assert_eq!(
            format_queue_flags(vk::QueueFlags::COMPUTE),
            "compute"
        );
    }

    #[test]
    fn test_format_queue_flags_graphics_compute() {
        let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        assert_eq!(format_queue_flags(flags), "graphics, compute");
    }

    #[test]
    fn test_format_queue_flags_all() {
        let flags =
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
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
        let idx =
            selector.find_memory_type(0b11, vk::MemoryPropertyFlags::HOST_VISIBLE);
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
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        let idx =
            selector.find_memory_type(0b1, vk::MemoryPropertyFlags::HOST_VISIBLE);
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
        let props = make_memory_properties(&[
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ]);
        let selector = MemoryTypeSelector { properties: props };
        let idx =
            selector.find_memory_type(0, vk::MemoryPropertyFlags::DEVICE_LOCAL);
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
            vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
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
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
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
}
