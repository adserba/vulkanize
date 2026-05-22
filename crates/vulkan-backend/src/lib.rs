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

/// Handles for the queues and command pool acquired from a logical device.
pub struct QueueHandles {
    /// The compute queue handle (raw Vulkan handle).
    pub compute_queue: vk::Queue,
    /// Index of the queue family that owns `compute_queue`.
    pub queue_family_index: u32,
    /// Index within the family.
    pub queue_index: u32,
    /// Command pool bound to the compute queue family.
    pub command_pool: vk::CommandPool,
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
    /// Logical device.
    device: ash::Device,
    /// Compute queue and command pool.
    queues: QueueHandles,
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
        let (device, queues) = create_device(&instance, physical_device)?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            physical_device_info: info,
            device,
            queues,
        })
    }

    // -- Accessors ----------------------------------------------------------

    pub fn entry(&self) -> &Entry {
        &self.entry
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn physical_device_info(&self) -> &PhysicalDeviceInfo {
        &self.physical_device_info
    }

    pub fn queues(&self) -> &QueueHandles {
        &self.queues
    }

    /// Wait for the device to finish all pending work.
    ///
    /// Useful for cleanup / shutdown.  Avoid in hot paths.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        unsafe { self.device.device_wait_idle() }
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
            self.device.destroy_command_pool(self.queues.command_pool, None);
            self.device.destroy_device(None);
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
            .map_err(|e| VulkanError::InstanceCreation(e))?;

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
            .map_err(|e| VulkanError::InstanceCreation(e))?
    };

    Ok(instance)
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------

fn get_driver_name(instance: &Instance, pd: vk::PhysicalDevice) -> String {
    // Use VkPhysicalDeviceProperties2 + pNext chaining to get driver name
    let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default();
    props2.p_next = &mut driver_props as *mut _ as *mut std::ffi::c_void;

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
// Logical device creation
// ---------------------------------------------------------------------------

fn create_device(
    instance: &Instance,
    pd: vk::PhysicalDevice,
) -> Result<(ash::Device, QueueHandles), VulkanError> {
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(pd)
    };

    // Find first compute queue family
    let queue_family_index = queue_families
        .iter()
        .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or(VulkanError::NoComputeQueue)? as u32;

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
            .map_err(|e| VulkanError::DeviceCreation(e))?
    };

    let compute_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    let command_pool = unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .map_err(|e| VulkanError::CommandPoolCreation(e))?
    };

    let queues = QueueHandles {
        compute_queue,
        queue_family_index,
        queue_index: 0,
        command_pool,
    };

    Ok((device, queues))
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
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(VulkanError::LoaderNotFound);
        assert!(!format!("{}", err).is_empty());
    }
}
