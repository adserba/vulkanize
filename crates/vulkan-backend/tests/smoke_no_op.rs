use ash::vk;
use vulkanize_vulkan_backend::*;

#[test]
#[ignore]
fn smoke_no_op_dispatch() {
    let ctx = VulkanContext::new().expect("failed to create Vulkan context");
    println!(
        "[1] Using device: {} ({})",
        ctx.physical_device_info().name,
        format_vendor_id(ctx.physical_device_info().vendor_id)
    );

    let buf_size = 4096;

    let device_buf = ctx
        .create_device_local_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .expect("failed to create device-local buffer");
    println!("[2] Created device-local buffer");

    let initial_data: u32 = 0xDEADBEEF;
    ctx.upload_to_device_local(&device_buf, &initial_data.to_le_bytes())
        .expect("failed to upload");
    println!("[3] Uploaded initial data");

    // Verify upload
    let pre_result = ctx
        .readback_buffer_data(&device_buf, 4)
        .expect("failed to readback pre-dispatch");
    let pre_u32 = u32::from_le_bytes([pre_result[0], pre_result[1], pre_result[2], pre_result[3]]);
    assert_eq!(pre_u32, 0xDEADBEEF, "upload verification failed");
    println!("[4] Verified upload: 0x{:08X}", pre_u32);

    let shader_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("shaders/no_op.spv");
    let shader_module = ctx
        .create_shader_module_from_spv_file(&shader_path)
        .expect("failed to load no_op.spv");
    println!("[5] Loaded shader");

    let desc_layout = ctx
        .create_descriptor_set_layout_with_buffer(0)
        .expect("failed to create descriptor set layout");
    println!("[6] Created descriptor set layout");

    let pipeline = ctx
        .create_compute_pipeline_with_layouts(&shader_module, "main", &[desc_layout.handle])
        .expect("failed to create compute pipeline");
    println!("[7] Created compute pipeline");

    let desc_set = ctx
        .create_descriptor_set_with_buffer(&desc_layout, 0, &device_buf, vk::WHOLE_SIZE)
        .expect("failed to create descriptor set");
    println!("[8] Created descriptor set");

    // Record WITHOUT barriers - just bind + dispatch
    let mut cmd = ctx
        .allocate_command_buffer()
        .expect("failed to allocate command buffer");
    cmd.begin().expect("failed to begin");
    cmd.bind_compute_pipeline(&pipeline)
        .expect("failed to bind pipeline");
    cmd.bind_descriptor_sets(pipeline.layout(), 0, &[desc_set.handle()])
        .expect("failed to bind descriptors");
    cmd.dispatch(1, 1, 1).expect("failed to dispatch");
    cmd.end().expect("failed to end");
    cmd.submit_and_wait(
        ctx.device().compute_queue,
        ctx.queue_family().queue_family_index,
    )
    .expect("failed to submit and wait");
    println!("[9] Dispatch completed");

    // Wait and read back
    ctx.wait_idle().expect("wait_idle failed");
    let result = ctx
        .readback_buffer_data(&device_buf, 4)
        .expect("failed to readback");
    let result_u32 = u32::from_le_bytes([result[0], result[1], result[2], result[3]]);

    assert_eq!(
        result_u32, 0x12345678,
        "shader should write 0x12345678, got 0x{:08X}",
        result_u32
    );

    println!("[10] Smoke test PASSED: shader wrote 0x{:08X}", result_u32);
}
