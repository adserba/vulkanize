use ash::vk;
use vulkanize_runtime::embedding::{compare_f32, embedding_lookup_f32};
use vulkanize_vulkan_backend::*;

#[test]
#[ignore]
fn smoke_embedding_lookup_f32() {
    let ctx = VulkanContext::new().expect("failed to create Vulkan context");
    println!(
        "[1] Using device: {} ({})",
        ctx.physical_device_info().name,
        format_vendor_id(ctx.physical_device_info().vendor_id)
    );

    let vocab_size: u32 = 4;
    let embedding_dim: u32 = 64;
    let batch_size: u32 = 1;
    let token_id: u32 = 2;

    // Build synthetic weights: 0.0, 1.0, 2.0, ... column-major [embedding_dim, vocab_size]
    let total_weights = (vocab_size * embedding_dim) as usize;
    let weights: Vec<f32> = (0..total_weights).map(|i| i as f32).collect();
    let weights_bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Expected output for token_id=2: elements [128..192]
    let expected = embedding_lookup_f32(&weights, embedding_dim as usize, token_id as usize);

    // Use 4096-byte buffers to avoid GPU memory controller alignment issues
    let buf_size = 4096u64;

    // Create weight buffer: device-local, STORAGE_BUFFER | TRANSFER_DST
    let weight_buf = ctx
        .create_device_local_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .expect("failed to create weight buffer");
    println!("[2] Created weight buffer ({} bytes)", buf_size);

    // Create token ID buffer: device-local, STORAGE_BUFFER | TRANSFER_DST
    let mut token_data = vec![0u8; buf_size as usize];
    token_data[0..4].copy_from_slice(&token_id.to_le_bytes());
    let token_buf = ctx
        .create_device_local_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
        .expect("failed to create token buffer");
    println!("[3] Created token ID buffer");

    // Create output buffer: device-local, STORAGE_BUFFER | TRANSFER_SRC
    let output_buf = ctx
        .create_device_local_buffer(
            buf_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .expect("failed to create output buffer");
    println!("[4] Created output buffer ({} bytes)", buf_size);

    // Upload weights and token IDs
    ctx.upload_to_device_local(&weight_buf, &weights_bytes)
        .expect("failed to upload weights");
    ctx.upload_to_device_local(&token_buf, &token_data)
        .expect("failed to upload token IDs");
    println!("[5] Uploaded weights and token IDs");

    // Load shader
    let shader_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("shaders/embedding_lookup.spv");
    let shader_module = ctx
        .create_shader_module_from_spv_file(&shader_path)
        .expect("failed to load embedding_lookup.spv");
    println!("[6] Loaded shader");

    // Create descriptor layout with bindings 0, 1, 2
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
    let desc_layout = ctx
        .create_descriptor_set_layout_with_buffers(&bindings)
        .expect("failed to create descriptor set layout");
    println!("[7] Created descriptor set layout (3 bindings)");

    // Create pipeline with 12-byte push constant range
    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(12)
        .build();
    let pipeline = ctx
        .create_compute_pipeline_with_push_constants(
            &shader_module,
            "main",
            &[desc_layout.handle],
            push_constant_range,
        )
        .expect("failed to create compute pipeline");
    println!("[8] Created compute pipeline with push constants");

    // Create descriptor set with three buffers
    let desc_set = ctx
        .create_descriptor_set_with_buffers(
            &desc_layout,
            &[
                (0, &weight_buf, weights_bytes.len() as u64),
                (1, &token_buf, 4),
                (2, &output_buf, (embedding_dim * 4) as u64),
            ],
        )
        .expect("failed to create descriptor set");
    println!("[9] Created descriptor set");

    // Record command buffer
    let mut cmd = ctx
        .allocate_command_buffer()
        .expect("failed to allocate command buffer");
    cmd.begin().expect("failed to begin command buffer");

    // Barrier: transfer write -> shader read for weight and token buffers
    cmd.record_barrier_transfer_to_compute(&[&weight_buf, &token_buf])
        .expect("failed to record transfer->compute barrier");

    // Bind pipeline
    cmd.bind_compute_pipeline(&pipeline)
        .expect("failed to bind pipeline");

    // Push constants: vocab_size, embedding_dim, batch_size (12 bytes)
    let push_data = [
        vocab_size.to_le_bytes(),
        embedding_dim.to_le_bytes(),
        batch_size.to_le_bytes(),
    ]
    .concat();
    cmd.push_constants(pipeline.layout(), 0, &push_data)
        .expect("failed to push constants");

    // Bind descriptor set
    cmd.bind_descriptor_sets(pipeline.layout(), 0, &[desc_set.handle()])
        .expect("failed to bind descriptors");

    // Dispatch 1,1,1 (64 invocations, covers 64 embedding_dim elements for batch_size=1)
    cmd.dispatch(1, 1, 1).expect("failed to dispatch");

    // Barrier: shader write -> transfer read for output buffer
    cmd.record_barrier_compute_to_transfer(&[&output_buf])
        .expect("failed to record compute->transfer barrier");

    cmd.end().expect("failed to end command buffer");
    cmd.submit_and_wait(
        ctx.device().compute_queue,
        ctx.queue_family().queue_family_index,
    )
    .expect("failed to submit and wait");
    println!("[10] Dispatch completed");

    // Read back output
    ctx.wait_idle().expect("wait_idle failed");
    let output_bytes = ctx
        .readback_buffer_data(&output_buf, (embedding_dim * 4) as u64)
        .expect("failed to readback output");
    println!("[11] Read back {} bytes", output_bytes.len());

    // Convert bytes to f32
    let mut actual = Vec::with_capacity(embedding_dim as usize);
    for chunk in output_bytes.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        actual.push(val);
    }

    // Compare with CPU reference
    compare_f32(&actual, &expected, 1e-5).expect("GPU output mismatched CPU reference");
    println!("[12] Smoke test PASSED: embedding lookup matches CPU reference");
}
