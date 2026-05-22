#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer EmbeddingTable {
    float weights[];
} embedding_table;

layout(set = 0, binding = 1, std430) readonly buffer TokenIds {
    uint ids[];
} token_ids;

layout(set = 0, binding = 2, std430) buffer OutputBuffer {
    float output_data[];
} output_buf;

layout(push_constant) uniform PushConstants {
    uint vocab_size;
    uint embedding_dim;
    uint batch_size;
} params;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = params.batch_size * params.embedding_dim;
    if (gid >= total) {
        return;
    }

    uint token_idx = gid / params.embedding_dim;
    uint dim_idx = gid % params.embedding_dim;

    if (token_idx >= params.batch_size) {
        return;
    }

    uint token_id = token_ids.ids[token_idx];

    if (token_id >= params.vocab_size) {
        output_buf.output_data[gid] = 0.0;
        return;
    }

    uint weight_idx = token_id * params.embedding_dim + dim_idx;
    output_buf.output_data[gid] = embedding_table.weights[weight_idx];
}
