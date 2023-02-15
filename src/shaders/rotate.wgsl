@group(0) @binding(0) var input_texture : texture_2d<f32>;
@group(0) @binding(1) var rotated_value : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> rotation_matrix : mat3x3<f32>;
@group(0) @binding(3) var samp : sampler;

@compute
@workgroup_size(16, 16)
fn rotate_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let dimensions = textureDimensions(input_texture);
    let coords = vec2<i32>(global_id.xy);

    if coords.x >= dimensions.x || coords.y >= dimensions.y {
        return;
    }

    let dims = vec2<f32>(dimensions);
    let fcords = vec2<f32>(coords);

    let trans_coords = rotation_matrix * vec3<f32>((fcords - dims.xy * 0.5), 1.0);
    let sample_coords = trans_coords.xy + 2.0 * dims.xy;

    let color = textureSampleLevel(input_texture, samp, sample_coords, 0.0);

    let hsv = rgb2hsv(color.rgb);

    textureStore(rotated_value, coords.xy, vec4<f32>(hsv.z, 0.0, 0.0, 0.0));
}

fn rgb2hsv(pixel: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(pixel.bg, K.wz), vec4<f32>(pixel.gb, K.xy), step(pixel.b, pixel.g));
    let q = mix(vec4<f32>(p.xyw, pixel.r), vec4<f32>(pixel.r, p.yzx), step(p.x, pixel.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;

    return vec3<f32>(abs(q.z + (q.w - q.y)) / (6.0 * d + e), d / (q.x + e), q.x);
}
