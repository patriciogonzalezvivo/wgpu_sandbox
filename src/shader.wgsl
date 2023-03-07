// Vertex shader=
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = in.tex_coords;
    out.position = vec4<f32>(in.position, 1.0);
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    pad1: f32,
}
@group(1) @binding(0)
var<uniform> u: Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let pixel = 1.0/u.resolution.xy;
    var st : vec2<f32> = in.position.xy * pixel;
    var uv : vec2<f32> = in.tex_coords;
    let time = u.time;

    let pct = sin(time * 0.5) * 0.5 + 0.5;
    let res = pixel * (10.0 + 40.0 * pct) ;
    uv.y = 1.0 - uv.y;
    uv = (floor(uv / res) + 0.5) * res;
    
    color = textureSample(t_diffuse, s_diffuse, uv);
    // color = vec4<f32>(uv.x, uv.y, sin(time), 1.0);
    // color = vec4<f32>(st.x, st.y, 0.0, 1.0);

    return color;
}
