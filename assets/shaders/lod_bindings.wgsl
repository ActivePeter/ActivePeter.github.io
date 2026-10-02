#define_import_path lod_bindings

// The far-view material's own bindings plus the two cut tests, shared by the colour pass
// (`lod_material.wgsl`) and the shadow/depth pass (`lod_prepass.wgsl`).
//
// They live in a module of their own because a shader may only carry one `@fragment` entry point:
// the prepass shader importing `lod_material` would pull that entry point in with it. Same shape as
// `bevy_pbr::pbr_bindings` and `bevy_pbr::pbr_prepass`.

struct LodParams {
    // min_x, min_z, max_x, max_z of the near-window footprint.
    window_rect: vec4<f32>,
    // min_y, max_y, finer_ring_enabled, window_enabled.
    window_y: vec4<f32>,
    // min_x, min_z, max_x, max_z of the next finer level's ring (ignored unless enabled).
    finer_rect: vec4<f32>,
    // fade_start, fade_end: where the far view mixes its shadow out (design §7.5).
    shadow_fade: vec4<f32>,
};

/// How far past the window box's high edge the cut reaches, in blocks. The near-window and far-view
/// surfaces are both exactly `surface_y + 1`, so an inclusive-by-a-hair test is what stops the two
/// coplanar tops from z-fighting. Small enough not to hide any real terrain.
const LOD_CUT_EPSILON: f32 = 0.01;

// StandardMaterial uses group(2). We reserve bindings 0-99 for it, so the extension starts at 100.
@group(2) @binding(100) var<uniform> lod_params: LodParams;
// The near block textures as one 2D array, in `LodAtlasLayer` order (`lod_mesh.rs`). A mesh vertex
// names the layer in `ATTRIBUTE_UV_1.x` and the face's world-aligned UV in `ATTRIBUTE_UV_0`; the
// tint the near material multiplies its texture by rides in `ATTRIBUTE_COLOR`. `texture * tint` is
// therefore the near material, per pixel (design §7.5) — one flat colour per block was what made
// the window edge a step from pixels to paint.
@group(2) @binding(101) var lod_atlas: texture_2d_array<f32>;
@group(2) @binding(102) var lod_atlas_sampler: sampler;

/// True when the fragment is inside the near-window box: the near chunks own that terrain.
fn inside_near_window(world: vec3<f32>) -> bool {
    let rect = lod_params.window_rect;
    let y = lod_params.window_y;
    let inside_xz = world.x >= rect.x && world.x < rect.z && world.z >= rect.y && world.z < rect.w;
    // The y test is inclusive by a hair: a far tile's top face sits at exactly the section box's
    // high edge (both are `surface + 1`), and a strict `y < max_y` left those two coplanar faces
    // fighting for depth, which showed as flickering patches on every flat top in the window.
    let inside_y = world.y >= y.x - LOD_CUT_EPSILON && world.y <= y.y + LOD_CUT_EPSILON;
    return inside_xz && inside_y;
}

/// True when the fragment is inside the ring of the next finer level.
fn inside_finer_ring(world: vec3<f32>) -> bool {
    let rect = lod_params.finer_rect;
    return world.x >= rect.x && world.x < rect.z && world.z >= rect.y && world.z < rect.w;
}
