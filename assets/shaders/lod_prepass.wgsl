#define_import_path lod_prepass

// Shadow-pass (and depth-prepass) fragment shader for the far view (design §7.5).
//
// The shadow map is its own pass and knows nothing about the colour pass's cuts, so a far tile keeps
// casting even where the colour pass discards it — and the coarse levels merge to the highest block,
// so their shadow geometry sits at canopy height across the whole near window. The beach and the
// water next to the player ended up in the shadow of invisible far terrain: the sand went black
// (the evening ambient light alone is weak) and the sun's glitter left the water.
//
// This repeats the colour pass's two cuts and its leaf cut-out, then does what
// `bevy_pbr::pbr_prepass` does for a material without a normal map: write the depth that
// `DEPTH_CLAMP_ORTHO` needs and nothing else.

#import bevy_pbr::{pbr_bindings, pbr_types, prepass_io}
#import lod_bindings::{inside_finer_ring, inside_near_window, lod_atlas, lod_atlas_sampler, lod_params}

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(
    in: prepass_io::VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> prepass_io::FragmentOutput {
    // Sample before cutting: `textureSample` needs uniform control flow, and the cuts depend on the
    // fragment's position. Same order as the colour pass.
    let layer = i32(round(in.uv_b.x));
    let color = pbr_bindings::material.base_color * in.color
        * textureSample(lod_atlas, lod_atlas_sampler, in.uv, layer);

    if (lod_params.window_y.w > 0.5 && inside_near_window(in.world_position.xyz)) {
        discard;
    }
    if (lod_params.window_y.z > 0.5 && inside_finer_ring(in.world_position.xyz)) {
        discard;
    }

    // The far view keeps its leaves' holes out of the shadow map too, from the atlas sample rather
    // than the material's `base_color_texture`, which the far view does not use.
    let alpha_mode = pbr_bindings::material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
    if (alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK
        && color.a < pbr_bindings::material.alpha_cutoff) {
        discard;
    }

    var out: prepass_io::FragmentOutput;
#ifdef DEPTH_CLAMP_ORTHO
    out.frag_depth = in.clip_position_unclamped.z;
#endif // DEPTH_CLAMP_ORTHO
    return out;
}
#endif // PREPASS_FRAGMENT
