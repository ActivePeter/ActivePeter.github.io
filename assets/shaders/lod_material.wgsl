#define_import_path lod_material

// Far-view LOD tile material (design `docs/far_view_lod_design.md` §7.2).
//
// Three things differ from a plain StandardMaterial:
// - the near window is punched out of every tile with a hard cut at the box edge, so the far view
//   never fights the near chunks for depth. That box comes from a uniform, so walking around
//   rewrites 32 bytes instead of rebuilding a mesh.
// - vertex colours carry the per-block *tint* and the vertices carry the block face's UV plus the
//   atlas layer holding that face's near texture, so the far view draws the same pixels as the near
//   window instead of one flat colour per block (design §7.5).
// - one material draws every tile and level.
//
// Everything else is the standard PBR path, including fog, which
// `main_pass_post_lighting_processing` applies.

#import bevy_pbr::{
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_types,
}
#import bevy_pbr::mesh_view_bindings::view
#import lod_bindings::{lod_atlas, lod_atlas_sampler, lod_params, inside_finer_ring, inside_near_window}

#import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}


@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    if (lod_params.window_y.w > 0.5 && inside_near_window(in.world_position.xyz)) {
        // Hard cut: inside the box the near chunks draw the real terrain, so the far view steps
        // aside. Do not soften this edge with a dithered band *outside* the box — there is
        // nothing behind the far view out there, so every dropped fragment shows sky and the
        // horizon turns into a dot pattern. A band would have to live inside the box instead.
        discard;
    }
    if (lod_params.window_y.z > 0.5 && inside_finer_ring(in.world_position.xyz)) {
        // Every level keeps its whole ring (8x8 tiles on the two finest levels, 4x4 above), so
        // neighbouring levels overlap where their grids do not line up; the finer level wins
        // (design §5.2). Without this the coarse level's single-sample surface drew banks over the
        // finer level's water and rivers looked like land.
        discard;
    }

    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // The block's own texture, from the layer the vertex names. All four vertices of a merged quad
    // carry the same index, but `uv_b` is a normal (non-flat) varying — Bevy's `VertexOutput` is
    // its struct, so `@interpolate(flat)` is not ours to set — and `i32` truncates toward zero, so
    // an interpolated `1.9999999` would silently sample layer 1. `round` makes that impossible.
    let layer = i32(round(in.uv_b.x));
    pbr_input.material.base_color *= textureSample(lod_atlas, lod_atlas_sampler, in.uv, layer);

    // Far-view terrain is lit exactly like the near window: real normals, the same sun, the same
    // ambient light, and the same cascaded shadows (design §7.1/§7.5). The earlier "every face
    // points up" hack compensated for the fake cliffs the old single-height data model built along
    // banks; the column model does not have them, and the hack only made the far view's shading
    // disagree with the near window's along the boundary.
    //
    // The discard comes after the atlas sample so the leaves' cut-out holes are cut the same way the
    // near material's `AlphaMode::Mask` cuts them (`alpha_discard` tests the base colour's alpha).
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    var out: FragmentOutput;
    if ((pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        out.color = apply_pbr_lighting(pbr_input);

        // Shadow fade (design §7.5). The browser has one cascade, so the shadow map simply ends at
        // `shadow_fade.y`; without this the far view shows that end as a straight line of sudden
        // sunlight. The band sits entirely inside the far view (the near window ends around 72
        // blocks), so only far-view fragments pay for the second lighting evaluation, which is the
        // same one with the shadow-receiver flag cleared (that flag is what `apply_pbr_lighting`
        // tests before fetching the shadow).
        // `fade > 0` alone is true for everything past the fade band too — `smoothstep` saturates at
        // 1 — which meant the whole far view paid for a second lighting evaluation to mix in the
        // same result. Only fragments inside the band need it.
        let to_camera = distance(in.world_position.xyz, view.world_position.xyz);
        let fade = smoothstep(lod_params.shadow_fade.x, lod_params.shadow_fade.y, to_camera);
        if (fade > 0.0 && fade < 1.0) {
            var unshadowed = pbr_input;
            unshadowed.flags = unshadowed.flags & ~MESH_FLAGS_SHADOW_RECEIVER_BIT;
            out.color = mix(out.color, apply_pbr_lighting(unshadowed), fade);
        }
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
