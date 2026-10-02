#define_import_path water_material

// Extended StandardMaterial shader for water:
// - Animate the vanilla `water_still.png` texture (stacked vertically).
// - Add simple world-position based waves to perturb normals on top faces.
//
// This is intentionally "lightweight": works on WebGL2, avoids depth-texture tricks.

#import bevy_pbr::{
    pbr_functions,
    pbr_functions::{SampleBias, alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_bindings,
    pbr_types,
    mesh_view_bindings::{view, lights, view_transmission_texture, view_transmission_sampler},
}

#import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}

#ifdef TONEMAP_IN_SHADER
#import bevy_core_pipeline::tonemapping::approximate_inverse_tone_mapping
#endif

/// Average grey of `blocks/water_still.png` (all 32 frames, sRGB). The far-view water fades its
/// texture sample into this flat value with distance: that texture has no mipmaps, so a far tile
/// (whose UVs run for hundreds of texels) samples pure noise pixel by pixel.
const LOD_WATER_TEXTURE_AVERAGE: f32 = 0.70;

struct WaterParams {
    time: f32,
    wave_strength: f32,
    scroll_speed: f32,
    // Debug toggle / mode from JS query string.
    debug: f32,
    // Boat interior water mask (client-only).
    boat_mask_on: f32,
    boat_center_x: f32,
    boat_center_z: f32,
    boat_half_x: f32,
    boat_half_z: f32,
    boat_cos: f32,
    boat_sin: f32,
    // 1.0 for the far-view LOD water materials, 0.0 for the near material. The near material
    // leaves every field below zeroed, so its path is exactly what it was.
    lod_mode: f32,
    // min_x, min_z, max_x, max_z of the near-window footprint.
    window_rect: vec4<f32>,
    // min_y, max_y, finer_ring_enabled, window_enabled.
    window_y: vec4<f32>,
    // min_x, min_z, max_x, max_z of the next finer level's ring (ignored unless enabled).
    finer_rect: vec4<f32>,
};

// StandardMaterial uses group(2). We reserve bindings 0-99 for it, so the extension starts at 100.
@group(2) @binding(100) var<uniform> water_params: WaterParams;
@group(2) @binding(101) var water_noises_tex: texture_2d<f32>;
@group(2) @binding(102) var water_noises_samp: sampler;

/// True when the fragment is inside the near-window box: the near chunks own that terrain.
fn inside_near_window(world: vec3<f32>) -> bool {
    let rect = water_params.window_rect;
    let y = water_params.window_y;
    let inside_xz = world.x >= rect.x && world.x < rect.z && world.z >= rect.y && world.z < rect.w;
    let inside_y = world.y >= y.x && world.y < y.y;
    return inside_xz && inside_y;
}

/// True when the fragment is inside the ring of the next finer level.
fn inside_finer_ring(world: vec3<f32>) -> bool {
    let rect = water_params.finer_rect;
    return world.x >= rect.x && world.x < rect.z && world.z >= rect.y && world.z < rect.w;
}

fn animated_uv(uv: vec2<f32>) -> vec2<f32> {
    // Vanilla water_still.png is 16x512: 32 frames stacked vertically.
    let frames: f32 = 32.0;
    let fps: f32 = 8.0;
    let frame: f32 = floor(water_params.time * fps) % frames;

    var out = uv;
    // Near water carries one texture tile per block, so its `uv` already sits in [0, 1] and the
    // frame stack below stays inside one frame. Far-view LOD water spans many blocks per quad and
    // carries world-block UVs (lod_mesh.rs), so fold them back to one tile per block here — same
    // tiling, same phase as the near mesh.
    if (water_params.lod_mode > 0.5) {
        out = vec2<f32>(fract(out.x), fract(out.y));
    }
    out.y = (out.y + frame) / frames;
    out.x = fract(out.x + water_params.time * water_params.scroll_speed);
    return out;
}

fn animated_uv_side(uv: vec2<f32>, world_n: vec3<f32>) -> vec2<f32> {
    // Dedicated side animation (approximate MC `water_flow` using the still texture):
    // - pin V to [0..1] from waterline->bottom in the mesh
    // - scroll "downward" to read as falling/flowing
    // - add a tiny horizontal drift based on face direction so opposite sides don't match
    let frames: f32 = 32.0;
    let fps: f32 = 12.0;
    let frame: f32 = floor(water_params.time * fps) % frames;

    var out = uv;
    // Downward flow (faster than surface scroll).
    out.y = fract(out.y + water_params.time * 0.22);
    // Small horizontal drift; flip by face direction for variety.
    // (`u` is a world-block coordinate for LOD water and 0..1 for near water; the `fract` below
    // folds both to the same per-block tile.)
    let s = sign(world_n.x + world_n.z * 0.7);
    out.x = fract(out.x + water_params.time * 0.03 * s);
    // Sample within the animated frame stack.
    out.y = (out.y + frame) / frames;
    return out;
}

fn noises(uv: vec2<f32>) -> vec4<f32> {
    // We explicitly wrap in-shader; sampler is also repeat, but this makes behavior robust.
    return textureSample(water_noises_tex, water_noises_samp, fract(uv));
}

fn get_wave_normal_fast(world_pos: vec3<f32>) -> vec3<f32> {
    // Cheap normal approximation for far water:
    // 2 noise samples -> pseudo slope, keeps specular/refraction "alive" while cutting cost.
    let uv = world_pos.xz / 32.0 + vec2<f32>(water_params.time * 0.02, water_params.time * 0.015);
    let n0 = noises(uv).r * 2.0 - 1.0;
    let n1 = noises(uv + vec2<f32>(0.37, 0.21)).g * 2.0 - 1.0;
    return normalize(vec3<f32>(n0, 1.0, n1));
}

// Ported (simplified) from the provided shaderpack's `lib/waterBump.glsl`.
fn get_water_heightmap(pos_xz: vec2<f32>) -> f32 {
    var pos = pos_xz;
    var height_sum: f32 = 0.0;
    let movement: f32 = water_params.time * 0.035;

    // radiance = 2.39996
    let c: f32 = -0.7373688;
    let s: f32 = 0.6754903;
    let rot = mat2x2<f32>(vec2<f32>(c, -s), vec2<f32>(s, c));

    let wave0 = vec2<f32>(48.0, 12.0);
    let wave1 = vec2<f32>(12.0, 48.0);
    let wave2 = vec2<f32>(32.0, 32.0);

    let waves_large: f32 = max(noises(pos / 600.0).b, 0.1);

    pos = rot * pos;
    height_sum += noises(pos / wave0 + vec2<f32>(waves_large * 0.5 + movement)).b;
    pos = rot * pos;
    height_sum += noises(pos / wave1 + vec2<f32>(waves_large * 0.5 + movement)).b;
    pos = rot * pos;
    height_sum += noises(pos / wave2 + vec2<f32>(waves_large * 0.5 + movement)).b;

    return (height_sum / 60.0) * waves_large;
}

fn get_wave_normal(world_pos: vec3<f32>, cam_pos: vec3<f32>) -> vec3<f32> {
    // Vary detail with distance, matching the shaderpack idea: fewer fine ripples far away.
    let range = min(length(world_pos - cam_pos) / (16.0 * 14.0), 3.0);
    let delta = range + 0.15;

    let coord = world_pos.xz;
    let h0 = get_water_heightmap(coord);
    let hx = get_water_heightmap(coord + vec2<f32>(delta, 0.0));
    let hz = get_water_heightmap(coord + vec2<f32>(0.0, delta));

    // Boost slope a bit so waves show up as "sparkly" normals even without perfect SSR/IBL.
    let amp = 6.0;
    let dhdx = (hx - h0) / delta * amp;
    let dhdz = (hz - h0) / delta * amp;

    // Height is along +Y; build a world normal.
    //
    // Note: the reference shaderpack's `getWaveNormal()` encodes slope with the opposite sign
    // compared to the "textbook" heightmap normal. Matching that sign makes the apparent
    // refraction/offset direction feel correct relative to the shaderpack.
    return normalize(vec3<f32>(dhdx, 1.0, dhdz));
}

fn clip_world_to_screen_uv(world_pos: vec3<f32>) -> vec2<f32> {
    // Matches Bevy's `specular_transmissive_light()` mapping.
    let clip = view.clip_from_world * vec4(world_pos, 1.0);
    return (clip.xy / clip.w) * vec2(0.5, -0.5) + 0.5;
}

fn uv_in_bounds(uv: vec2<f32>, margin: f32) -> bool {
    return all(uv >= vec2(margin)) && all(uv <= vec2(1.0 - margin));
}

fn uv_in_bounds_asym(uv: vec2<f32>, margin_x: f32, margin_top: f32, margin_bottom: f32) -> bool {
    return uv.x >= margin_x
        && uv.x <= (1.0 - margin_x)
        && uv.y >= margin_top
        && uv.y <= (1.0 - margin_bottom);
}

fn safe_scale_1d(base: f32, full: f32, lo: f32, hi: f32) -> f32 {
    let d = full - base;
    if (abs(d) < 1e-6) {
        return 1.0;
    }
    if (full < lo) {
        return (lo - base) / d;
    }
    if (full > hi) {
        return (hi - base) / d;
    }
    return 1.0;
}

fn is_nan_f32(x: f32) -> bool {
    // WGSL doesn't guarantee an `isNan()` builtin on all supported backends (notably WebGL2 via Naga),
    // but NaN has the property that it is never equal to itself.
    return x != x;
}

fn safe_normalize2(v: vec2<f32>) -> vec2<f32> {
    let l = length(v);
    if (l < 1e-6) {
        return vec2(0.0, 0.0);
    }
    return v / l;
}

fn sample_view_transmission(uv: vec2<f32>) -> vec4<f32> {
    var c = textureSampleLevel(view_transmission_texture, view_transmission_sampler, uv, 0.0);
#ifdef TONEMAP_IN_SHADER
    c = approximate_inverse_tone_mapping(c, view.color_grading);
#endif
    // Transmission texture already has exposure applied; undo it (matching Bevy).
    c = vec4(c.rgb / view.exposure, c.a);
    return c;
}

fn apply_simple_attenuation(rgb: vec3<f32>, attenuation_color: vec3<f32>, attenuation_distance: f32, thickness: f32) -> vec3<f32> {
    // Approximate Bevy's built-in attenuation fog used for transmissive materials.
    // This is not physically perfect, but gives the key cue: thicker water darkens/saturates.
    let dist = max(attenuation_distance, 1e-3);
    let absorb = (1.0 - attenuation_color) / dist;
    return rgb * exp(-absorb * thickness);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    // Far-view LOD water (design §7.2): the same two hard cuts the LOD terrain material makes, so
    // the far water steps aside for the near window and for the next finer level exactly like the
    // opaque tiles do — otherwise the far water would fight the near water at the window edge.
    // Both are off for the near material (`lod_mode == 0`).
    if (water_params.lod_mode > 0.5) {
        if (water_params.window_y.w > 0.5 && inside_near_window(in.world_position.xyz)) {
            discard;
        }
        if (water_params.window_y.z > 0.5 && inside_finer_ring(in.world_position.xyz)) {
            discard;
        }
    }

    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Camera-space factors for "sunset water":
    // - Near water: more transparent (see the blocks below).
    // - Far water (grazing angles): brighter / more reflective.
    let cam_pos = view.world_position;
    let dist = length(cam_pos - pbr_input.world_position.xyz);
    // Bring the effect closer: typical gameplay view distances are not huge in V1.
    let far_t = smoothstep(6.0, 90.0, dist);
    // 1.0 near -> 0.0 far. Used to fade out expensive noise work.
    let near_detail = 1.0 - smoothstep(0.55, 1.0, far_t);
    // Far-view LOD water only: how much of the near water's per-pixel detail (wave normals, sun
    // glitter, the texture sample) still applies at this distance.
    //
    // Far tiles reach out to about 2 km, where that detail is pure noise: the water texture has no
    // mipmaps, so neighbouring pixels sample texels from different animation frames, and a wave
    // normal under a `pow(..., 220.0)` sun glint turns the surface into a shimmering mesh. The far
    // water fades to a flat, averagely-coloured surface instead. The near material never takes this
    // branch (`lod_mode == 0`), so nothing about the near water changes.
    let lod_detail = select(1.0, 1.0 - smoothstep(48.0, 160.0, dist), water_params.lod_mode > 0.5);

    // Only wave the (mostly) top surface (avoid wobbling the vertical sides).
    // With flowing water geometry, the surface can be slightly sloped, so use a lower threshold.
    let wn = normalize(pbr_input.world_normal);
    // Vertical faces ("water walls") should have weaker refraction / reflection, otherwise they read
    // as overly distorted, mirror-like strips.
    let top_w = smoothstep(0.15, 0.55, abs(wn.y));
    let side_w = 1.0 - top_w;

    // Boat interior mask: discard water *surface* fragments inside the boat's hollow footprint.
    // We only enable this while the local player is mounted to avoid obvious "holes" in the world.
    if (water_params.boat_mask_on > 0.5 && top_w > 0.75) {
        let dx = pbr_input.world_position.x - water_params.boat_center_x;
        let dz = pbr_input.world_position.z - water_params.boat_center_z;
        let c = water_params.boat_cos;
        let s = water_params.boat_sin;
        // Rotate world XZ into boat-local coordinates.
        let lx = c * dx + s * dz;
        let lz = -s * dx + c * dz;
        if (abs(lx) < water_params.boat_half_x && abs(lz) < water_params.boat_half_z) {
            discard;
        }
    }
    var wave_n = pbr_input.N;
    if abs(wn.y) > 0.35 {
        if near_detail > 0.25 {
            wave_n = get_wave_normal(pbr_input.world_position.xyz, cam_pos);
        } else {
            wave_n = get_wave_normal_fast(pbr_input.world_position.xyz);
        }
        // When viewing the top surface from below, `pbr_input.N` is flipped (double-sided shading).
        // Keep the wave normal consistent with that orientation, otherwise we end up mixing +Y waves
        // into a -Y base normal which collapses lighting/refraction.
        if !is_front {
            wave_n = -wave_n;
        }
        // Blend toward the wave normal instead of snapping; keeps lighting stable near edges.
        // `lod_detail` takes the whole wobble to zero over the far view's outer range.
        let wave_w = water_params.wave_strength * mix(0.35, 1.0, near_detail) * lod_detail;
        pbr_input.N = normalize(mix(pbr_input.N, wave_n, wave_w));
    }

// Replace base-color sampling with animated UVs (if the base has a texture).
#ifdef VERTEX_UVS
    if ((pbr_bindings::material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        var bias: SampleBias;
        bias.mip_bias = view.mip_bias;
        // Side faces use a dedicated UV animation; top faces use the surface animation.
        let uv_anim = select(animated_uv_side(in.uv, wn), animated_uv(in.uv), top_w > 0.5);
        let tex = pbr_functions::sample_texture(
            pbr_bindings::base_color_texture,
            pbr_bindings::base_color_sampler,
            uv_anim,
            bias,
        );
        // The vanilla water texture's alpha isn't directly usable for our blend; keep alpha driven by
        // the material + our distance/fresnel logic, and use the texture for color variation only.
        let base = pbr_bindings::material.base_color;
        // Slightly darken sides so they don't read as "milky white panels" under direct light,
        // but keep them close to the top surface color.
        let side_mul = mix(0.92, 1.0, top_w);
        // With no mipmaps, a far tile's per-pixel texel is noise; fade the sample into the texture's
        // own average (`water_still.png` averages 0.70 grey over its animation frames) so the far
        // surface keeps the material's colour without the speckle.
        let tex_rgb = mix(tex.rgb, vec3(LOD_WATER_TEXTURE_AVERAGE), 1.0 - lod_detail);
        pbr_input.material.base_color = vec4(base.rgb * tex_rgb * side_mul, base.a);
    }
#endif

    // Fresnel:
    // - Use a "sharper" curve for transmission (so near water can still refract/see-through).
    // - Use a "wider" curve for reflection at distance, so far water gets noticeably more reflective.
    let ndotv = max(dot(pbr_input.N, pbr_input.V), 0.0);
    let one_minus = clamp(1.0 - ndotv, 0.0, 1.0);
    let fresnel_trans = pow(one_minus, 5.0);
    let refl_exp = mix(5.0, 2.2, far_t);
    let fresnel_reflect = pow(one_minus, refl_exp);

    // Opacity:
    // - Near: fairly transparent so you can see into shallow water.
    // - Far: less transparent (water reads as a "surface" and not invisible glass).
    // - Grazing angles: less transparent (stronger reflection).
    //
    // Water volume absorption (Beer-Lambert-ish).
    // With real screen-space refraction enabled (specular transmission), keep this subtle and
    // let Bevy's built-in attenuation handle most of the "deep water" cue.
    let has_refraction = pbr_input.material.specular_transmission > 0.0 && pbr_input.material.thickness > 0.0;
    // Beer-Lambert absorption coefficients (stylized):
    // - Absorb red much more strongly than green/blue, to push deep water toward teal/cyan.
    // - Keep absorption strong even when refraction is enabled, otherwise the refracted scene
    //   stays too bright and water reads like glass.
    let absorb_rgb = vec3(0.55, 0.06, 0.03) * select(1.0, 1.0, has_refraction);
    // Get a per-vertex "depth hint" (computed on CPU by scanning water blocks downward).
    // 0 => shallow, 1 => deep. This is the main cue that makes looking straight down darken.
    var depth_hint: f32 = 0.0;
#ifdef VERTEX_COLORS
    depth_hint = clamp(in.color.r, 0.0, 1.0);
#endif

    // "Optical path length" approximation (tuned aggressive so the effect is obvious):
    // - depends on (estimated) water depth when looking down
    // - increases a lot when looking down into the surface (ndotv ~= 1)
    // - increases at grazing angles (longer path through water)
    // - increases with distance (far water looks murkier / less clear)
    // For side faces we have no good voxel depth hint (CPU packs it mainly for top faces).
    // Fake a modest depth so side walls get the same "crystal clear" body tint instead of white.
    let depth_hint2 = clamp(max(depth_hint, side_w * 0.42), 0.0, 1.0);
    let depth_blocks = mix(3.5, 75.0, depth_hint2);
    let looking_down = smoothstep(0.75, 0.98, ndotv);
    let angle_term = 1.0 / max(ndotv, 0.06);
    // Path units are arbitrary here; we just need stable, plausible scaling.
    let path = depth_blocks * (0.18 + 0.65 * looking_down) * angle_term + dist * 0.15;
    let trans_rgb = vec3(
        exp(-absorb_rgb.x * path),
        exp(-absorb_rgb.y * path),
        exp(-absorb_rgb.z * path),
    );
    let trans_luma = dot(trans_rgb, vec3(0.3333));
    let depth_factor = clamp(1.0 - trans_luma, 0.0, 1.0);
    let depth_mix = clamp(depth_factor * 0.85 + depth_hint2 * 0.65, 0.0, 1.0);

    // Debug: visualize depth signals driving absorption.
    // RGB = (depth_hint, depth_factor, far_t)
    if (water_params.debug > 1.5) {
        var dbg: FragmentOutput;
        dbg.color = vec4(depth_hint2, depth_factor, far_t, 1.0);
        return dbg;
    }

    // Alpha correction: far/deep water should be less see-through.
    // Alpha isn't used for the transmissive refraction pass (we render as Opaque).
    // Keep at 1.0 to avoid accidentally darkening/attenuating via alpha.
    let alpha = 1.0;

    // Tint / darken the transmitted component so the bottom doesn't stay "too clear" at distance.
    // (This is the missing cue that makes far water look deep instead of like glass.)
    // Deep water body color (teal-ish).
    let deep_col = vec3(0.008, 0.036, 0.055);
    let bc_t = pbr_input.material.base_color;
    // Keep tinting a bit gentler when refraction is active, but not so weak that deep water glows.
    let tint_mul = select(1.0, 0.55, has_refraction);
    let tint_strength = clamp((depth_factor * 0.75 + depth_hint2 * 0.45 + far_t * 0.25) * tint_mul, 0.0, 1.0);
    let tinted_rgb = mix(bc_t.rgb, deep_col, tint_strength);
    pbr_input.material.base_color = vec4(tinted_rgb, bc_t.a);

    // Darken the water *body* (diffuse / base color) with depth + distance, but keep specular reflections.
    let dark_t = clamp(depth_factor * 0.90 + far_t * 0.65 + depth_hint2 * 0.35, 0.0, 1.0);
    let body_mul = mix(1.0, 0.26, dark_t);
    let bc_body = pbr_input.material.base_color;
    pbr_input.material.base_color = vec4(bc_body.rgb * body_mul, bc_body.a);

    // Naga (used by Bevy) is conservative about assigning to swizzles / struct fields.
    // Avoid `pbr_input.material.base_color.a *= ...`-style updates by writing the whole vec4.
    let bc_alpha = pbr_input.material.base_color;
    pbr_input.material.base_color = vec4(bc_alpha.rgb, alpha);

    // Make the water smoother in the distance for sharper "sparkles".
    let rough_near = 0.10;
    let rough_far = 0.02;
    pbr_input.material.perceptual_roughness = min(pbr_input.material.perceptual_roughness, mix(rough_near, rough_far, far_t));

    // Boost reflectance with angle and distance (sunset glints).
    let refl = mix(0.10, 0.98, max(fresnel_reflect, far_t));
    pbr_input.material.reflectance =
        max(pbr_input.material.reflectance, refl * mix(0.35, 1.0, top_w));

    // Avoid "far water glow": make the surface read via reflection, not by boosting base color.

    // "Shaderpack-style" reflections without SSR/IBL:
    // - Use the scene directional light (if present) for a sun glint.
    // - Add a procedural sky gradient reflection via emissive so shaded far water still reads.
    var sun_dir = normalize(vec3(0.35, 0.10, -0.93));
    if (lights.n_directional_lights > 0u) {
        sun_dir = normalize(lights.directional_lights[0].direction_to_light);
    }
    let R = reflect(-pbr_input.V, pbr_input.N);
    let spec = pow(max(dot(R, sun_dir), 0.0), 220.0);
    // `lod_detail` fades the glint out over the far view: at 1–2 km a 220-exponent highlight on a
    // per-pixel wave normal is a field of flickering single pixels, not a sun glint.
    let glitter = spec * mix(0.0, 1.0, max(fresnel_reflect, far_t)) * fresnel_reflect * lod_detail;
    let em = pbr_input.material.emissive;
    pbr_input.material.emissive =
        vec4(em.rgb + vec3(1.15, 0.70, 0.30) * glitter * mix(0.30, 1.0, top_w), em.a);

    // Procedural "sky" reflection term.
    // This is what makes far/shaded water look less like invisible glass and more like a surface.
    let sky_t = pow(clamp(R.y * 0.5 + 0.5, 0.0, 1.0), 0.65);
    let sky_horizon = vec3(1.10, 0.58, 0.30);
    let sky_zenith = vec3(0.22, 0.40, 0.95);
    let sky_col = mix(sky_horizon, sky_zenith, sky_t);
    let env_strength = mix(0.08, 1.10, far_t);
    let em2 = pbr_input.material.emissive;
    pbr_input.material.emissive = vec4(
        em2.rgb
            + sky_col * env_strength * pow(fresnel_reflect, 0.65) * mix(0.35, 1.0, top_w),
        em2.a,
    );

    // Extra "sparkly" highlights distributed across the surface (not only a single sun glint).
    // Driven by the shaderpack noise texture.
    if near_detail > 0.15 {
        let sp_uv = pbr_input.world_position.xz / 4.0 + vec2(water_params.time * 0.05, water_params.time * 0.03);
        let sp_n = noises(sp_uv).r;
        var sp = pow(clamp((sp_n - 0.62) / 0.38, 0.0, 1.0), 6.0);
        sp *= mix(0.10, 1.0, max(far_t, fresnel_reflect));
        let sp_col = vec3(1.10, 0.78, 0.45);
        let em3 = pbr_input.material.emissive;
        pbr_input.material.emissive =
            vec4(em3.rgb + sp_col * sp * fresnel_reflect * mix(0.35, 1.0, top_w), em3.a);
    }

    // Subtle diffuse modulation so waves remain perceptible even when specular is weak.
    if near_detail > 0.25 && top_w > 0.5 {
        let h = get_water_heightmap(pbr_input.world_position.xz);
        let shade = mix(0.96, 1.05, clamp(h * 18.0, 0.0, 1.0));
        let bc_mod = pbr_input.material.base_color;
        pbr_input.material.base_color = vec4(bc_mod.rgb * shade, bc_mod.a);
    }

    // alpha discard (no-op for blend, but keeps mask/atc behavior consistent if changed later).
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    var out: FragmentOutput;
    // We keep the material flagged as transmissive so Bevy provides ViewTransmissionTexture,
    // but we do the refraction sampling ourselves to avoid WebGL2 edge stretching artifacts.
    var lit_input = pbr_input;
    lit_input.material.specular_transmission = 0.0;
    lit_input.material.thickness = 0.0;
    if ((pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        out.color = apply_pbr_lighting(lit_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    // Side walls should read mostly via transmission + tint, not heavy diffuse shading,
    // but keep close to the top surface so the color matches.
    out.color = vec4(out.color.rgb * mix(0.85, 1.0, top_w), out.color.a);

    // Custom refraction compositing (screen-space):
    // - sample the view transmission texture at a refracted UV
    // - clamp UV and fade refraction if it would go out-of-bounds (avoids clamp-to-edge stretching)
    //
    // Far-view LOD water never refracts: it is drawn beyond the near window, so the screen-space
    // sample would smear whatever the sky or another far tile wrote behind it. Its material ships
    // `specular_transmission: 0.0`, which alone keeps it out of the branch below; `lod_mode` is
    // the belt-and-braces guard, so a stray non-zero transmission cannot revive the sample.
    let tr_strength = pbr_input.material.specular_transmission;
    let thickness = pbr_input.material.thickness;
    if (water_params.lod_mode < 0.5 && tr_strength > 0.0 && thickness > 0.0) {
        // NOTE: Instead of projecting `world_pos + T * thickness` (Bevy's default refraction),
        // we use a shaderpack-style *screen-space* refraction offset derived from the (wavy)
        // surface normal. This avoids the "UV hits bottom border => stretched strip" artifact
        // that peaks at intermediate view angles (often ~45 degrees).
        let wp = pbr_input.world_position.xyz;
        let uv_base_raw = clip_world_to_screen_uv(wp);

        // View-space normal: small XY tilt -> small screen-space offset.
        let n_view = normalize((view.view_from_world * vec4(pbr_input.N, 0.0)).xyz);
        let view_pos = (view.view_from_world * vec4(wp, 1.0)).xyz;
        let view_z = max(-view_pos.z, 0.05);

        let ior_scale = clamp((pbr_input.material.ior - 1.0) / 0.333, 0.0, 1.0);
        // Base refraction strength tuned for "Minecraft shaderpack" feel on WebGL2.
        // Keep it weaker at distance and at grazing angles (where reflection dominates anyway).
        let ref_str = (0.085 * thickness)
            * ior_scale
            * (1.0 - fresnel_trans)
            * mix(1.2, 0.25, far_t)
            // Keep sides close to the surface strength (user expects similar wobble).
            * mix(0.90, 1.0, top_w);

        var duv = n_view.xy * (ref_str / view_z);
        // Side faces should still have some refraction wobble, but normal-based offset can read as
        // a rigid "strip". Add a tiny low-frequency noise wobble on sides to keep it alive.
        if (side_w > 0.001) {
            // Build a per-face 2D basis so the wobble direction makes sense on vertical walls.
            // (Otherwise adding a fixed screen-space jitter can look "wrong" or be imperceptible.)
            let up_w = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(wn.y) > 0.92);
            let t_w = normalize(cross(up_w, wn));       // horizontal along the face
            let b_w = normalize(cross(wn, t_w));        // vertical along the face
            let t_v = (view.view_from_world * vec4(t_w, 0.0)).xyz;
            let b_v = (view.view_from_world * vec4(b_w, 0.0)).xyz;
            let t_xy = safe_normalize2(t_v.xy);
            let b_xy = safe_normalize2(b_v.xy);

            // Surface-local coordinates for low-frequency wobble, scrolling downward along `b_w`.
            let surf = vec2<f32>(dot(wp, t_w), dot(wp, b_w));
            let s_uv0 = surf / 18.0 + vec2<f32>(water_params.time * 0.010, water_params.time * 0.060);
            let j0 = noises(s_uv0).r * 2.0 - 1.0;
            let j1 = noises(s_uv0 + vec2<f32>(0.17, 0.53)).g * 2.0 - 1.0;

            // Scale with view depth so it stays perceptible (but stable) on walls.
            let inv_vz = 1.0 / max(view_z, 0.35);
            let side_j = (0.030 * thickness) * side_w * inv_vz * (1.0 - fresnel_trans) * mix(1.0, 0.35, far_t);
            duv += (t_xy * j0 + b_xy * j1) * side_j;
        }
        // Clamp maximum offset to avoid pushing large regions to the screen borders.
        let duv_len = length(duv);
        let duv_max = 0.045;
        if (duv_len > duv_max) {
            duv = duv * (duv_max / max(duv_len, 1e-6));
        }

        let uv_exit = uv_base_raw + duv;

        if (!(is_nan_f32(uv_exit.x) || is_nan_f32(uv_exit.y))) {
                let vp = view.viewport;
                let dims_i = textureDimensions(view_transmission_texture);
                let dims = vec2<f32>(f32(dims_i.x), f32(dims_i.y));
                let safe_px: f32 = 2.0;
                let uv_min = vec2(safe_px, safe_px) / max(dims, vec2(1.0));

                // WebGL2 / transmission: we've observed that the bottom of ViewTransmissionTexture
                // can contain invalid / stretched pixels (likely due to viewport-vs-texture padding).
                //
                // Fix strategy (as requested): do not "fill" or repeat content. Instead, compute how
                // much of the transmission texture is actually valid (based on view.viewport) and
                // stretch that valid region to cover the full screen range.
                //
                // If the texture is taller than the viewport, `valid_max_y < 1.0` and we sample only
                // the top valid portion: sample_y = uv_y * valid_max_y.
                let valid_max_y = clamp((vp.w - safe_px) / max(dims.y, 1.0), 0.0, 1.0);
                let uv_max = vec2(1.0 - uv_min.x, max(valid_max_y - uv_min.y, uv_min.y));

                // Baseline UV at this fragment.
                let uv_base_s = vec2(uv_base_raw.x, uv_base_raw.y * valid_max_y);
                let uv_b = clamp(uv_base_s, uv_min, uv_max);

                // Refracted UV (scaled toward baseline as needed to stay in-bounds; avoids clamp-to-edge stretching).
                let uv_exit_s = vec2(uv_exit.x, uv_exit.y * valid_max_y);

                // Refraction is fundamentally screen-space: if the refracted UV would leave the screen,
                // the result is undefined (clamp-to-edge => stretched strips). Fade the refraction offset
                // out as we approach the screen borders. This also makes the result feel more "natural"
                // near the bottom of the view where view rays get steep.
                let edge_screen = min(
                    min(uv_base_raw.x, 1.0 - uv_base_raw.x),
                    min(uv_base_raw.y, 1.0 - uv_base_raw.y),
                );
                let edge_factor = smoothstep(0.0, 0.10, edge_screen);
                let uv_exit_s2 = uv_b + (uv_exit_s - uv_b) * edge_factor;
                // Keep a small "guard" away from the borders; if the refracted UV hits a border,
                // you'll sample a constant row/column and get a stretched strip.
                let guard_px: f32 = 10.0;
                let guard_uv = vec2(guard_px) / max(dims, vec2(1.0));
                var uv_min_s = uv_min;
                var uv_max_s = uv_max;
                if (all(uv_min + guard_uv < uv_max - guard_uv)) {
                    uv_min_s = uv_min + guard_uv;
                    uv_max_s = uv_max - guard_uv;
                }

                let sx = safe_scale_1d(uv_b.x, uv_exit_s2.x, uv_min_s.x, uv_max_s.x);
                let sy = safe_scale_1d(uv_b.y, uv_exit_s2.y, uv_min_s.y, uv_max_s.y);
                let s = clamp(min(sx, sy), 0.0, 1.0);
                let uv_refr = clamp(uv_b + (uv_exit_s2 - uv_b) * s, uv_min_s, uv_max_s);

                // If a large portion of the surface ends up clamped to a border (especially bottom),
                // the sampled background row becomes constant and appears as a "stretched strip".
                // Fix: fade refraction back to baseline as we approach any border.
                let px = vec2(1.0 / max(dims.x, 1.0), 1.0 / max(dims.y, 1.0));
                let edge = min(uv_refr - uv_min_s, uv_max_s - uv_refr);
                let edge_fade2 = min(
                    smoothstep(0.0, px.x * 10.0, edge.x),
                    smoothstep(0.0, px.y * 10.0, edge.y),
                );
                let uv_sample = mix(uv_b, uv_refr, edge_fade2);

                var bg = sample_view_transmission(uv_sample);
                let bg_b = sample_view_transmission(uv_b);
                let l0 = dot(bg.rgb, vec3(0.3333));
                let l1 = dot(bg_b.rgb, vec3(0.3333));
                // If the refracted sample is near-black (invalid band), fall back to baseline UV.
                bg = select(bg, bg_b, l0 < 0.002);

                let bg_luma = dot(bg.rgb, vec3(0.3333));
                let bg_valid = smoothstep(0.002, 0.02, max(bg_luma, l1));

                // Apply absorption/tint cues to the refracted scene:
                // - deep-water absorption driven by voxel depth hint (trans_rgb)
                // - thickness-based attenuation (material attenuation params)
                var refr_rgb = bg.rgb * trans_rgb;
                if ((pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ATTENUATION_ENABLED_BIT) != 0u) {
                    refr_rgb = apply_simple_attenuation(
                        refr_rgb,
                        pbr_input.material.attenuation_color.rgb,
                        pbr_input.material.attenuation_distance,
                        thickness,
                    );
                }
                // Extra darkening for deep/far refracted scene so it doesn't look emissive.
                // We bias toward depth_hint because that's the best proxy for "how much water volume"
                // you're looking through when staring down into a lake.
                let looking_down2 = smoothstep(0.60, 0.96, ndotv);
                refr_rgb *= mix(1.0, 0.24, depth_mix * looking_down2) * mix(1.0, 0.82, far_t);
                // Suppress overly bright highlights in the refracted scene (stylized "scattering"),
                // to avoid harsh white patches under water.
                refr_rgb = refr_rgb / (vec3(1.0) + refr_rgb * 0.45);
                // More transmission when looking down, less at grazing angles (Fresnel).
                // Keep some surface reflection visible by capping transmission.
                // Fade transmission near UV bounds (where we had to shrink the refraction offset).
                let edge_fade = smoothstep(0.0, 0.85, s) * edge_fade2;
                // Deep/far water should show less of the "bottom" through refraction.
                let depth_w = mix(1.0, 0.14, depth_mix);
                let far_w = mix(1.0, 0.45, far_t);
                // Slight boost for shallow/near water so it's a bit clearer without making deep water glow.
                let shallow = 1.0 - depth_mix;
                let near = 1.0 - far_t;
                let trans_boost = 1.0 + 0.35 * shallow * near;
                let trans_max = mix(0.62, 0.45, top_w);
                let trans_w = clamp(
                    tr_strength
                        * 0.72
                        * (1.0 - fresnel_trans)
                        * bg_valid
                        * edge_fade
                        * depth_w
                        * far_w
                        * trans_boost
                        // Give sides more "glassiness": offset is already tiny on sides, so higher
                        // transmission reads as crystal-clear water rather than wobbly strips.
                        * mix(0.65, 1.0, top_w),
                    0.0,
                    trans_max,
                );
                out.color = vec4(mix(out.color.rgb, refr_rgb, trans_w), out.color.a);

                // Debug: visualize the inferred "valid transmission Y max" on screen.
                // - magenta line: y == valid_max_y
                // - red tint: y > valid_max_y (the region that would have sampled invalid rows without rescaling)
                if (water_params.debug > 0.5) {
                    let screen_uv = uv_base_raw;
                    let line_w = 2.0 / max(vp.w, 1.0);
                    if (abs(screen_uv.y - valid_max_y) < line_w) {
                        out.color = vec4(1.0, 0.0, 1.0, 1.0);
                    } else if (screen_uv.y > valid_max_y) {
                        out.color = vec4(mix(out.color.rgb, vec3(1.0, 0.0, 0.0), 0.35), out.color.a);
                    } else {
                        // Additional diagnostics:
                        // - green: refraction offset got scaled down (s < 1) to stay in bounds
                        // - yellow: refracted UV pinned to the bottom bound (common cause of "vertical stretch")
                        // - cyan: refracted UV pinned to the top bound
                        let px_y = 2.0 / max(dims.y, 1.0);
                        if (s < 0.995) {
                            out.color = vec4(mix(out.color.rgb, vec3(0.0, 1.0, 0.0), 0.45), out.color.a);
                        }
                        if (abs(uv_refr.y - uv_max_s.y) < px_y) {
                            out.color = vec4(1.0, 1.0, 0.0, 1.0);
                        } else if (abs(uv_refr.y - uv_min_s.y) < px_y) {
                            out.color = vec4(0.0, 1.0, 1.0, 1.0);
                        }
                    }
                }
            }
        }
	    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
	    return out;
	}
