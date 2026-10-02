#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

// Bindings match `UnderwaterPostPipeline` in `underwater_post.rs`.
@group(0) @binding(0) var screen_tex: texture_2d<f32>;
@group(0) @binding(1) var screen_samp: sampler;
@group(0) @binding(2) var noise_tex: texture_2d<f32>;

struct UnderwaterSettings {
    amount: f32,
    density: f32,
    distort: f32,
    time: f32,
    tint: vec3<f32>,
    submersion: f32,
    shallow: f32,
    _pad2: vec3<f32>,
}
@group(0) @binding(3) var<uniform> uw: UnderwaterSettings;

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn saturate2(v: vec2<f32>) -> vec2<f32> {
    return clamp(v, vec2<f32>(0.0), vec2<f32>(1.0));
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = saturate2(in.uv);
    let base = textureSample(screen_tex, screen_samp, uv).rgb;

    // Aspect-correct coordinates so distortion frequency/amplitude match in pixel space.
    // Without this, the effect can look "stretched" horizontally on wide viewports.
    let dims_i = vec2<i32>(textureDimensions(screen_tex));
    let dims = vec2<f32>(max(vec2<i32>(1), dims_i));
    let aspect = dims.x / dims.y; // width / height
    let uvp = vec2<f32>(uv.x * aspect, uv.y);

    // No-op fast path (still draws, but avoids extra samples/math).
    if (uw.amount <= 0.0001) {
        return vec4<f32>(base, 1.0);
    }

    // Approximate "looking toward the surface" (top of screen) vs "looking deeper".
    // This helps distinguish shore/sky vs underwater objects without needing a depth texture.
    // surface = 1 near top, 0 near bottom
    let surface = 1.0 - smoothstep(0.18, 0.62, uv.y);

    // Distortion (screen-space "refraction" wobble).
    // Goal: large-amplitude, low-frequency wobble (avoid "comb/ECG" high-frequency jitter).
    let t = uw.time;
    // Low-frequency noise layers.
    let n_uv0 = uvp * 0.65 + vec2<f32>(t * 0.010, t * -0.008);
    let n0 = textureSample(noise_tex, screen_samp, n_uv0).rg * 2.0 - 1.0;
    let n_uv1 = uvp * 0.22 + vec2<f32>(t * -0.006, t * 0.005);
    let n1 = textureSample(noise_tex, screen_samp, n_uv1).ba * 2.0 - 1.0;

    // Slow large wave drift to avoid a "static noise slide".
    // Cross-axis drift: dx depends on uv.y, dy depends on uv.x (matches "container water" wobble feel).
    let wx = (sin(uvp.y * 1.20 + t * 0.35) + sin(uvp.y * 0.65 - t * 0.22)) * 0.5;
    let wy = (cos(uvp.x * 1.20 + t * 0.30) + cos(uvp.x * 0.65 - t * 0.18)) * 0.5;

    // Stronger wobble when more submerged. Reduce wobble toward the surface to avoid "half-screen"
    // ghosting where shoreline/sky gets dragged around too much.
    // Fade distortion in shallow water to prevent near-shore ghosting / double edges.
    let shallow_fade = mix(0.10, 1.0, uw.shallow);
    let surface_fade = mix(1.0, 0.45, surface);
    let wobble = (0.020 + 0.020 * uw.submersion) * surface_fade * shallow_fade * uw.distort * uw.amount;
    // Apply distortion in both axes with different mixes so it doesn't collapse to a single-direction jitter.
    let dx = (n0.x * 0.60 + n1.x * 0.25 + wx * 0.25);
    let dy = (n0.y * 0.55 + n1.y * 0.30 + wy * 0.25);
    // Convert back to unscaled UV space: x offset must be divided by aspect.
    let duv = vec2<f32>(dx / aspect, dy) * wobble;
    let uv2 = saturate2(uv + duv);
    let refracted = textureSample(screen_tex, screen_samp, uv2).rgb;

    // WebGL2 path: no depth texture reads (wgpu/naga limitation when translating to GLSL).
    // Instead, approximate "path length in water" from the camera's submersion depth + a mild vignette.
    let vignette = saturate(length(vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5)) * 1.35);
    // Toward the surface, reduce the effective water column to avoid a "blue overlay" look.
    let water_column = mix(1.0, 0.40, surface);
    let dist = (6.0 + 26.0 * uw.submersion) * water_column * (1.0 + vignette * 0.25);
    let fog = 1.0 - exp(-uw.density * dist);

    // Beer-Lambert-ish absorption: red attenuates faster than green/blue.
    // Keep absorption weaker for clarity; depth is represented by `dist` above.
    let absorb = exp(-dist * vec3<f32>(0.060, 0.030, 0.016));
    let absorbed = refracted * absorb;

    // Blend toward a blue-green water tint with fog.
    // Reduce tint a bit when looking toward the surface so shore/sky keep their identity.
    let tint = mix(uw.tint, vec3<f32>(1.0), surface * 0.18);
    let fogged = mix(absorbed, tint, fog * 0.70);

    // Final mix controlled by `amount` (smooth transition when entering/leaving water).
    // IMPORTANT: avoid a "double image" where the un-refracted scene is still visible under the
    // refracted scene. Once underwater (`amount` ~ 1), we want to fully switch to the refracted view;
    // surface/shore differentiation should come from reduced fog/tint/distortion, not blending `base`.
    let out_col = mix(base, fogged, uw.amount);
    return vec4<f32>(out_col, 1.0);
}
