#pragma once
// GLSL 4.30 sources. Programs are assembled from chunks (see Simulation.hpp) so that the uniform
// blocks are declared once and stay in sync with the structs in Model.hpp.
#include <string_view>

namespace optics::shaders {
inline constexpr std::string_view version = "#version 430 core\n";
inline constexpr std::string_view computeLayout = "layout(local_size_x = 128) in;\n";

// Simulation and display parameters; mirrors optics::Params (std140, binding 0).
inline constexpr std::string_view paramsBlock = R"GLSL(
layout(std140, binding = 0) uniform Params {
    int n, scene, sourceMode, absorbing;
    int brushActive, display, sliceAxis;
    float dt;
    float time, wavelength, opening, separation;
    float refractiveIndex, exposure, slicePosition;
    vec3 brush;
};
)GLSL";

// Camera frame; mirrors optics::ViewParams (std140, binding 1).
inline constexpr std::string_view viewBlock = R"GLSL(
layout(std140, binding = 1) uniform View {
    vec4 eye, cameraForward, cameraRight, cameraUp;
    float aspect, tanHalfFov;
};
)GLSL";

// Storage and geometry helpers shared by the kernels and the renderer.
inline constexpr std::string_view common = R"GLSL(
layout(std430, binding = 0) buffer Current { float field[]; };
layout(std430, binding = 1) buffer Previous { float oldField[]; };
layout(std430, binding = 2) buffer Next { float nextField[]; };
layout(std430, binding = 3) buffer Average { float intensity[]; };   // EMA of u²
layout(std430, binding = 4) buffer Medium { vec2 medium[]; };        // x: n² (0 = opaque), y: damping rate
struct Metric { vec4 totals; vec4 bounds; vec4 spatial; };          // see optics::Metrics
layout(std430, binding = 5) buffer Metrics { Metric metrics[]; };    // [0] total, [1..] per workgroup
layout(std430, binding = 6) buffer Screen { float detector[]; };     // [y + n z] at the detector plane

const float pi = 3.14159265358979323846, courant = .45;
const vec3 lensCenter = vec3(.49, .5, .5);

int indexOf(ivec3 p) { return p.x + n * (p.y + n * p.z); }
ivec3 coord(int id) { return ivec3(id % n, (id / n) % n, id / (n * n)); }
bool boundary(ivec3 p) { return any(equal(p, ivec3(0))) || any(equal(p, ivec3(n - 1))); }
int sourceX() { return int(round(.16 * float(n - 1))); }
int barrierX() { return int(round(.36 * float(n - 1))); }
int screenX() { return int(round(.82 * float(n - 1))); }   // keep in sync with optics::detectorPlane
// Transparent part of the screen: two rectangular slits (scene 0) or one circular aperture (scene 1).
bool hole(vec3 p) {
    if (scene == 0) return abs(abs(p.y - .5) - separation * .5) < opening * .5 && abs(p.z - .5) < .22;
    return length(p.yz - vec2(.5)) < opening;
}
)GLSL";

// Medium (screen, lens, sponge) and a zero field.
inline constexpr std::string_view initialize = R"GLSL(
void main() {
    int id = int(gl_GlobalInvocationID.x);
    if (id >= n * n * n) return;
    ivec3 p = coord(id);
    vec3 pos = vec3(p) / float(n - 1);
    float mass = 1., sigma = 0.;
    if (scene < 2 && p.x == barrierX() && !hole(pos)) mass = 0.;                         // opaque screen
    if (scene == 2 && length(pos - lensCenter) < opening) mass = refractiveIndex * refractiveIndex;
    if (absorbing != 0) {   // sponge: sigma rises as a cubic toward the outer faces
        vec3 edge = min(pos, 1. - pos);
        float distance = min(min(edge.x, edge.y), edge.z);
        sigma = 80. * pow(max(0., 1. - distance / .13), 3.);
    }
    medium[id] = vec2(mass, sigma);
    field[id] = 0.;
    oldField[id] = 0.;
    nextField[id] = 0.;
    intensity[id] = 0.;
    if (id < n * n) detector[id] = 0.;
}
)GLSL";

// One leapfrog step of m (u_tt + 2 sigma u_t) = laplacian(u) + forcing, plus the intensity EMA.
inline constexpr std::string_view step = R"GLSL(
void main() {
    int id = int(gl_GlobalInvocationID.x);
    if (id >= n * n * n) return;
    ivec3 p = coord(id);
    vec2 m = medium[id];
    float u = 0.;
    if (!boundary(p) && m.x > 0.) {
        float lap = field[id - 1] + field[id + 1] + field[id - n] + field[id + n] + field[id - n * n] +
                    field[id + n * n] - 6. * field[id];
        float omegaDt = 2. * pi * dt / wavelength;
        float drive = 0.;
        if (sourceMode != 2 && p.x == sourceX()) {   // apodized incident plane: ramped beam or Gaussian pulse
            vec2 r = (vec2(p.y, p.z) / float(n - 1) - .5) / .32;
            float envelope = exp(-pow(length(r), 8.));
            float temporal = sourceMode == 0 ? 1. - exp(-pow(time / (2. * wavelength), 2.))
                                             : exp(-pow((time - 3. * wavelength) / wavelength, 2.));
            drive = 2. * courant * sin(omegaDt) * sin(2. * pi * time / wavelength) * envelope * temporal;
        }
        if (brushActive != 0) {   // mouse source: Gaussian of 1.5 cells sharing the carrier phase
            vec3 r = (vec3(p) / float(n - 1) - brush) * float(n - 1) / 1.5;
            drive += .12 * exp(-dot(r, r)) * sin(2. * pi * time / wavelength);
        }
        float damping = m.y * dt;
        u = (2. * field[id] - (1. - damping) * oldField[id] + courant * courant * lap / m.x + drive) / (1. + damping);
    }
    nextField[id] = u;
    float alpha = 1. - exp(-dt / (4. * wavelength));
    float average = mix(intensity[id], u * u, alpha);
    intensity[id] = average;
    if (p.x == screenX()) detector[p.y + n * p.z] = average;
}
)GLSL";

// Per-workgroup partial sums and maxima; see optics::Metrics for the fields.
inline constexpr std::string_view reduce = R"GLSL(
shared vec4 a[128], b[128], c[128];
void main() {
    int id = int(gl_GlobalInvocationID.x);
    uint k = gl_LocalInvocationID.x;
    vec4 total = vec4(0), bound = vec4(0), spatial = vec4(0);
    if (id < n * n * n) {
        ivec3 p = coord(id);
        float u = field[id], prev = oldField[id], avg = intensity[id], mass = medium[id].x;
        bool bad = isnan(u) || isinf(u) || isnan(prev) || isinf(prev) || isnan(avg) || isinf(avg) || avg < 0.;
        if (!bad) {
            float h = 1. / float(n - 1), velocity = (u - prev) / dt, gradient = 0.;
            if (p.x < n - 1) gradient += (field[id + 1] - u) * (oldField[id + 1] - prev);
            if (p.y < n - 1) gradient += (field[id + n] - u) * (oldField[id + n] - prev);
            if (p.z < n - 1) gradient += (field[id + n * n] - u) * (oldField[id + n * n] - prev);
            total = vec4(.5 * (mass * velocity * velocity + gradient / (h * h)) * h * h * h, avg * h * h * h, 0,
                         mass == 0. ? u * u : 0.);
            bound = vec4(abs(u), avg, p.x == screenX() ? avg : 0., abs(float(p.z) / float(n - 1) - .5) > .18 ? abs(u) : 0.);
            spatial = vec4(p.x > int(.6 * float(n - 1)) ? avg * h * h * h : 0., u * u * h * h * h, 0, 0);
        } else total.z = 1.;
    }
    a[k] = total; b[k] = bound; c[k] = spatial;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (k < s) { a[k] += a[k + s]; b[k] = max(b[k], b[k + s]); c[k] += c[k + s]; }
        barrier();
    }
    if (k == 0u) {
        uint o = gl_WorkGroupID.x + 1u;
        metrics[o].totals = a[0]; metrics[o].bounds = b[0]; metrics[o].spatial = c[0];
    }
}
)GLSL";

// Folds the per-workgroup partials into metrics[0]. Runs as one workgroup of 128 invocations.
inline constexpr std::string_view finalReduce = R"GLSL(
shared vec4 a[128], b[128], c[128];
void main() {
    uint k = gl_LocalInvocationID.x;
    vec4 total = vec4(0), bound = vec4(0), spatial = vec4(0);
    int groups = (n * n * n + 127) / 128;
    for (int i = int(k) + 1; i <= groups; i += 128) {
        total += metrics[i].totals; bound = max(bound, metrics[i].bounds); spatial += metrics[i].spatial;
    }
    a[k] = total; b[k] = bound; c[k] = spatial;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (k < s) { a[k] += a[k + s]; b[k] = max(b[k], b[k + s]); c[k] += c[k + s]; }
        barrier();
    }
    if (k == 0u) { metrics[0].totals = a[0]; metrics[0].bounds = b[0]; metrics[0].spatial = c[0]; }
}
)GLSL";

inline constexpr std::string_view vertex = R"GLSL(
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2. - 1., 0, 1);
}
)GLSL";

// Ray-marched volume, a single slice, or the detector plane, with the detector inset.
inline constexpr std::string_view fragment = R"GLSL(
in vec2 uv;
out vec4 color;

float valueAt(int id, bool averaged) { return averaged ? intensity[id] : field[id]; }
// Trilinear sample of the amplitude or the mean intensity at a domain position.
float sampleField(vec3 p, bool averaged) {
    vec3 q = clamp(p, 0., 1.) * float(n - 1);
    ivec3 lo = min(ivec3(q), ivec3(n - 2));
    vec3 f = q - vec3(lo);
    int i = indexOf(lo);
    float x00 = mix(valueAt(i, averaged), valueAt(i + 1, averaged), f.x);
    float x10 = mix(valueAt(i + n, averaged), valueAt(i + n + 1, averaged), f.x);
    float x01 = mix(valueAt(i + n * n, averaged), valueAt(i + n * n + 1, averaged), f.x);
    float x11 = mix(valueAt(i + n * n + n, averaged), valueAt(i + n * n + n + 1, averaged), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}
// Dark blue through cyan to yellow.
vec3 heat(float t) {
    t = clamp(t, 0., 1.);
    return mix(mix(vec3(.025, .035, .12), vec3(.03, .65, .85), smoothstep(0., .55, t)), vec3(1., .87, .24),
               smoothstep(.48, 1., t));
}
// Bilinear detector sample; horizontal is z, vertical is y.
float screenValue(vec2 p) {
    vec2 q = clamp(p, 0., 1.) * float(n - 1);
    ivec2 lo = min(ivec2(q), ivec2(n - 2));
    vec2 f = q - vec2(lo);
    int i = lo.y + n * lo.x;
    return mix(mix(detector[i], detector[i + n], f.x), mix(detector[i + 1], detector[i + n + 1], f.x), f.y);
}
// Logarithmic scale normalized by the current detector peak.
vec3 screenColor(vec2 p) {
    float normalized = screenValue(p) / max(metrics[0].bounds.z, 1e-12);
    return heat(log(1. + 30. * normalized) / log(31.));
}
bool boxIntersection(vec3 origin, vec3 direction, out float first, out float last) {
    vec3 inverse = 1. / (direction + vec3(1e-20));
    vec3 a = -origin * inverse, b = (1. - origin) * inverse;
    vec3 low = min(a, b), high = max(a, b);
    first = max(max(low.x, low.y), low.z);
    last = min(min(high.x, high.y), high.z);
    return last > max(first, 0.);
}
bool inside(vec3 p) { return all(greaterThanEqual(p, vec3(0))) && all(lessThanEqual(p, vec3(1))); }
// 1 on the cube edges (within 0.005 of two faces at once).
float frame(vec3 p) {
    vec3 d = min(p, 1. - p);
    return float(int(d.x < .005) + int(d.y < .005) + int(d.z < .005) >= 2);
}

void main() {
    if (display == 4) {   // detector only
        vec2 p = (uv - .5) * vec2(aspect, 1.) / .86 + .5;
        vec3 col = all(greaterThanEqual(p, vec2(0))) && all(lessThanEqual(p, vec2(1))) ? screenColor(p) : vec3(.035, .05, .075);
        color = vec4(col, 1);
        return;
    }
    vec3 ray = normalize(cameraForward.xyz + (2. * uv.x - 1.) * aspect * tanHalfFov * cameraRight.xyz +
                         (2. * uv.y - 1.) * tanHalfFov * cameraUp.xyz);
    vec3 origin = eye.xyz;
    vec3 background = mix(vec3(.018, .028, .05), vec3(.055, .078, .11), uv.y);
    float ground = (-.055 - origin.y) / ray.y;   // floor grid below the cube
    if (ground > 0.) {
        vec3 p = origin + ground * ray;
        if (all(greaterThan(p.xz, vec2(-.15))) && all(lessThan(p.xz, vec2(1.15)))) {
            vec2 grid = abs(fract(p.xz * 10. - .5) - .5) / max(fwidth(p.xz * 10.), vec2(.001));
            background = mix(vec3(.08, .105, .14), vec3(.16, .21, .27), (1. - min(min(grid.x, grid.y), 1.)) * .6);
        }
    }
    float first, last;
    vec3 result = background;
    if (boxIntersection(origin, ray, first, last)) {
        first = max(first, 0.);
        vec3 accumulated = vec3(0);
        float alpha = 0.;
        if (display < 2) {   // volume: front-to-back absorption/emission
            float stepLength = .9 / float(n);
            int steps = min(360, int(ceil((last - first) / stepLength)));
            for (int i = 0; i < steps && alpha < .985; ++i) {
                float t = first + (float(i) + .5) * stepLength;
                if (t > last) break;
                vec3 p = origin + ray * t;
                float value = sampleField(p, display == 1);
                float magnitude = display == 1 ? sqrt(max(value, 0.)) : abs(value);
                vec3 shade = display == 1 ? heat(1. - exp(-magnitude * exposure))
                                          : (value > 0. ? vec3(1., .34, .08) : vec3(.045, .55, 1.));
                float a = 1. - exp(-magnitude * exposure * stepLength * 8.);
                if (scene < 2 && abs(p.x - float(barrierX()) / float(n - 1)) < stepLength * .7 && !hole(p)) {
                    shade = mix(shade, vec3(.61, .72, .80), .75);   // the screen
                    a = max(a, .12);
                }
                if (scene == 2 && abs(length(p - lensCenter) - opening) < stepLength * .65) {
                    shade = mix(shade, vec3(.22, .95, .72), .7);    // the lens outline
                    a = max(a, .08);
                }
                accumulated += (1. - alpha) * a * shade;
                alpha += (1. - alpha) * a;
            }
        } else {   // slice: intersect the ray with the selected plane
            float t = (slicePosition - origin[sliceAxis]) / ray[sliceAxis];
            vec3 p = origin + ray * t;
            if (t >= first && t <= last && inside(p)) {
                float value = sampleField(p, display == 3);
                vec3 shade = display == 3 ? heat(1. - exp(-sqrt(max(value, 0.)) * exposure))
                                          : mix(vec3(.025, .035, .06), value > 0. ? vec3(1., .39, .12) : vec3(.045, .56, 1.),
                                                1. - exp(-abs(value) * exposure));
                vec2 m = medium[indexOf(clamp(ivec3(round(p * float(n - 1))), ivec3(0), ivec3(n - 1)))];
                if (m.x == 0.) shade = vec3(.66, .74, .8);                       // opaque screen
                if (m.x > 1.01) shade = mix(shade, vec3(.1, .6, .35), .20);      // inside the lens
                accumulated = shade;
                alpha = 1.;
            }
        }
        result = accumulated + (1. - alpha) * background;
        result = mix(result, vec3(.52, .67, .79), max(frame(origin + first * ray), frame(origin + last * ray)) * .8);
        // Outline of the slice / mouse-source plane and the brush position.
        float t = (slicePosition - origin[sliceAxis]) / ray[sliceAxis];
        vec3 p = origin + t * ray;
        if (t >= first && t <= last && inside(p)) {
            vec3 edge = min(p, 1. - p);
            edge[sliceAxis] = 1.;
            if (min(min(edge.x, edge.y), edge.z) < .006) result = mix(result, vec3(.30, .82, .76), .8);
            if (brushActive != 0 && length(p - brush) < 2. / float(n)) result = vec3(1., .95, .65);
        }
    }
    if (uv.x > .73 && uv.x < .98 && uv.y > .025 && uv.y < .31) {   // detector inset, bottom right
        vec2 p = vec2((uv.x - .742) / .226, (uv.y - .04) / .252);
        result = all(greaterThanEqual(p, vec2(0))) && all(lessThanEqual(p, vec2(1))) ? screenColor(p) : vec3(.26, .36, .45);
    }
    color = vec4(result, 1);
}
)GLSL";
} // namespace optics::shaders
