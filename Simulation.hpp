#pragma once
// GPU-resident scalar wave solver: three rotating field buffers, the intensity average, the medium
// and the detector live on the GPU; the CPU orchestrates dispatches and reads back a 48-byte summary.
#include "GpuApi.hpp"
#include "Math3D.hpp"
#include "Model.hpp"
#include "OpticsShaders.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace optics {
class Simulation {
  public:
    Settings settings;
    Metrics metrics{};
    double time = 0;         // simulated time, domain-crossing units
    std::uint64_t steps = 0; // FDTD steps since reset
    float gpuMs = 0;         // most recently completed timing of one tick

    explicit Simulation(GpuApi &api) : gpu(api) {
        auto kernel = [&](std::string_view body, const char *label) {
            return gpu.program({{gl::COMPUTE_SHADER,
                                 {shaders::version, shaders::computeLayout, shaders::paramsBlock, shaders::common, body}}},
                               label);
        };
        programs.initialize = kernel(shaders::initialize, "Initialize optical domain");
        programs.step = kernel(shaders::step, "3D scalar wave FDTD");
        programs.reduce = kernel(shaders::reduce, "Wave diagnostics");
        programs.finalReduce = kernel(shaders::finalReduce, "Final wave reduction");
        programs.render = gpu.program(
            {{gl::VERTEX_SHADER, {shaders::version, shaders::vertex}},
             {gl::FRAGMENT_SHADER,
              {shaders::version, shaders::paramsBlock, shaders::viewBlock, shaders::common, shaders::fragment}}},
            "Volume, slice and detector renderer");
        for (GLuint &field : fields)
            field = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 0, sizeof(float), gl::DYNAMIC_COPY);
        buffers.intensity = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 3, sizeof(float), gl::DYNAMIC_COPY);
        buffers.medium = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 4, sizeof(float) * 2, gl::DYNAMIC_COPY);
        buffers.metrics = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 5, sizeof(Metrics), gl::DYNAMIC_COPY);
        buffers.detector = gpu.buffer(gl::SHADER_STORAGE_BUFFER, 6, sizeof(float), gl::DYNAMIC_COPY);
        buffers.params = gpu.buffer(gl::UNIFORM_BUFFER, 0, sizeof(Params), gl::DYNAMIC_DRAW);
        buffers.view = gpu.buffer(gl::UNIFORM_BUFFER, 1, sizeof(ViewParams), gl::DYNAMIC_DRAW);
        gpu.GenVertexArrays(1, &emptyVao); // attribute-less fullscreen triangle
        gpu.GenQueries(1, &query);
        gpu.GenFramebuffers(1, &fbo);
        glGenTextures(1, &renderTexture);
        reset();
    }
    ~Simulation() {
        glDeleteTextures(1, &renderTexture);
        gpu.DeleteFramebuffers(1, &fbo);
        gpu.DeleteQueries(1, &query);
        gpu.DeleteVertexArrays(1, &emptyVao);
        for (GLuint buffer : {fields[0], fields[1], fields[2], buffers.intensity, buffers.medium, buffers.metrics,
                              buffers.detector, buffers.params, buffers.view})
            gpu.DeleteBuffers(1, &buffer);
        for (GLuint program : {programs.initialize, programs.step, programs.reduce, programs.finalReduce,
                               programs.render})
            gpu.DeleteProgram(program);
    }
    Simulation(const Simulation &) = delete;
    Simulation &operator=(const Simulation &) = delete;

    int cells() const {
        return settings.cells();
    }

    /// Reallocates the fields for the current grid, rebuilds the medium and clears the wave.
    void reset() {
        if (settings.n < minGrid || settings.n > maxGrid || settings.wavelength <= 0 || settings.refractiveIndex < 1)
            throw std::runtime_error("Unsupported optical grid or material parameters.");
        if (timerPending) {
            std::uint64_t ignored{};
            gpu.GetQueryObjectui64v(query, gl::QUERY_RESULT, &ignored); // drain before reuse
            timerPending = false;
        }
        gpuMs = 0;
        time = 0;
        steps = 0;
        settings.brushActive = false;
        const std::size_t count = static_cast<std::size_t>(cells());
        for (GLuint field : fields)
            allocate(field, count * sizeof(float));
        allocate(buffers.intensity, count * sizeof(float));
        allocate(buffers.medium, count * sizeof(float) * 2);
        allocate(buffers.metrics, static_cast<std::size_t>(workgroups(cells()) + 1) * sizeof(Metrics));
        allocate(buffers.detector, static_cast<std::size_t>(settings.n) * static_cast<std::size_t>(settings.n) * sizeof(float));
        bindFields();
        uploadParams();
        dispatch(programs.initialize, cells());
        measure();
    }

    /// Advances `stepCount` FDTD steps (one nominal tick by default) and refreshes the diagnostics.
    void tick(int stepCount = 0) {
        if (stepCount <= 0)
            stepCount = settings.stepsPerTick();
        pollTimer();
        const bool timing = !timerPending;
        if (timing)
            gpu.BeginQuery(gl::TIME_ELAPSED, query);
        for (int i = 0; i < stepCount; ++i) {
            ++steps;
            time = static_cast<double>(steps) * static_cast<double>(settings.stepSize());
            uploadParams(); // the source phase depends on the current time
            dispatch(programs.step, cells());
            rotateFields();
        }
        reduceDiagnostics();
        if (timing) {
            gpu.EndQuery(gl::TIME_ELAPSED);
            timerPending = true;
        }
    }

    /// Runs the diagnostic reductions and reads back the 48-byte summary (a GPU sync point).
    void measure() {
        uploadParams();
        reduceDiagnostics();
        gpu.MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        gpu.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffers.metrics);
        gpu.GetBufferSubData(gl::SHADER_STORAGE_BUFFER, 0, sizeof(metrics), &metrics);
    }
    bool healthy() const {
        return metrics.invalid() == 0 && std::isfinite(metrics.energy()) && metrics.energy() > -1e-5f &&
               metrics.maxAmplitude() < 1000 && metrics.screenLeak() == 0;
    }

    /// Writes the raw detector plane: y, z and the mean squared amplitude for every grid point.
    void exportDetector(const std::filesystem::path &path) {
        const auto data = detectorSnapshot();
        const int n = settings.n;
        const double spacing = 1. / (n - 1);
        std::string text;
        text.reserve(data.size() * 40);
        auto out = std::back_inserter(text);
        std::format_to(out, "# Wave Optics Lab 3D; EMA of squared scalar amplitude; not calibrated radiance\n");
        std::format_to(out, "# n={}, time={}, wavelength={}, x={}\n", n, time, settings.wavelength,
                       std::round(detectorPlane * (n - 1)) * spacing);
        std::format_to(out, "y,z,mean_squared_amplitude\n");
        for (int z = 0; z < n; ++z)
            for (int y = 0; y < n; ++y)
                std::format_to(out, "{},{},{}\n", y * spacing, z * spacing, data[static_cast<std::size_t>(y + n * z)]);
        std::ofstream file(path);
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!file)
            throw std::runtime_error("Unable to write detector CSV.");
    }

    /// Renders the scene into an offscreen target (half resolution for volume modes) and blits it.
    void draw(const Camera &camera, const Viewport &canvas) {
        const bool halve = settings.volumeDisplay();
        const int w = halve ? (canvas.width + 1) / 2 : canvas.width, h = halve ? (canvas.height + 1) / 2 : canvas.height;
        gpu.BindFramebuffer(gl::FRAMEBUFFER, fbo);
        ensureRenderTarget(w, h);
        glViewport(0, 0, w, h);
        glDisable(GL_BLEND);
        glDisable(GL_SCISSOR_TEST);
        uploadParams();
        uploadView(camera, static_cast<float>(canvas.width) / static_cast<float>(canvas.height));
        gpu.MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT); // compute results feed the fragment shader
        gpu.UseProgram(programs.render);
        gpu.BindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        gpu.BindFramebuffer(gl::READ_FRAMEBUFFER, fbo);
        gpu.BindFramebuffer(gl::DRAW_FRAMEBUFFER, 0);
        gpu.BlitFramebuffer(0, 0, w, h, canvas.x, canvas.y, canvas.x + canvas.width, canvas.y + canvas.height,
                            GL_COLOR_BUFFER_BIT, GL_LINEAR);
        gpu.BindFramebuffer(gl::FRAMEBUFFER, 0);
    }

  private:
    GpuApi &gpu;
    struct {
        GLuint initialize{}, step{}, reduce{}, finalReduce{}, render{};
    } programs;
    struct {
        GLuint intensity{}, medium{}, metrics{}, detector{}, params{}, view{};
    } buffers;
    std::array<GLuint, 3> fields{}; // rotated each step: [current, previous, next]
    GLuint emptyVao{}, query{}, fbo{}, renderTexture{};
    int renderW = 0, renderH = 0;
    bool timerPending = false;

    void allocate(GLuint buffer, std::size_t size) {
        gpu.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffer);
        gpu.BufferData(gl::SHADER_STORAGE_BUFFER, static_cast<std::ptrdiff_t>(size), nullptr, gl::DYNAMIC_COPY);
    }
    /// Binding 0 reads the current level, 1 the previous one, 2 receives the next one.
    void bindFields() {
        for (GLuint slot = 0; slot < 3; ++slot)
            gpu.BindBufferBase(gl::SHADER_STORAGE_BUFFER, slot, fields[slot]);
    }
    void rotateFields() {
        std::rotate(fields.begin(), fields.begin() + 2, fields.end()); // next -> current -> previous
        bindFields();
    }
    void dispatch(GLuint program, int invocations) {
        gpu.UseProgram(program);
        gpu.DispatchCompute(static_cast<GLuint>(workgroups(invocations)), 1, 1);
        gpu.MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);
    }
    void reduceDiagnostics() {
        dispatch(programs.reduce, cells());
        dispatch(programs.finalReduce, workgroupSize);
    }
    std::vector<float> detectorSnapshot() {
        std::vector<float> data(static_cast<std::size_t>(settings.n) * static_cast<std::size_t>(settings.n));
        gpu.MemoryBarrier(gl::BUFFER_UPDATE_BARRIER_BIT);
        gpu.BindBuffer(gl::SHADER_STORAGE_BUFFER, buffers.detector);
        gpu.GetBufferSubData(gl::SHADER_STORAGE_BUFFER, 0, static_cast<std::ptrdiff_t>(data.size() * sizeof(float)),
                             data.data());
        return data;
    }
    void uploadParams() {
        const Params params{
            .n = settings.n,
            .scene = std::to_underlying(settings.scene),
            .sourceMode = std::to_underlying(settings.source),
            .absorbing = settings.absorbing ? 1 : 0,
            .brushActive = settings.brushActive ? 1 : 0,
            .display = std::to_underlying(settings.display),
            .sliceAxis = settings.sliceAxis,
            .dt = settings.stepSize(),
            .time = static_cast<float>(time),
            .wavelength = settings.wavelength,
            .opening = settings.opening,
            .separation = settings.separation,
            .refractiveIndex = settings.refractiveIndex,
            .exposure = settings.exposure,
            .slicePosition = settings.slice,
            .padding0 = 0,
            .brush = settings.brush,
            .padding1 = 0,
        };
        gpu.BindBuffer(gl::UNIFORM_BUFFER, buffers.params);
        gpu.BufferSubData(gl::UNIFORM_BUFFER, 0, sizeof(params), &params);
    }
    void uploadView(const Camera &camera, float aspect) {
        auto f4 = [](V3 v) { return F4{v.x, v.y, v.z, 0}; };
        const ViewParams view{
            .eye = f4(camera.eye()),
            .forward = f4(camera.forward()),
            .right = f4(camera.right()),
            .up = f4(camera.up()),
            .aspect = aspect,
            .tanHalfFov = std::tan(fieldOfView / 2),
            .padding = {},
        };
        gpu.BindBuffer(gl::UNIFORM_BUFFER, buffers.view);
        gpu.BufferSubData(gl::UNIFORM_BUFFER, 0, sizeof(view), &view);
    }
    void ensureRenderTarget(int w, int h) {
        if (w == renderW && h == renderH)
            return;
        renderW = w;
        renderH = h;
        glBindTexture(GL_TEXTURE_2D, renderTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, gl::RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE);
        gpu.FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);
        if (gpu.CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("Volume render framebuffer is incomplete.");
    }
    /// Collects the previous tick's timing once the driver has it, without stalling.
    void pollTimer() {
        if (!timerPending)
            return;
        GLuint ready{};
        gpu.GetQueryObjectuiv(query, gl::QUERY_RESULT_AVAILABLE, &ready);
        if (!ready)
            return;
        std::uint64_t nanoseconds{};
        gpu.GetQueryObjectui64v(query, gl::QUERY_RESULT, &nanoseconds);
        gpuMs = static_cast<float>(nanoseconds) * 1e-6f;
        timerPending = false;
    }
};
} // namespace optics
