// tetraplex_recorder.swift
//
// A deliberately-stubbish SwiftUI app that runs the tetraplex demo in-process
// against the real engine and captures frames via AVAssetWriter to an mp4.
// No HTTP bridge, no browser, no external screen-capture tooling. The whole
// point is to produce a recording of the *actual kernel path* without any
// user-led GUI actions.
//
// We do not intend to maintain this file. It exists because MacBook tooling
// is frustrating enough that spinning up a dedicated recorder in 300 lines
// of Swift is cheaper than learning ScreenCaptureKit or wiring OBS.
//
// Build via `make tetraplex_recorder` at the repo root; the Makefile feeds
// all the engine sources + this file through swiftc with -parse-as-library,
// so the `@main` struct below is the executable entry point (main.swift is
// excluded from this target).
//
// Run:
//   GGUF_PATH=/path/to/model.gguf ./tetraplex_recorder
//   # writes tetraplex-demo.mp4 to CWD and quits when all 4 streams finish.

import SwiftUI
import AVFoundation
import AppKit
import Metal
import Foundation

// =============================================================================
// Per-stream state — one ObservableObject per pane.
// =============================================================================

final class StreamState: ObservableObject, Identifiable {
    let id: String
    let color: Color
    let prefix: String           // text before the image chunk (empty if no image)
    let suffix: String           // text after the image chunk (or the whole prompt if no image)
    let imagePath: String?       // nil = text-only stream
    let promptLabel: String      // short human-readable description of what this pane asks
    @Published var text: String = ""
    @Published var tokenCount: Int = 0
    @Published var ttftSec: Double? = nil
    @Published var pages: [TetraplexPageCell] = []
    @Published var position: Int = 0
    @Published var stateLabel: String = "idle"
    @Published var instantRate: Double = 0

    // Non-published engine handles.
    fileprivate var session: Session?
    fileprivate var submitTime: Date?
    fileprivate var arrivalTimes: [Date] = []

    // Text-only convenience init — prompt becomes the "suffix" (whole prompt).
    init(id: String, color: Color, prompt: String) {
        self.id = id; self.color = color
        self.prefix = ""; self.suffix = prompt
        self.imagePath = nil
        self.promptLabel = prompt.replacingOccurrences(of: "<|turn>user\n", with: "")
            .replacingOccurrences(of: "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", with: "")
    }

    // Multimodal init: prefix + <image> + suffix, with a visible thumbnail.
    init(id: String, color: Color, imagePath: String, userPrompt: String) {
        self.id = id; self.color = color
        self.imagePath = imagePath
        self.prefix = "<|turn>user\n"
        self.suffix = "\n\(userPrompt)<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        self.promptLabel = userPrompt
    }
}

struct TetraplexPageCell: Identifiable {
    let id: Int     // phys page id
    let refcount: Int
}

// =============================================================================
// DemoState — drives the engine, pumps tick(), updates the streams.
// =============================================================================

final class DemoState: ObservableObject {
    @Published var streams: [StreamState]
    @Published var status: String = "loading…"
    @Published var aggregateRate: Double = 0
    // Running aggregate rate samples for the chart (t_s_since_start, rate).
    @Published var rateHistory: [(Double, Double)] = []
    // Per-stream instantaneous rate history, same time axis as above.
    @Published var streamRateHistory: [String: [(Double, Double)]] = [:]
    @Published var yMaxHigh: Double = 60
    @Published var demoStart: Date = Date()
    @Published var finished: Bool = false

    private var engine: LmEngine?
    private var pumpThread: Thread?
    private var running = false

    init() {
        // Mode selector: TETRAPLEX_MODE=multimodal switches to the 4-image
        // variant. Otherwise the default text-only prompts run.
        let mode = ProcessInfo.processInfo.environment["TETRAPLEX_MODE"] ?? "text"
        switch mode {
        case "multimodal":
            let frames = "/Users/mdot/metal-microbench/test_data/frames"
            self.streams = [
                StreamState(id: "A", color: .cyan,
                            imagePath: "\(frames)/frame_00_fbb737dcf6b0.png",
                            userPrompt: "Describe this image in one short sentence."),
                StreamState(id: "B", color: .orange,
                            imagePath: "\(frames)/frame_100_933b3a8dd9ce.png",
                            userPrompt: "Describe this image in one short sentence."),
                StreamState(id: "C", color: .green,
                            imagePath: "\(frames)/frame_200_77689ab7b649.png",
                            userPrompt: "Write a four-line sonnet about this image."),
                StreamState(id: "D", color: .pink,
                            imagePath: "\(frames)/frame_300_38b60f37136f.png",
                            userPrompt: "Write a four-line sonnet about this image."),
            ]
        default:
            self.streams = [
                StreamState(id: "A", color: .cyan,
                            prompt: "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
                StreamState(id: "B", color: .orange,
                            prompt: "<|turn>user\nWrite a one-sentence haiku about compilers.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
                StreamState(id: "C", color: .green,
                            prompt: "<|turn>user\nList three colors and their hex codes.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
                StreamState(id: "D", color: .pink,
                            prompt: "<|turn>user\nCount from one to ten out loud.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"),
            ]
        }
        for s in streams { streamRateHistory[s.id] = [] }
    }

    private var visionWeights: VisionWeights?

    func start() {
        guard !running else { return }
        running = true
        status = "loading engine…"
        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            bootstrapGlobalState()
            let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"]
                ?? "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
            do {
                let w = try loadLmWeights(ggufPath: ggufPath)
                let eng = LmEngine(weights: w)
                // If any stream needs vision, load vision weights too.
                let anyImages = self.streams.contains { $0.imagePath != nil }
                var visW: VisionWeights? = nil
                if anyImages {
                    DispatchQueue.main.async { self.status = "loading vision weights…" }
                    let visionPath = ProcessInfo.processInfo.environment["VISION_ST"]
                        ?? "/Users/mdot/models/gemma-4-a4b-bf16/model-00001-of-00002.safetensors"
                    let st = try SafetensorsFile(visionPath)
                    visW = try loadVisionWeights(st, device: device)
                }
                DispatchQueue.main.async {
                    self.engine = eng
                    self.visionWeights = visW
                    self.status = anyImages
                        ? "engine + vision ready, running vision tower on 4 images…"
                        : "engine ready, dispatching 4 streams…"
                    self.kickoff()
                }
            } catch {
                DispatchQueue.main.async { self.status = "load failed: \(error)" }
            }
        }
    }

    private func kickoff() {
        guard let engine else { return }
        // Pre-run vision tower for any streams with images BEFORE we begin
        // recording, so the mp4 only captures the AR decode phase (vision is
        // 7 s of solid GPU work per image, very boring to watch). Softs are
        // kept in MTLBuffers until submit below.
        struct ImageSofts {
            let stream: StreamState
            let softs: MTLBuffer  // padded to 280 rows, fp32
            let count: Int
        }
        var visionDone: [ImageSofts] = []
        if let visWeights = visionWeights {
            for stream in streams {
                guard let path = stream.imagePath else { continue }
                do {
                    let batch = try gemma4ImagePreprocess(path: path, device: device)
                    let (raw, nPooled) = runVisionTowerForward(batch: batch, weights: visWeights,
                                                                device: device, queue: queue)
                    let target = 280
                    let padded = device.makeBuffer(length: target * HIDDEN * 4,
                                                    options: .storageModeShared)!
                    memset(padded.contents(), 0, padded.length)
                    memcpy(padded.contents(), raw.contents(), min(nPooled, target) * HIDDEN * 4)
                    visionDone.append(ImageSofts(stream: stream, softs: padded, count: target))
                } catch {
                    print("vision tower failed for \(stream.id): \(error)")
                }
            }
        }

        demoStart = Date()
        status = "dispatching \(streams.count) streams"
        let BOI: UInt32 = 255999
        let EOI: UInt32 = 258882
        for stream in streams {
            guard let s = engine.openSession(maxNewTokens: 256) else { continue }
            stream.session = s
            stream.submitTime = Date()

            if let img = visionDone.first(where: { $0.stream === stream }) {
                // prefix text tokens (with BOS once) → BOI → softs → EOI → suffix text
                s.submit(engine.tokenize(stream.prefix, addBos: true))
                s.submit([BOI])
                s.submit(softTokens: img.softs, count: img.count, isFp32: true)
                s.submit([EOI])
                s.submit(engine.tokenize(stream.suffix, addBos: false))
            } else {
                // Text-only path: "suffix" carries the whole prompt.
                s.submit(engine.tokenize(stream.suffix, addBos: true))
            }
        }
        pumpThread = Thread { [weak self] in self?.pumpLoop() }
        pumpThread?.name = "gemma-pump"
        pumpThread?.start()
    }

    private func pumpLoop() {
        guard let engine else { return }
        while running {
            if engine.hasWork {
                _ = engine.tick()
            } else {
                Thread.sleep(forTimeInterval: 0.001)
            }
            // Drain tokens + update per-stream state + KV snapshot.
            var anyAlive = false
            for stream in streams {
                guard let s = stream.session else { continue }
                if s.state != .done { anyAlive = true }
                var newTokens: [UInt32] = []
                while let t = s.nextToken() { newTokens.append(t) }
                let pages = s.ownedPagesForDebug
                let pos = s.positionForDebug
                let stLabel: String
                switch s.state {
                case .idle: stLabel = "idle"
                case .priming: stLabel = "priming"
                case .generating: stLabel = "generating"
                case .paused: stLabel = "paused"
                case .done: stLabel = "done"
                }
                let cells = pages.map { phys -> TetraplexPageCell in
                    let rc = engine.pageManager.pageRefcount(phys)
                    return TetraplexPageCell(id: phys, refcount: rc)
                }
                let frag = newTokens.isEmpty ? "" : engine.detokenize(newTokens)
                DispatchQueue.main.async {
                    if !frag.isEmpty {
                        if stream.ttftSec == nil, let sub = stream.submitTime {
                            stream.ttftSec = Date().timeIntervalSince(sub)
                        }
                        stream.text += frag
                        stream.tokenCount += newTokens.count
                        let now = Date()
                        for _ in newTokens { stream.arrivalTimes.append(now) }
                    }
                    stream.pages = cells
                    stream.position = pos
                    stream.stateLabel = stLabel
                }
            }
            if !anyAlive {
                // Give the UI / recorder a beat to capture the final frame.
                Thread.sleep(forTimeInterval: 0.8)
                DispatchQueue.main.async { self.finish() }
                return
            }
        }
    }

    // Sliding 500 ms instantaneous rate for a pane.
    func rate(for stream: StreamState, now: Date = Date(), windowMs: Double = 500) -> Double {
        let cutoff = now.addingTimeInterval(-windowMs / 1000)
        var n = 0
        for t in stream.arrivalTimes.reversed() {
            if t < cutoff { break }
            n += 1
        }
        return Double(n) / (windowMs / 1000)
    }

    // Called at ~10 Hz from the UI to append a rate sample to the history.
    func sampleRates() {
        let now = Date()
        let tSec = now.timeIntervalSince(demoStart)
        var agg = 0.0
        for stream in streams {
            let r = rate(for: stream, now: now)
            stream.instantRate = r
            agg += r
            if var arr = streamRateHistory[stream.id] {
                arr.append((tSec, r))
                // Keep last 30 s.
                while arr.count > 1 && arr.first!.0 < tSec - 30 { arr.removeFirst() }
                streamRateHistory[stream.id] = arr
            }
        }
        aggregateRate = agg
        rateHistory.append((tSec, agg))
        while rateHistory.count > 1 && rateHistory.first!.0 < tSec - 30 { rateHistory.removeFirst() }
        // High-water decay so the plateau stays visible after streams finish.
        if agg > yMaxHigh { yMaxHigh = agg }
        yMaxHigh = max(60, yMaxHigh * 0.996)
    }

    func finish() {
        running = false
        finished = true
        status = "done — closing in 2 s"
    }
}

// =============================================================================
// SwiftUI views.
// =============================================================================

struct TetraplexView: View {
    @ObservedObject var demo: DemoState

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Text("gemma metal bridge — tetraplex")
                    .foregroundColor(Color(red: 0.4, green: 0.67, blue: 0.6))
                    .font(.system(.body, design: .monospaced))
                Spacer()
                Text(demo.status)
                    .foregroundColor(Color(white: 0.55))
                    .font(.system(size: 11, design: .monospaced))
            }
            .padding(.horizontal, 8)

            HStack(spacing: 8) {
                VStack(spacing: 8) {
                    StreamPaneView(stream: demo.streams[0])
                    StreamPaneView(stream: demo.streams[2])
                }
                VStack(spacing: 8) {
                    StreamPaneView(stream: demo.streams[1])
                    StreamPaneView(stream: demo.streams[3])
                }
            }
            .frame(maxHeight: .infinity)

            BandwidthChart(demo: demo)
                .frame(height: 160)
        }
        .padding(10)
        .background(Color(white: 0.04))
        .foregroundColor(Color(white: 0.84))
        .font(.system(size: 12, design: .monospaced))
    }
}

/// Keep only the trailing `maxLines` lines of a string. ImageRenderer can't
/// scroll, so we clip on the model side.
func truncatedTail(_ s: String, maxLines: Int) -> String {
    let lines = s.split(separator: "\n", omittingEmptySubsequences: false)
    if lines.count <= maxLines { return s }
    return lines.suffix(maxLines).joined(separator: "\n")
}

struct StreamPaneView: View {
    @ObservedObject var stream: StreamState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Circle().fill(stream.color).frame(width: 8, height: 8)
                Text("stream \(stream.id)").font(.system(size: 12, design: .monospaced))
                Spacer()
                if let ttft = stream.ttftSec {
                    Text(String(format: "ttft=%.2fs · %d tok · %.1f tok/s",
                                ttft, stream.tokenCount, stream.instantRate))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(Color(white: 0.5))
                }
            }
            // Prompt subtitle + optional image thumbnail so the viewer can
            // see what each pane is actually asking about.
            HStack(alignment: .top, spacing: 6) {
                if let path = stream.imagePath, let img = NSImage(contentsOfFile: path) {
                    Image(nsImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 40, height: 40)
                        .cornerRadius(3)
                        .overlay(RoundedRectangle(cornerRadius: 3).stroke(Color(white: 0.18), lineWidth: 1))
                }
                Text(stream.promptLabel)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(white: 0.55))
                    .lineLimit(3)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            // Plain Text, not ScrollView — ImageRenderer doesn't render
            // ScrollView contents off-screen. We tail the last ~18 lines
            // with a manual clip so long generations don't blow the pane
            // but still show recent output.
            Text(truncatedTail(stream.text, maxLines: 18))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Color(white: 0.82))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(6)
                .background(Color(white: 0.03))
                .overlay(RoundedRectangle(cornerRadius: 3).stroke(Color(white: 0.13), lineWidth: 1))
                .clipped()
            // kv tenancy strip
            KvStrip(stream: stream)
        }
        .padding(8)
        .background(Color(white: 0.07))
        .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color(white: 0.13), lineWidth: 1))
    }
}

struct KvStrip: View {
    @ObservedObject var stream: StreamState
    var body: some View {
        HStack(alignment: .center, spacing: 2) {
            Text("s\(stream.session?.id ?? 0) · \(stream.stateLabel) · \(stream.position) tok · \(stream.pages.count) pages")
                .font(.system(size: 9, design: .monospaced))
                .foregroundColor(Color(white: 0.4))
            ForEach(stream.pages) { p in
                pageCell(p)
            }
        }
        .padding(4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(white: 0.03))
        .overlay(RoundedRectangle(cornerRadius: 3).stroke(Color(white: 0.13), lineWidth: 1))
    }

    private func pageCell(_ p: TetraplexPageCell) -> some View {
        let color: Color = p.refcount >= 3 ? Color(red: 0.95, green: 0.2, blue: 0.4) :
                           p.refcount >= 2 ? Color(red: 1.0, green: 0.65, blue: 0.2) :
                                              Color(red: 0.16, green: 0.7, blue: 0.35)
        return RoundedRectangle(cornerRadius: 1.5).fill(color).frame(width: 6, height: 6)
    }
}

struct BandwidthChart: View {
    @ObservedObject var demo: DemoState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("bandwidth — tokens/sec (client-observed, sliding 500 ms)")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(Color(red: 0.4, green: 0.67, blue: 0.6))
                Spacer()
                ForEach(demo.streams) { s in
                    HStack(spacing: 3) {
                        Circle().fill(s.color).frame(width: 7, height: 7)
                        Text(s.id).font(.system(size: 10, design: .monospaced))
                    }
                }
                HStack(spacing: 3) {
                    Circle().fill(Color(white: 0.7)).frame(width: 7, height: 7)
                    Text("Σ").font(.system(size: 10, design: .monospaced))
                }
            }
            Canvas { ctx, size in
                let yMax = demo.yMaxHigh
                let now = Date().timeIntervalSince(demo.demoStart)
                let xStart = max(0.0, now - 30.0)
                let xEnd = max(30.0, now)
                func xToP(_ t: Double) -> CGFloat {
                    CGFloat((t - xStart) / (xEnd - xStart)) * size.width
                }
                func yToP(_ r: Double) -> CGFloat {
                    size.height - CGFloat(r / yMax) * size.height
                }
                // Grid.
                ctx.stroke(Path { p in
                    for i in 0...4 {
                        let y = size.height * CGFloat(i) / 4
                        p.move(to: CGPoint(x: 0, y: y)); p.addLine(to: CGPoint(x: size.width, y: y))
                    }
                }, with: .color(Color(white: 0.1)), lineWidth: 1)
                // Per-stream lines.
                for stream in demo.streams {
                    guard let arr = demo.streamRateHistory[stream.id], arr.count > 1 else { continue }
                    var path = Path()
                    var first = true
                    for (t, r) in arr where t >= xStart {
                        let pt = CGPoint(x: xToP(t), y: yToP(r))
                        if first { path.move(to: pt); first = false } else { path.addLine(to: pt) }
                    }
                    ctx.stroke(path, with: .color(stream.color), lineWidth: 1.5)
                }
                // Aggregate (behind, thicker, translucent).
                var aggPath = Path()
                var first = true
                for (t, r) in demo.rateHistory where t >= xStart {
                    let pt = CGPoint(x: xToP(t), y: yToP(r))
                    if first { aggPath.move(to: pt); first = false } else { aggPath.addLine(to: pt) }
                }
                ctx.stroke(aggPath, with: .color(Color(white: 0.8, opacity: 0.45)), lineWidth: 2.5)
                // Y-axis labels.
                for i in 0...4 {
                    let y = size.height * CGFloat(i) / 4
                    let val = Int(yMax * Double(4 - i) / 4)
                    ctx.draw(Text("\(val) tok/s").font(.system(size: 9, design: .monospaced))
                                .foregroundColor(Color(white: 0.3)),
                              at: CGPoint(x: 30, y: y + 6), anchor: .leading)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(white: 0.02))
            .overlay(RoundedRectangle(cornerRadius: 3).stroke(Color(white: 0.13), lineWidth: 1))

            HStack(spacing: 6) {
                ForEach(demo.streams) { s in
                    Text(String(format: "%@: %d tok @ %.1f tok/s", s.id, s.tokenCount, s.instantRate))
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(s.color)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color(white: 0.03))
                }
                Text(String(format: "Σ: %d tok @ %.1f tok/s",
                            demo.streams.reduce(0) { $0 + $1.tokenCount },
                            demo.aggregateRate))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(white: 0.85))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Color(white: 0.05))
            }
        }
    }
}

// =============================================================================
// Frame recorder — ImageRenderer → CVPixelBuffer → AVAssetWriter (mp4).
// =============================================================================

final class FrameRecorder {
    let url: URL
    let size: CGSize
    let fps: Int32
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private var frameCount: Int64 = 0
    private var started = false

    init?(url: URL, width: Int, height: Int, fps: Int) {
        self.url = url
        self.size = CGSize(width: width, height: height)
        self.fps = Int32(fps)
        try? FileManager.default.removeItem(at: url)
        guard let w = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
        self.writer = w
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        self.input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        self.input.expectsMediaDataInRealTime = true
        let pbAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        self.adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: pbAttrs)
        if writer.canAdd(input) { writer.add(input) } else { return nil }
    }

    func start() {
        guard !started else { return }
        started = true
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
    }

    func append(_ cgImage: CGImage) {
        guard started, input.isReadyForMoreMediaData else { return }
        guard let pb = Self.pixelBuffer(from: cgImage, width: Int(size.width), height: Int(size.height))
        else { return }
        let pts = CMTime(value: frameCount, timescale: fps)
        adaptor.append(pb, withPresentationTime: pts)
        frameCount += 1
    }

    func finish(_ completion: @escaping () -> Void) {
        input.markAsFinished()
        writer.finishWriting {
            completion()
        }
    }

    private static func pixelBuffer(from cgImage: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA,
                                         attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let buffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                   width: width, height: height,
                                   bitsPerComponent: 8,
                                   bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                   space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                              | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}

// =============================================================================
// App entry — @main.
// =============================================================================

@MainActor
final class RecordingDriver: ObservableObject {
    let demo: DemoState
    var recorder: FrameRecorder?
    var frameTimer: Timer?
    var rateTimer: Timer?
    let width: Int = 1280
    let height: Int = 800
    let fps: Int = 30
    private var recorderStarted = false
    private var trailingFramesAfterDone = 0

    init(demo: DemoState) { self.demo = demo }

    /// Resolve where to write the mp4. Preference order:
    ///   1. $TETRAPLEX_OUT if set (absolute or relative — whatever you give).
    ///   2. `./recordings/tetraplex-<yyyymmdd-hhmmss>.mp4` next to the binary.
    /// `recordings/` is gitignored so rapid iteration doesn't pollute `git status`.
    /// Finder CAN open /tmp but it's hidden and wiped at reboot — keeping
    /// outputs inside the repo tree is just friendlier.
    static func resolveOutputURL() -> URL {
        if let explicit = ProcessInfo.processInfo.environment["TETRAPLEX_OUT"], !explicit.isEmpty {
            return URL(fileURLWithPath: explicit)
        }
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let dir = cwd.appendingPathComponent("recordings", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        fmt.timeZone = TimeZone.current
        let stamp = fmt.string(from: Date())
        return dir.appendingPathComponent("tetraplex-\(stamp).mp4")
    }

    func begin() {
        // 10 Hz rate-sampling timer is always on — the chart history needs to
        // fill even before the recorder is armed so the first frames already
        // show some context.
        rateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.demo.sampleRates()
        }
        // 30 fps frame capture. The recorder itself only arms once the engine
        // has finished loading — there's no point baking ten seconds of
        // "loading engine…" into the mp4.
        frameTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / Double(fps), repeats: true) { [weak self] _ in
            guard let self else { return }
            self.captureFrameIfArmed()
        }
    }

    private func captureFrameIfArmed() {
        // Arm once engine has loaded AND sessions are open (so the first
        // recorded frame shows all four panes populated with their prompts).
        if !recorderStarted {
            let allHaveSession = demo.streams.allSatisfy { $0.session != nil }
            if !allHaveSession { return }
            let url = Self.resolveOutputURL()
            recorder = FrameRecorder(url: url, width: width, height: height, fps: fps)
            recorder?.start()
            recorderStarted = true
            print("[recorder] writing to \(url.path)")
        }

        let view = TetraplexView(demo: self.demo)
            .frame(width: CGFloat(self.width), height: CGFloat(self.height))
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2.0
        renderer.proposedSize = ProposedViewSize(width: CGFloat(self.width), height: CGFloat(self.height))
        if let cg = renderer.cgImage {
            recorder?.append(cg)
        }
        // Keep recording for ~3 more seconds after every stream hits .done so
        // the final page-count + aggregate-fall on the chart are visible.
        if demo.finished {
            trailingFramesAfterDone += 1
            if trailingFramesAfterDone > fps * 3 {
                wrapUp(url: recorder?.url)
            }
        }
    }

    func wrapUp(url: URL?) {
        frameTimer?.invalidate(); frameTimer = nil
        rateTimer?.invalidate(); rateTimer = nil
        recorder?.finish { [weak self] in
            print("[recorder] wrote \(url?.path ?? "<nil>")")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                NSApp.terminate(nil)
            }
            _ = self
        }
    }
}

@main
struct TetraplexRecorderApp: App {
    @StateObject var demo = DemoState()
    @State private var driver: RecordingDriver?

    init() {
        // Make this a regular dock-visible app (default for SwiftUI but
        // explicit so launching from a shell still produces a foregrounded
        // window) and activate it so the window stealably-focuses.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup {
            TetraplexView(demo: demo)
                .frame(width: 1280, height: 800)
                .onAppear {
                    demo.start()
                    let d = RecordingDriver(demo: demo)
                    driver = d
                    d.begin()
                    // A second activate pass after the window appears — sometimes
                    // the first call fires before AppKit has a window to foreground.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        NSApplication.shared.activate(ignoringOtherApps: true)
                    }
                }
        }
        .windowResizability(.contentSize)
    }
}
