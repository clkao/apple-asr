// apple-asr-shim — Apple SpeechAnalyzer streaming shim (macOS 26+)
//
// The native engine behind the `apple_asr` Python package. Speaks transport
// protocol v1 (see SPEC.md §5): raw float32 LE 16 kHz mono PCM on stdin,
// JSONL events on stdout, human diagnostics on stderr, and a JSONL command
// channel on a control fd (default 3, overridable via APPLE_ASR_CTL_FD).
//
// Modes:
//   --stdin              raw float32 16k mono PCM on stdin (the mode the
//                        Python package uses); EOF finishes the session
//   --file <path>        batch: the shim reads the file itself
//   --mic                self-captured microphone (needs mic TCC)
// Options:
//   --locale <id>        BCP-47 locale, default en-US
//   --context "a,b,c"    AnalysisContext contextual strings (hotwords)
//   --preset <p>         default|progressive|timeIndexedProgressive|transcription
//   --pause-commit <s>   quiet seconds before a pause commits (default 0.08)
//   --commit-interval <s>  commit-latency ceiling; 0 = pause-only (default 0)
//   --vad-sensitivity <l>  off|low|medium|high (inert in the prototype)
//   --no-fast            drop the .fastResults reporting option (adds accuracy)
//   --fast               DEPRECATED no-op, accepted for CLI compatibility only
//   --no-volatile        disable volatile (partial) results
//   --no-confidence      omit per-run transcriptionConfidence
//   --list-locales       print {"type":"locales",...} and exit 0
//   --ensure-installed <id>  install the locale asset or exit nonzero
//
// First stdout line is always `hello` (before any audio is accepted).
//
// Build:  swiftc -O -parse-as-library speechanalyzer.swift -o apple-asr-shim

@preconcurrency import AVFoundation
import Foundation
import Speech

let shimVersion = "0.1.0"
let protocolVersion = 1

// ---------------------------------------------------------------------------
// Process-wide state
// ---------------------------------------------------------------------------

final class ShimState {
    static let shared = ShimState()
    private let lock = NSLock()
    private var _hello = false
    private var _closed = false
    private var _finished = false
    private var _pendingReason = "pause"
    private let started = Date()

    var hello: Bool { lock.lock(); defer { lock.unlock() }; return _hello }
    func setHello() { lock.lock(); _hello = true; lock.unlock() }

    var closed: Bool { lock.lock(); defer { lock.unlock() }; return _closed }
    func setClosed() { lock.lock(); _closed = true; lock.unlock() }

    /// Returns true exactly once — guards `finalizeAndFinishThroughEndOfInput`.
    func markFinished() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if _finished { return false }
        _finished = true
        return true
    }

    func setReason(_ r: String) { lock.lock(); _pendingReason = r; lock.unlock() }

    /// The reason attached to the next final. Sticky (not consumed) because one
    /// `finalize` can drain several finals; the default covers finals the
    /// framework's own endpointer produces before any explicit finalize.
    func reason() -> String {
        lock.lock(); defer { lock.unlock() }
        return _pendingReason
    }

    func wall() -> Double { Date().timeIntervalSince(started) }
}

let emitLock = NSLock()

/// POSIX write loop: survives a closed reader (EPIPE) instead of raising an
/// NSFileHandleOperationException. Never throws.
func writeAll(_ fd: Int32, _ bytes: [UInt8]) {
    var off = 0
    bytes.withUnsafeBufferPointer { buf in
        while off < buf.count {
            let n = write(fd, buf.baseAddress! + off, buf.count - off)
            if n <= 0 { return }
            off += n
        }
    }
}

/// One JSON object per line on stdout, flushed per event. The lock keeps
/// concurrent emitters (consumer Task + control channel) from interleaving.
func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
          let s = String(data: data, encoding: .utf8) else { return }
    let bytes = Array((s + "\n").utf8)
    emitLock.lock()
    writeAll(1, bytes)
    emitLock.unlock()
}

func logStderr(_ s: String) {
    writeAll(2, Array((s + "\n").utf8))
}

/// Fatal: diagnostics on stderr, an `error` event if `hello` already went out
/// (hello must remain the first stdout line), then a nonzero exit.
func fatal(_ msg: String) -> Never {
    logStderr("ERROR: " + msg)
    if ShimState.shared.hello {
        emit(["type": "error", "message": msg, "detail": ""])
    }
    exit(1)
}

func commonFormatName(_ f: AVAudioCommonFormat) -> String {
    switch f {
    case .pcmFormatInt16: return "int16"
    case .pcmFormatInt32: return "int32"
    case .pcmFormatFloat32: return "float32"
    case .pcmFormatFloat64: return "float64"
    default: return "other"
    }
}

// ---------------------------------------------------------------------------
// Monotonic analyzer-domain frame counter (shared by feeder + controller)
// ---------------------------------------------------------------------------

final class FrameCounter {
    private let lock = NSLock()
    private var v: Int64 = 0
    func advance(_ n: Int) { lock.lock(); v += Int64(n); lock.unlock() }
    func set(_ n: Int64) { lock.lock(); v = n; lock.unlock() }
    var value: Int64 { lock.lock(); defer { lock.unlock() }; return v }
}

// ---------------------------------------------------------------------------
// Pause-driven commit
// ---------------------------------------------------------------------------

/// The transcriber's own endpointer only fires on long silences, so
/// continuous speech lands as one giant final — and when a caller's VAD strips
/// silence entirely the endpointer never sees a pause at all. When the fed
/// audio goes quiet for >= `seconds`, finalize results through the start of
/// that quiet span: the module commits its current hypothesis as a real final
/// (with word runs + punctuation) while the session keeps analyzing.
/// `finalize(through:)` is the non-terminating variant.
final class PauseCommitter {
    private let analyzer: SpeechAnalyzer
    let rate: Double
    private let seconds: Double
    private let intervalSeconds: Double
    private let quietPeak: Float
    private var quietFrames = 0
    private var quietStartFrame: Int64 = 0
    private var fired = false
    private var heardSpeech = false
    private var framesSinceCommit = 0
    private var speechSinceCommit = false

    init?(analyzer: SpeechAnalyzer, rate: Double, seconds: Double,
          intervalSeconds: Double = 0, quietPeak: Float = 1e-4) {
        guard seconds > 0 || intervalSeconds > 0, rate > 0 else { return nil }
        self.analyzer = analyzer
        self.rate = rate
        self.seconds = seconds
        self.intervalSeconds = intervalSeconds
        self.quietPeak = quietPeak
    }

    private func commit(through frame: Int64, reason: String) {
        framesSinceCommit = 0
        speechSinceCommit = false
        let secs = Double(frame) / rate
        let cm = CMTime(value: frame, timescale: Int32(rate))
        ShimState.shared.setReason(reason)
        emit(["type": "commit", "through": secs, "reason": reason,
              "wall": ShimState.shared.wall()])
        logStderr("commit[\(reason)]: finalize(through: \(secs)s)")
        Task { try? await analyzer.finalize(through: cm) }
    }

    func observe(frames: Int, sliceStartFrame: Int64, peak: Float) {
        guard frames > 0 else { return }
        if peak < quietPeak {
            if quietFrames == 0 { quietStartFrame = sliceStartFrame }
            quietFrames += frames
            if !fired && heardSpeech && seconds > 0 && Double(quietFrames) / rate >= seconds {
                fired = true
                commit(through: quietStartFrame, reason: "pause")
            }
        } else {
            heardSpeech = true
            speechSinceCommit = true
            quietFrames = 0
            fired = false
        }
        // Interval ceiling: a continuous speaker never pauses, so cap the
        // commit latency — the transcriber cuts mid-phrase (default off).
        framesSinceCommit += frames
        if intervalSeconds > 0, speechSinceCommit,
           Double(framesSinceCommit) / rate >= intervalSeconds {
            commit(through: sliceStartFrame + Int64(frames), reason: "interval")
        }
    }
}

// ---------------------------------------------------------------------------
// Control channel (fd 3): prepare / finalize / context / close
// ---------------------------------------------------------------------------

final class Controller {
    let analyzer: SpeechAnalyzer
    let fmt: AVAudioFormat
    let rate: Double
    let counter: FrameCounter
    private let transcriber: SpeechTranscriber

    init(analyzer: SpeechAnalyzer, fmt: AVAudioFormat, counter: FrameCounter,
         transcriber: SpeechTranscriber) {
        self.analyzer = analyzer
        self.fmt = fmt
        self.rate = fmt.sampleRate
        self.counter = counter
        self.transcriber = transcriber
    }

    var cursorSeconds: Double { Double(counter.value) / rate }

    func finalize(through: Double?, reason: String) {
        let t = through ?? cursorSeconds
        ShimState.shared.setReason(reason)
        emit(["type": "commit", "through": t, "reason": reason,
              "wall": ShimState.shared.wall()])
        logStderr("commit[\(reason)]: finalize(through: \(t)s)")
        let cm = CMTime(value: Int64(t * rate), timescale: Int32(rate))
        Task { try? await analyzer.finalize(through: cm) }
    }

    /// Idempotent terminal finalize. Close command and stdin EOF both land here.
    func finishOnce() async {
        guard ShimState.shared.markFinished() else { return }
        ShimState.shared.setReason("eof")
        emit(["type": "commit", "through": cursorSeconds, "reason": "eof",
              "wall": ShimState.shared.wall()])
        try? await analyzer.finalizeAndFinishThroughEndOfInput()
    }

    func handle(_ obj: [String: Any]) {
        guard let cmd = obj["cmd"] as? String else {
            logStderr("control: command without \"cmd\" ignored")
            return
        }
        switch cmd {
        case "prepare":
            Task { [analyzer, fmt] in
                do { try await analyzer.prepareToAnalyze(in: fmt) }
                catch { logStderr("control: prepareToAnalyze failed (non-fatal): \(error)") }
            }
        case "finalize":
            let through = obj["through"] as? Double
            finalize(through: through, reason: "flush")
        case "context":
            let strings = (obj["strings"] as? [String]) ?? []
            let context = AnalysisContext()
            context.contextualStrings[.general] = strings
            Task { [analyzer] in
                do { try await analyzer.setContext(context) }
                catch { logStderr("control: setContext failed (non-fatal): \(error)") }
            }
        case "close":
            ShimState.shared.setClosed()
            logStderr("control: close requested")
            Task { await self.finishOnce() }
        default:
            logStderr("control: unknown cmd \(cmd)")
        }
    }
}

func controlFD() -> Int32? {
    var fd: Int32 = 3
    if let s = ProcessInfo.processInfo.environment["APPLE_ASR_CTL_FD"], let v = Int32(s) {
        fd = v
    }
    if fcntl(fd, F_GETFD) == -1 { return nil }
    return fd
}

func startControlChannel(_ controller: Controller, fd: Int32) {
    Thread.detachNewThread {
        logStderr("control: channel on fd \(fd)")
        var buf = [UInt8](repeating: 0, count: 4096)
        var pending = Data()
        while true {
            let n = read(fd, &buf, buf.count)
            if n <= 0 { break }
            pending.append(contentsOf: buf[0..<n])
            while let idx = pending.firstIndex(of: 0x0A) {
                let lineData = pending.subdata(in: 0..<idx)
                pending.removeSubrange(0...idx)
                guard let s = String(data: lineData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty else { continue }
                guard let d = s.data(using: .utf8),
                      let obj = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else {
                    logStderr("control: unparseable line ignored")
                    continue
                }
                controller.handle(obj)
            }
        }
        logStderr("control: channel closed")
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

@main
struct AppleAsrShim {
    static func main() async {
        var localeId = "en-US"
        var contextStrings: [String] = []
        var filePath: String? = nil
        var fast = false
        var volatile = true
        var fastResults = true
        var confidence = true
        var stdinMode = false
        var pauseCommit = 0.08
        var commitInterval = 0.0
        var vadSensitivity = "off"
        var preset = "progressive"
        var listLocales = false
        var ensureInstalled: String? = nil

        var args = Array(CommandLine.arguments.dropFirst())
        while !args.isEmpty {
            let a = args.removeFirst()
            switch a {
            case "--locale":
                if args.isEmpty { fatal("--locale needs a value") }
                localeId = args.removeFirst()
            case "--context":
                if args.isEmpty { fatal("--context needs a value") }
                let v = args.removeFirst()
                contextStrings = v.split(separator: ",")
                    .map { $0.trimmingCharacters(in: CharacterSet.whitespaces) }
                    .filter { !$0.isEmpty }
            case "--file":
                if args.isEmpty { fatal("--file needs a path") }
                filePath = args.removeFirst()
            case "--fast": fast = true
            case "--no-fast": fastResults = false
            case "--no-confidence": confidence = false
            case "--no-volatile": volatile = false
            case "--stdin": stdinMode = true
            case "--pause-commit":
                if args.isEmpty { fatal("--pause-commit needs seconds (0 disables)") }
                pauseCommit = Double(args.removeFirst()) ?? 0.08
            case "--commit-interval":
                if args.isEmpty { fatal("--commit-interval needs seconds (0 disables)") }
                commitInterval = Double(args.removeFirst()) ?? 0.0
            case "--vad-sensitivity":
                if args.isEmpty { fatal("--vad-sensitivity needs off|low|medium|high") }
                vadSensitivity = args.removeFirst()
            case "--preset":
                if args.isEmpty { fatal("--preset needs progressive|transcription|timeIndexedProgressive") }
                preset = args.removeFirst()
            case "--list-locales":
                listLocales = true
            case "--ensure-installed":
                if args.isEmpty { fatal("--ensure-installed needs a locale") }
                ensureInstalled = args.removeFirst()
            case "--help", "-h":
                print("usage: apple-asr-shim {--stdin|--file PATH|--mic} "
                    + "[--locale L] [--preset P] [--context a,b,c] "
                    + "[--pause-commit S] [--commit-interval S] "
                    + "[--vad-sensitivity off|low|medium|high] "
                    + "[--no-fast] [--no-volatile] [--no-confidence]")
                print("       apple-asr-shim --list-locales")
                print("       apple-asr-shim --ensure-installed LOCALE")
                print("")
                print("  --fast is deprecated and ignored (fastResults is on unless")
                print("  --no-fast is given). --help does not build anything: the")
                print("  console script prints its own flag list without a build.")
                return
            default: fatal("unknown arg: \(a)")
            }
        }

        // Deprecated no-op (SPEC.md §4 lists it): the analyzer consumes as fast as it can.
        _ = fast

        guard #available(macOS 26.0, *) else {
            fatal("requires macOS 26+ (Apple SpeechAnalyzer); this OS is older")
        }

        if listLocales {
            let supported = await SpeechTranscriber.supportedLocales.map { $0.identifier(.bcp47) }.sorted()
            let installed = await SpeechTranscriber.installedLocales.map { $0.identifier(.bcp47) }.sorted()
            emit(["type": "locales", "supported": supported, "installed": installed])
            return
        }

        if let wanted = ensureInstalled {
            guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: wanted)) else {
                fatal("locale \(wanted) not supported by this system")
            }
            let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
            let installed = await SpeechTranscriber.installedLocales
            if !installed.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) {
                logStderr("downloading locale asset for \(locale.identifier(.bcp47))...")
                do {
                    if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                        try await req.downloadAndInstall()
                    }
                } catch {
                    fatal("locale asset install failed for \(wanted): \(error)")
                }
            }
            logStderr("ensure-installed: \(locale.identifier(.bcp47)) available")
            return
        }

        let wanted = Locale(identifier: localeId)
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: wanted) else {
            let supported = await SpeechTranscriber.supportedLocales
                .map { $0.identifier(.bcp47) }.joined(separator: ", ")
            fatal("locale \(localeId) not supported. supportedLocales: \(supported)")
        }

        // Preset chooses the endpointing/volatile cadence.
        // .progressiveTranscription streams volatiles incrementally.
        let presetMap: [String: SpeechTranscriber.Preset] = [
            "default": .transcription,
            "transcription": .transcription,
            "progressive": .progressiveTranscription,
            "timeIndexedProgressive": .timeIndexedProgressiveTranscription,
        ]
        guard var chosenPreset = presetMap[preset] else {
            fatal("--preset must be progressive|transcription|timeIndexedProgressive, got \(preset)")
        }
        if volatile && fastResults {
            chosenPreset.reportingOptions.insert(.fastResults)
        }
        // Word-level audioTimeRange (and confidence, when requested) must be in
        // attributeOptions or the results carry no per-run timings at all.
        chosenPreset.attributeOptions.insert(.audioTimeRange)
        if confidence { chosenPreset.attributeOptions.insert(.transcriptionConfidence) }
        let transcriber = SpeechTranscriber(locale: locale, preset: chosenPreset)

        // Optional SpeechDetector module (documented inert in the prototype).
        var modules: [any SpeechModule] = [transcriber]
        var detector: SpeechDetector? = nil
        let sensMap: [String: SpeechDetector.SensitivityLevel] = ["low": .low, "medium": .medium, "high": .high]
        if let level = sensMap[vadSensitivity.lowercased()] {
            detector = SpeechDetector(detectionOptions: .init(sensitivityLevel: level), reportResults: true)
            modules.append(detector!)
        } else if vadSensitivity.lowercased() != "off" {
            fatal("--vad-sensitivity must be off|low|medium|high, got \(vadSensitivity)")
        }

        guard let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: modules) else {
            fatal("no analyzer audio format available")
        }

        // hello — the first stdout line, before any audio is accepted.
        var caps = ["word_runs", "pause_commit", "flush", "context"]
        if volatile { caps.insert("volatile", at: 0) }
        var opts: [String] = []
        if volatile { opts.append("volatileResults") }
        if volatile && fastResults { opts.append("fastResults") }
        emit([
            "type": "hello",
            "protocol": protocolVersion,
            "shim_version": shimVersion,
            "locale": locale.identifier(.bcp47),
            "preset": preset,
            "format": [
                "sample_rate": analyzerFormat.sampleRate,
                "channels": Int(analyzerFormat.channelCount),
                "common_format": commonFormatName(analyzerFormat.commonFormat),
            ],
            "capabilities": caps,
            "reporting_options": opts,
        ])
        ShimState.shared.setHello()

        // One-time locale asset install (OS-managed storage).
        let installed = await SpeechTranscriber.installedLocales
        if !installed.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) {
            logStderr("downloading locale asset for \(locale.identifier(.bcp47))...")
            do {
                if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                    try await req.downloadAndInstall()
                }
            } catch { fatal("locale asset install failed: \(error)") }
        }

        let analyzer = SpeechAnalyzer(modules: modules)
        if !contextStrings.isEmpty {
            let context = AnalysisContext()
            context.contextualStrings[.general] = contextStrings
            do { try await analyzer.setContext(context) }
            catch { logStderr("setContext failed (non-fatal): \(error)") }
        }

        let fmtDesc = "\(analyzerFormat.sampleRate)Hz \(analyzerFormat.channelCount)ch \(commonFormatName(analyzerFormat.commonFormat))"
        logStderr("format=\(fmtDesc) vad=\(vadSensitivity) preset=\(preset) fast=\(volatile && fastResults)")

        // Warm-start the model for this audio format.
        do { try await analyzer.prepareToAnalyze(in: analyzerFormat) }
        catch { logStderr("prepareToAnalyze failed (non-fatal): \(error)") }

        // Ctrl-C -> graceful finalize via a signal-driven stream.
        let (interrupted, intCont) = AsyncStream<Bool>.makeStream()
        signal(SIGINT, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        src.setEventHandler { intCont.yield(true) }
        src.resume()

        // Results consumer: emits partial/final events with word runs.
        let consumer = Task {
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                guard !text.isEmpty else { continue }
                var runs: [[Any]] = []
                for run in result.text.runs {
                    if let rr = run.audioTimeRange {
                        var entry: [Any] = [String(result.text.characters[run.range]),
                                            rr.start.seconds, rr.end.seconds]
                        // Optional 4th element: transcriptionConfidence when requested.
                        if confidence, let c = run.transcriptionConfidence {
                            entry.append(c)
                        }
                        runs.append(entry)
                    }
                }
                let range = [result.range.start.seconds, result.range.end.seconds]
                if result.isFinal {
                    emit(["type": "final", "text": text, "range": range,
                          "runs": runs, "reason": ShimState.shared.reason()])
                } else {
                    emit(["type": "partial", "text": text, "range": range, "runs": runs])
                }
            }
        }

        // VAD consumer (only when --vad-sensitivity != off). Diagnostics go to
        // stderr, not the wire — the wire carries only protocol v1 events.
        let vadConsumer = detector.map { d in
            Task {
                logStderr("vad-consumer: started")
                var n = 0
                for try await r in d.results {
                    n += 1
                    if n <= 3 {
                        logStderr("vad#\(n): \(r.speechDetected ? "speech" : "silence") @ "
                            + "[\(r.range.start.seconds),\(r.range.end.seconds)]")
                    }
                }
                logStderr("vad-consumer: stream ended after \(n) events")
            }
        }

        // File mode: the analyzer reads + converts the file itself.
        var endedReason = "eof"
        do {
            if let path = filePath {
                let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
                _ = try await analyzer.analyzeSequence(from: file)
                ShimState.shared.setReason("eof")
                if ShimState.shared.markFinished() {
                    try await analyzer.finalizeAndFinishThroughEndOfInput()
                }
            } else if stdinMode {
                // Raw float32 16k mono on stdin; converted to the analyzer's
                // format (same rate — no resampler), yielded with contiguous
                // analyzer-domain timestamps. stdin EOF finishes the session.
                let counter = FrameCounter()
                let committer = PauseCommitter(analyzer: analyzer, rate: analyzerFormat.sampleRate,
                                               seconds: pauseCommit, intervalSeconds: commitInterval)
                let controller = Controller(analyzer: analyzer, fmt: analyzerFormat,
                                            counter: counter, transcriber: transcriber)
                if let fd = controlFD() { startControlChannel(controller, fd: fd) }
                let (inputs, inCont) = AsyncStream<AnalyzerInput>.makeStream()
                let feedTask = Task {
                    do { _ = try await analyzer.analyzeSequence(inputs) }
                    catch { logStderr("analyze error: \(error)") }
                }
                let feeder = Task.detached {
                    let chunkFrames = 800  // 50ms @ 16k (finer pause measurement)
                    var pending = [Float]()
                    var readBuf = [Float](repeating: 0, count: chunkFrames)
                    func yieldSlice(_ slice: ArraySlice<Float>) {
                        let n = slice.count
                        guard n > 0,
                              let buf = AVAudioPCMBuffer(pcmFormat: analyzerFormat,
                                                         frameCapacity: AVAudioFrameCount(n)) else { return }
                        var peak: Float = 0
                        if let dst = buf.int16ChannelData {
                            var i = 0
                            for v in slice {
                                dst[0][i] = Int16(max(-1.0, min(1.0, v)) * 32767.0)
                                peak = max(peak, abs(v))
                                i += 1
                            }
                        }
                        buf.frameLength = AVAudioFrameCount(n)
                        let startFrame = counter.value
                        let cm = CMTime(value: startFrame, timescale: Int32(analyzerFormat.sampleRate))
                        counter.advance(n)
                        inCont.yield(AnalyzerInput(buffer: buf, bufferStartTime: cm))
                        committer?.observe(frames: n, sliceStartFrame: startFrame, peak: peak)
                    }
                    while true {
                        let got = fread(&readBuf, MemoryLayout<Float>.size, chunkFrames, stdin)
                        if got == 0 { break }
                        pending.append(contentsOf: readBuf[0..<got])
                        while pending.count >= chunkFrames {
                            yieldSlice(pending.prefix(chunkFrames))
                            pending.removeFirst(chunkFrames)
                        }
                    }
                    if !pending.isEmpty { yieldSlice(pending[...]) }
                    inCont.finish()
                }
                _ = await feeder.value
                _ = await feedTask.value
                await controller.finishOnce()
            } else {
                let (inputs, inCont) = AsyncStream<AnalyzerInput>.makeStream()
                let feedTask = Task {
                    do { _ = try await analyzer.analyzeSequence(inputs) }
                    catch { logStderr("analyze error: \(error)") }
                }
                let engine = try await startMic(inCont, analyzerFormat: analyzerFormat)
                logStderr("mic live — speak, Ctrl-C to stop")
                _ = await interrupted.first { _ in true }
                inCont.finish()
                _ = await feedTask.value
                await analyzer.cancelAndFinishNow()
                engine.stop()
            }
        } catch {
            logStderr("feed error: \(error)")
            emit(["type": "error", "message": "feed error", "detail": "\(error)"])
            await analyzer.cancelAndFinishNow()
            _ = await consumer.result
            if let vc = vadConsumer { _ = await vc.result }
            logStderr("done (error)")
            exit(1)
        }

        _ = await consumer.result
        if let vc = vadConsumer { _ = await vc.result }
        if ShimState.shared.closed { endedReason = "closed" }
        emit(["type": "ended", "reason": endedReason])
        logStderr("done")
    }

    // ------------------------------------------------------------------
    static func startMic(_ cont: AsyncStream<AnalyzerInput>.Continuation,
                         analyzerFormat: AVAudioFormat) async throws -> AVAudioEngine {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: hwFormat, to: analyzerFormat) else {
            fatal("cannot convert hw \(hwFormat) -> \(analyzerFormat)")  // Never
        }
        let ratio = Double(analyzerFormat.sampleRate) / Double(hwFormat.sampleRate)
        // Timestamps must be contiguous + monotonic in the ANALYZER's frame
        // domain; a running counter makes each [start, start+frameLength] abut
        // the next with no gap or overlap.
        let counter = FrameCounter()
        var tapCount = 0
        inputNode.installTap(onBus: 0, bufferSize: 4000, format: hwFormat) { buffer, _ in
            let cap = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 64)
            guard let converted = AVAudioPCMBuffer(pcmFormat: analyzerFormat, frameCapacity: cap) else { return }
            var err: NSError? = nil
            _ = converter.convert(to: converted, error: &err) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if let e = err, tapCount == 0 { logStderr("tap convert error: \(e)") }
            guard err == nil, converted.frameLength > 0 else { return }
            tapCount += 1
            if tapCount == 1 {
                logStderr("tap0: \(buffer.frameLength)fr @\(hwFormat.sampleRate)Hz -> "
                    + "\(converted.frameLength)fr @\(analyzerFormat.sampleRate)Hz")
            }
            let start = counter.value
            let cm = CMTime(value: start, timescale: Int32(analyzerFormat.sampleRate))
            counter.advance(Int(converted.frameLength))
            cont.yield(AnalyzerInput(buffer: converted, bufferStartTime: cm))
        }
        engine.prepare()
        do { try engine.start() }
        catch {
            logStderr("engine.start failed: \(error)")
            throw error
        }
        logStderr("engine started; hw=\(hwFormat.sampleRate)Hz analyzer=\(analyzerFormat.sampleRate)Hz")
        return engine
    }
}
