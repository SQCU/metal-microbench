// CLI driver for profile_vision_tower.
// Usage: profile_vision_tower <image_path> [safetensors_path]
import Foundation

@main
struct ProfileVisionTowerMain {
    static func main() {
        print("entered main")
        fflush(stdout)
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("usage: \(args[0]) <image_path> [safetensors_path]")
            exit(2)
        }
        print("about to bootstrap")
        fflush(stdout)
        bootstrapGlobalState()
        print("bootstrapped")
        fflush(stdout)
        let imagePath = args[1]
        let safetensorsPath = args.count >= 3
            ? args[2]
            : (ProcessInfo.processInfo.environment["GEMMA_SAFETENSORS"]
                ?? "/Users/mdot/models/gemma-4-a4b-bf16/model-00001-of-00002.safetensors")
        runVisionProfileMain(imagePath: imagePath, safetensorsPath: safetensorsPath)
    }
}
