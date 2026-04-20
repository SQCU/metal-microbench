# After the 2026-04-18 refactor, forward_graph is multi-file. `main.swift`
# holds top-level script entry (Swift requires that name for file-scope
# executable statements); other files are library-style (declarations only).
# bootstrap.swift contains the former main.swift's declarations + helpers +
# runEnvDrivenDemos() function. main.swift is a tiny top-level entry that only
# the executable target builds; the dylib target omits it.
FORWARD_GRAPH_LIB_SRCS = bootstrap.swift common.swift kernels.swift vision_tower.swift vision_residency.swift harness.swift tokenizer.swift lm_session.swift lm_engine.swift page_manager.swift kv_visualizer.swift
FORWARD_GRAPH_SRCS = $(FORWARD_GRAPH_LIB_SRCS) main.swift

all: mem_mountain tile_gemm paged_attention moe_matmul dense_gemv forward_ops forward_graph gguf_loader

mem_mountain: mem_mountain.swift
	swiftc -O mem_mountain.swift -o mem_mountain -framework Metal -framework Foundation

tile_gemm: tile_gemm.swift
	swiftc -O tile_gemm.swift -o tile_gemm -framework Metal -framework Foundation

paged_attention: paged_attention.swift
	swiftc -O paged_attention.swift -o paged_attention -framework Metal -framework Foundation

moe_matmul: moe_matmul.swift
	swiftc -O moe_matmul.swift -o moe_matmul -framework Metal -framework Foundation

dense_gemv: dense_gemv.swift
	swiftc -O dense_gemv.swift -o dense_gemv -framework Metal -framework Foundation

forward_ops: forward_ops.swift
	swiftc -O forward_ops.swift -o forward_ops -framework Metal -framework Foundation

forward_graph: $(FORWARD_GRAPH_SRCS)
	swiftc -O $(FORWARD_GRAPH_SRCS) -o forward_graph -framework Metal -framework Foundation

# libgemma_metal.dylib — C-ABI shim for the Python bridge. Same sources as
# forward_graph but omits main.swift (which contains the only top-level
# statement — runEnvDrivenDemos()) and adds ffi.swift with @_cdecl exports.
libgemma_metal.dylib: $(FORWARD_GRAPH_LIB_SRCS) ffi.swift
	swiftc -O -emit-library $(FORWARD_GRAPH_LIB_SRCS) ffi.swift \
	    -o libgemma_metal.dylib \
	    -framework Metal -framework Foundation

gguf_loader: gguf_loader.swift gguf_tool.swift
	swiftc -O gguf_loader.swift gguf_tool.swift -o gguf_loader -framework Metal -framework Foundation

# tetraplex_recorder — stubbish SwiftUI app that runs the 4-stream demo in
# one process and writes tetraplex-demo.mp4 via AVAssetWriter. No HTTP, no
# browser, no external screen capture. Uses the same engine sources as
# forward_graph minus main.swift (the recorder brings its own @main via
# SwiftUI.App + -parse-as-library).
tetraplex_recorder: $(FORWARD_GRAPH_LIB_SRCS) tetraplex_recorder.swift
	swiftc -O -parse-as-library \
	    $(FORWARD_GRAPH_LIB_SRCS) tetraplex_recorder.swift \
	    -o tetraplex_recorder \
	    -framework Metal -framework Foundation \
	    -framework SwiftUI -framework AppKit \
	    -framework AVFoundation -framework CoreMedia -framework CoreVideo

clean:
	rm -f mem_mountain tile_gemm paged_attention moe_matmul dense_gemv forward_ops forward_graph gguf_loader
