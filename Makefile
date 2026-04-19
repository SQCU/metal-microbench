# After the 2026-04-18 refactor, forward_graph is multi-file. `main.swift`
# holds top-level script entry (Swift requires that name for file-scope
# executable statements); other files are library-style (declarations only).
FORWARD_GRAPH_SRCS = main.swift common.swift kernels.swift vision_tower.swift harness.swift tokenizer.swift lm_session.swift lm_engine.swift

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

gguf_loader: gguf_loader.swift gguf_tool.swift
	swiftc -O gguf_loader.swift gguf_tool.swift -o gguf_loader -framework Metal -framework Foundation

clean:
	rm -f mem_mountain tile_gemm paged_attention moe_matmul dense_gemv forward_ops forward_graph gguf_loader
