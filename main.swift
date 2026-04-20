// main.swift — executable entry point.
//
// All the declarations + functions live in bootstrap.swift (formerly the
// contents of this file). This split lets libgemma_metal.dylib compile
// the entire codebase without tripping over top-level statements: the
// dylib build simply omits main.swift, so the only top-level statement
// in the project stays out of the library image.

bootstrapGlobalState()
runEnvDrivenDemos()
