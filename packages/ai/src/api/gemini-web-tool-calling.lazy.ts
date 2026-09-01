import type { ProviderStreams } from "../types.ts";
import { lazyApi } from "./lazy.ts";

/**
 * Browser-safe provider entrypoint: keep the Node-only Gemini Web transport
 * outside eager bundles while preserving the synchronous ProviderStreams API.
 */
export const geminiWebToolCallingApi = (): ProviderStreams =>
	lazyApi(() => import("./gemini-web-tool-calling.ts").then((module) => module.geminiWebToolCallingApi()));
