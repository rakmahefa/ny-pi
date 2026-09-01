import type { ProviderStreams } from "../types.ts";
import { lazyApi } from "./lazy.ts";

export const geminiWebApi = (): ProviderStreams => lazyApi(() => import("./gemini-web.ts"));
