import { geminiWebApi } from "../api/gemini-web.lazy.ts";
import { geminiCookieAuth } from "../auth/gemini-cookie.ts";
import { createProvider, type Provider } from "../models.ts";
import { GEMINI_WEB_MODEL_LIST } from "./gemini-web.models.ts";

export type { GeminiWebOptions } from "../api/gemini-web.ts";

export function geminiWebProvider(): Provider<"gemini-web"> {
	return createProvider({
		id: "gemini-web",
		name: "Gemini Web",
		baseUrl: "https://gemini.google.com",
		auth: { apiKey: geminiCookieAuth() },
		models: GEMINI_WEB_MODEL_LIST,
		api: geminiWebApi(),
	});
}
