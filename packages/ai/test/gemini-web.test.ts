import { describe, expect, it } from "vitest";
import { geminiCookieAuth } from "../src/auth/gemini-cookie.ts";
import { geminiWebProvider } from "../src/providers/gemini-web.ts";

describe("Gemini Web provider", () => {
	it("exposes the ny-gemini-acp web model catalog", () => {
		const provider = geminiWebProvider();
		expect(provider.id).toBe("gemini-web");
		expect(provider.getModels().map((model) => model.id)).toEqual([
			"gemini-3.6-flash",
			"gemini-3.5-flash",
			"gemini-3.5-flash-thinking",
			"gemini-3.1-pro",
			"gemini-3.1-pro-enhanced",
			"gemini-auto",
			"gemini-3.5-flash-thinking-lite",
			"gemini-flash-lite",
		]);
	});

	it("resolves GEMINI_COOKIE_FILE without reading cookie contents", async () => {
		const auth = geminiCookieAuth();
		const result = await auth.resolve?.({
			credential: undefined,
			signal: new AbortController().signal,
			ctx: {
				env: async (name) => (name === "GEMINI_COOKIE_FILE" ? "/tmp/test-cookie.json" : undefined),
				fileExists: async (path) => path === "/tmp/test-cookie.json",
			},
		});

		expect(result).toEqual({
			auth: { apiKey: "/tmp/test-cookie.json" },
			source: "/tmp/test-cookie.json",
		});
	});
});
