import type { ApiKeyAuth } from "./types.ts";

const DEFAULT_COOKIE_FILES = ["./cookie.json", "./vendor/cookie.json"] as const;

async function resolveCookieFile(ctx: Parameters<NonNullable<ApiKeyAuth["resolve"]>>[0]["ctx"]): Promise<string | undefined> {
	const configured = await ctx.env("GEMINI_COOKIE_FILE");
	if (configured?.trim()) {
		return (await ctx.fileExists(configured.trim())) ? configured.trim() : undefined;
	}
	for (const candidate of DEFAULT_COOKIE_FILES) {
		if (await ctx.fileExists(candidate)) return candidate;
	}
	return undefined;
}

/** Ambient auth for the Gemini Web provider. The api-key field carries a cookie-file path, never cookie contents. */
export function geminiCookieAuth(): ApiKeyAuth {
	return {
		name: "Gemini cookie.json",
		login: async (interaction) => {
			interaction.signal.throwIfAborted();
			const path = await interaction.prompt({
				type: "text",
				message: "Path to Gemini cookie.json",
				placeholder: "./cookie.json",
			});
			interaction.signal.throwIfAborted();
			if (!path.trim()) throw new Error("Gemini cookie file path cannot be empty");
			return { type: "api_key", key: path.trim() };
		},
		check: async ({ ctx, credential, signal }) => {
			signal.throwIfAborted();
			if (credential?.key && (await ctx.fileExists(credential.key))) {
				return { type: "api_key", source: credential.key };
			}
			const path = await resolveCookieFile(ctx);
			return path ? { type: "api_key", source: path } : undefined;
		},
		resolve: async ({ ctx, credential, signal }) => {
			signal.throwIfAborted();
			if (credential?.key && (await ctx.fileExists(credential.key))) {
				return { auth: { apiKey: credential.key }, source: credential.key };
			}
			const path = await resolveCookieFile(ctx);
			return path ? { auth: { apiKey: path }, source: path } : undefined;
		},
	};
}
