import { createHash, randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import type {
	Api,
	AssistantMessage,
	Context,
	Model,
	SimpleStreamOptions,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	ToolCall,
} from "../types.ts";
import { AssistantMessageEventStream } from "../utils/event-stream.ts";
import { formatProviderError } from "../utils/error-body.ts";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.ts";

export interface GeminiWebOptions extends StreamOptions {
	/** Optional user index used by Google accounts with multiple profiles. */
	authUser?: number;
	/** Gemini Web backend revision, matching ny-gemini-acp's `bl` parameter. */
	bl?: string;
	/** Optional thinking budget override (0..4) used by the Gemini Web payload. */
	think?: number;
}

const ENDPOINT = "_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate";
const DEFAULT_BL = "boq_assistant-bard-web-server_20260716.08_p0";
const TOKEN_TTL_MS = 10 * 60_000;
const MAX_FRAME_BYTES = 8 * 1024 * 1024;
const MAX_METADATA_BYTES = 64 * 1024;
const MAX_TOOL_ARGUMENT_BYTES = 1024 * 1024;

interface ParsedCookie {
	name: string;
	value: string;
	domain?: string;
	expiresAt?: number;
}

interface CookieJar {
	cookies: ParsedCookie[];
	sapisid?: string;
}

interface PageTokens {
	at?: string;
	pushId?: string;
	pctx?: string;
	fetchedAt: number;
}

let pageTokens: PageTokens | undefined;
let requestCounter = 0;
let toolCounter = 0;

function nowSeconds(): number {
	return Math.floor(Date.now() / 1000);
}

function isGoogleDomain(domain?: string): boolean {
	return !domain || domain === "google.com" || domain.endsWith(".google.com");
}

function parseCookieFile(raw: string): CookieJar {
	const trimmed = raw.trim();
	if (!trimmed) return { cookies: [] };

	if (trimmed.startsWith("[")) {
		const parsed = JSON.parse(trimmed) as Array<Record<string, unknown>>;
		return {
			cookies: parsed.map((cookie) => ({
				name: String(cookie.name ?? ""),
				value: String(cookie.value ?? ""),
				domain: cookie.domain ? String(cookie.domain) : undefined,
				expiresAt:
					cookie.expirationDate !== undefined ? Number(cookie.expirationDate) : undefined,
			})),
		};
	}

	if (trimmed.startsWith("{")) {
		const parsed = JSON.parse(trimmed) as { cookie?: string; sapisid?: string };
		return {
			cookies: parseRawCookies(parsed.cookie ?? ""),
			sapisid: parsed.sapisid,
		};
	}

	return { cookies: parseRawCookies(trimmed) };
}

function parseRawCookies(raw: string): ParsedCookie[] {
	return raw.split(";").flatMap((part) => {
		const pair = part.trim().split("=");
		if (pair.length < 2) return [];
		return [{ name: pair.shift()?.trim() ?? "", value: pair.join("=").trim() }];
	});
}

function cookieHeader(jar: CookieJar): string | undefined {
	const now = Date.now() / 1000;
	const values = jar.cookies
		.filter(
			(cookie) =>
				cookie.name &&
				cookie.value &&
				isGoogleDomain(cookie.domain) &&
				(cookie.expiresAt === undefined || cookie.expiresAt > now),
		)
		.map((cookie) => `${cookie.name}=${cookie.value}`);
	return values.length ? values.join("; ") : undefined;
}

function sapisid(jar: CookieJar): string | undefined {
	return jar.sapisid ?? jar.cookies.find((cookie) => cookie.name === "SAPISID")?.value;
}

function sapisidHash(value: string, origin: string): string {
	const timestamp = nowSeconds();
	const digest = createHash("sha1")
		.update(`${timestamp} ${value} ${origin}`, "utf8")
		.digest("hex");
	return `SAPISIDHASH ${timestamp}_${digest}`;
}

function encodeFormComponent(value: string): string {
	return encodeURIComponent(value).replace(/%20/g, "+");
}

function formEncode(entries: Record<string, string>): string {
	return Object.entries(entries)
		.map(([key, value]) => `${encodeFormComponent(key)}=${encodeFormComponent(value)}`)
		.join("&");
}

function nextRequestId(): number {
	requestCounter = (requestCounter + 1) % 100_000;
	return (Date.now() % 1_000_000) * 100_000 + requestCounter;
}

function extractField(body: string, key: string): string | undefined {
	const needle = `\"${key}\":\"`;
	const start = body.indexOf(needle);
	if (start < 0) return undefined;
	const from = start + needle.length;
	const end = body.indexOf("\"", from);
	if (end < 0) return undefined;
	const value = body.slice(from, end);
	return value || undefined;
}

async function getPageTokens(
	fetchImpl: typeof globalThis.fetch,
	baseUrl: string,
	authUser: number | undefined,
	cookie: string,
	auth: string | undefined,
	signal: AbortSignal,
): Promise<PageTokens> {
	if (pageTokens && Date.now() - pageTokens.fetchedAt < TOKEN_TTL_MS) return pageTokens;
	const prefix = authUser === undefined ? "" : `/u/${authUser}`;
	const headers: Record<string, string> = {
		cookie,
		"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
		accept: "text/html,application/xhtml+xml",
		referer: `${baseUrl}${prefix}/app`,
	};
	if (auth) headers.authorization = auth;
	const response = await fetchImpl(`${baseUrl}${prefix}/app`, { headers, signal });
	if (!response.ok) throw new Error(`Gemini Web /app returned HTTP ${response.status}`);
	const body = await response.text();
	pageTokens = {
		at: extractField(body, "SNlM0e"),
		pushId: extractField(body, "qKIAYe"),
		pctx: extractField(body, "Ylro7b"),
		fetchedAt: Date.now(),
	};
	return pageTokens;
}

function resolveGeminiModel(modelId: string, requestedThink?: number): {
	mode: number;
	think: number;
	extra?: Array<[number, number]>;
} {
	const normalized = modelId.includes("@think=") ? modelId.slice(0, modelId.indexOf("@think=")) : modelId;
	const overrideRaw = modelId.includes("@think=") ? modelId.slice(modelId.indexOf("@think=") + 7) : undefined;
	const override = requestedThink ?? (overrideRaw ? Number(overrideRaw) : undefined);
	const table: Record<string, { mode: number; think: number; extra?: Array<[number, number]> }> = {
		"gemini-3.6-flash": { mode: 1, think: 4 },
		"gemini-3.5-flash": { mode: 1, think: 4 },
		"gemini-3.5-flash-thinking": { mode: 2, think: 0 },
		"gemini-3.1-pro": { mode: 3, think: 4 },
		"gemini-3.1-pro-enhanced": { mode: 3, think: 4, extra: [[31, 2], [80, 3]] },
		"gemini-auto": { mode: 4, think: 4 },
		"gemini-3.5-flash-thinking-lite": { mode: 5, think: 0 },
		"gemini-flash-lite": { mode: 6, think: 4 },
	};
	const config = table[normalized] ?? table["gemini-3.6-flash"];
	return { ...config, think: Math.max(0, Math.min(4, override ?? config.think)) };
}

function contextPrompt(context: Context): string {
	const lines: string[] = [];
	if (context.systemPrompt?.trim()) lines.push(`[system]\n${context.systemPrompt.trim()}`);
	for (const message of context.messages) {
		if (message.role === "user") {
			const text = typeof message.content === "string" ? message.content : message.content.map((part) => (part.type === "text" ? part.text : "[image]")).join("\n");
			lines.push(`[user]\n${text}`);
		} else if (message.role === "assistant") {
			const text = message.content.filter((part): part is TextContent | ThinkingContent => part.type === "text" || part.type === "thinking").map((part) => (part.type === "text" ? part.text : `<thinking>\n${part.thinking}\n</thinking>`)).join("\n");
			if (text) lines.push(`[assistant]\n${text}`);
		} else if (message.role === "toolResult") {
			const text = message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
			if (text) lines.push(`[tool:${message.toolName}]\n${text}`);
		}
	}
	return lines.join("\n\n");
}

function buildPayload(prompt: string, modelId: string, think: number, at?: string): string {
	const resolved = resolveGeminiModel(modelId, think);
	const inner: unknown[] = Array.from({ length: 102 }, () => null);
	inner[0] = [prompt, 0, null, null, null, null, 0];
	inner[1] = ["en"];
	inner[2] = ["", "", "", null, null, null, null, null, null, ""];
	inner[6] = [0];
	inner[7] = 1;
	inner[10] = 1;
	inner[11] = 0;
	inner[17] = [[resolved.think]];
	inner[18] = 0;
	inner[27] = 1;
	inner[30] = [4];
	inner[41] = [2];
	inner[53] = 0;
	inner[59] = randomUUID();
	inner[61] = [];
	inner[68] = 1;
	inner[79] = resolved.mode;
	for (const [index, value] of resolved.extra ?? []) inner[index] = value;
	return formEncode({
		"f.req": JSON.stringify([null, JSON.stringify(inner)]),
		...(at ? { at } : {}),
	});
}

function parseJsonLine(line: string): unknown | undefined {
	const trimmed = line.trim();
	if (!trimmed || trimmed === ")]}'") return undefined;
	try {
		return JSON.parse(trimmed) as unknown;
	} catch {
		return undefined;
	}
}

function parseInner(value: unknown): unknown | undefined {
	if (!Array.isArray(value)) return undefined;
	const first = Array.isArray(value[0]) ? value[0] : undefined;
	if (!first || first[0] !== "wrb.fr") return undefined;
	const inner = first[2];
	if (typeof inner === "string") {
		try {
			return JSON.parse(inner) as unknown;
		} catch {
			return undefined;
		}
	}
	return inner;
}

function longestCandidateText(inner: unknown): string | undefined {
	if (!Array.isArray(inner) || !Array.isArray(inner[4])) return undefined;
	let longest = "";
	for (const candidate of inner[4]) {
		if (!Array.isArray(candidate) || !Array.isArray(candidate[1])) continue;
		const text = candidate[1].filter((part): part is string => typeof part === "string").join("");
		if (text.length > longest.length) longest = text;
	}
	return longest || undefined;
}

function collectToolCalls(inner: unknown): ToolCall[] {
	const calls: ToolCall[] = [];
	const seen = new Set<string>();
	const walk = (value: unknown): void => {
		if (Array.isArray(value)) {
			for (const child of value) walk(child);
			return;
		}
		if (!value || typeof value !== "object") return;
		const object = value as Record<string, unknown>;
		for (const key of ["functionCall", "function_call", "toolCall", "tool_call", "toolUse", "tool_use"]) {
			const call = object[key];
			if (!call || typeof call !== "object") continue;
			const candidate = call as Record<string, unknown>;
			const name = String(candidate.name ?? candidate.functionName ?? candidate.toolName ?? "").trim();
			const args = candidate.arguments ?? candidate.args ?? candidate.parameters;
			if (!name || args === undefined) continue;
			const id = String(candidate.id ?? candidate.callId ?? candidate.call_id ?? `gemini_call_${toolCounter++}`);
			if (seen.has(id)) continue;
			seen.add(id);
			const serialized = JSON.stringify(args);
			if (serialized.length > MAX_TOOL_ARGUMENT_BYTES) continue;
			calls.push({ type: "toolCall", id, name, arguments: (typeof args === "object" && args !== null ? args : {}) as Record<string, any> });
		}
		for (const child of Object.values(object)) walk(child);
	};
	walk(inner);
	return calls;
}

function metadata(inner: unknown, key: string): unknown | undefined {
	if (!inner || typeof inner !== "object" || Array.isArray(inner)) return undefined;
	const value = (inner as Record<string, unknown>)[key];
	if (value === undefined) return undefined;
	try {
		if (JSON.stringify(value).length <= MAX_METADATA_BYTES) return value;
	} catch {
		return undefined;
	}
	return undefined;
}

function finishReason(inner: unknown): string | undefined {
	const value = metadata(inner, "finishReason");
	return typeof value === "string" ? value : undefined;
}

function usageFrom(inner: unknown): { input: number; output: number; totalTokens: number } | undefined {
	const value = metadata(inner, "usageMetadata") ?? metadata(inner, "usage");
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const object = value as Record<string, unknown>;
	const input = Number(object.promptTokenCount ?? object.inputTokens ?? 0);
	const output = Number(object.candidatesTokenCount ?? object.outputTokens ?? 0);
	const total = Number(object.totalTokenCount ?? input + output);
	return { input, output, totalTokens: total };
}

function deltaFromCumulative(previous: string, next: string): { delta: string; next: string; diverged: boolean } {
	if (!previous) return { delta: next, next, diverged: false };
	if (next === previous) return { delta: "", next: previous, diverged: false };
	if (next.startsWith(previous)) return { delta: next.slice(previous.length), next, diverged: false };
	return { delta: next, next: previous + next, diverged: true };
}

function createOutput(model: Model<Api>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "pending",
		timestamp: Date.now(),
	};
}

export const stream: StreamFunction<"gemini-web", GeminiWebOptions> = (
	model,
	context,
	options,
): AssistantMessageEventStream => {
	const events = new AssistantMessageEventStream();
	void runStream(events, model, context, options);
	return events;
};

export const streamSimple = (
	model: Model<"gemini-web">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => stream(model, context, options as GeminiWebOptions | undefined);

async function runStream(
	streamEvents: AssistantMessageEventStream,
	model: Model<"gemini-web">,
	context: Context,
	options?: GeminiWebOptions,
): Promise<void> {
	const output = createOutput(model);
	streamEvents.push({ type: "start", partial: output });

	try {
		const cookieFile = options?.apiKey;
		if (!cookieFile) throw new Error("Gemini Web requires a cookie.json file");
		const rawCookies = await readFile(cookieFile, "utf8");
		const jar = parseCookieFile(rawCookies);
		const cookie = cookieHeader(jar);
		if (!cookie) throw new Error(`No usable Google cookies found in ${cookieFile}`);
		const sapisidValue = sapisid(jar);
		const authorization = sapisidValue ? sapisidHash(sapisidValue, "https://gemini.google.com") : undefined;
		const fetchImpl = options?.fetch ?? globalThis.fetch;
		if (options?.fetch && options.fetch !== globalThis.fetch) throw new Error("Custom fetch is not supported by Gemini Web");
		const prefix = options?.authUser === undefined ? "" : `/u/${options.authUser}`;
		const baseUrl = "https://gemini.google.com";
		const tokens = await getPageTokens(fetchImpl, baseUrl, options?.authUser, cookie, authorization, options?.signal ?? new AbortController().signal);
		const reqid = nextRequestId();
		const url = `${baseUrl}${prefix}/${ENDPOINT}?bl=${encodeURIComponent(options?.bl ?? DEFAULT_BL)}&hl=en&_reqid=${reqid}&rt=c`;
		const headers: Record<string, string> = {
			cookie,
			"content-type": "application/x-www-form-urlencoded;charset=UTF-8",
			origin: baseUrl,
			referer: `${baseUrl}${prefix}/app`,
			"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
			"x-same-domain": "1",
		};
		if (authorization) headers.authorization = authorization;
		if (options?.authUser !== undefined) headers["x-goog-authuser"] = String(options.authUser);

		const payload = buildPayload(contextPrompt(context), model.id, options?.think ?? 4, tokens.at);
		const preparedPayload = await options?.onPayload?.(payload, model);
		const response = await fetchImpl(url, {
			method: "POST",
			headers: { ...headers, ...(options?.headers ?? {}) },
			body: (preparedPayload ?? payload) as BodyInit,
			signal: options?.signal,
		});
		await options?.onResponse?.({ status: response.status, headers: Object.fromEntries(response.headers.entries()) }, model);
		if (!response.ok) throw new Error(`Gemini Web returned HTTP ${response.status}`);
		if (!response.body) throw new Error("Gemini Web returned an empty response body");

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		let emitted = "";
		let currentTextIndex = -1;
		let endedText = false;
		let sawTool = false;

		for (;;) {
			const { done, value } = await reader.read();
			buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
			if (buffer.length > MAX_FRAME_BYTES && !buffer.includes("\n")) throw new Error("Gemini Web frame exceeded safety limit");
			let newline = buffer.indexOf("\n");
			while (newline >= 0) {
				const line = buffer.slice(0, newline).replace(/\r$/, "");
				buffer = buffer.slice(newline + 1);
				const value = parseJsonLine(line);
				const inner = value === undefined ? undefined : parseInner(value);
				if (inner !== undefined) {
					const reason = metadata(inner, "blockReason");
					if (reason) throw new Error(`Gemini safety block: ${String(reason)}`);
					const text = longestCandidateText(inner);
					if (text !== undefined) {
						const { delta, next } = deltaFromCumulative(emitted, sanitizeSurrogates(text));
						emitted = next;
						if (delta) {
							if (endedText) {
								streamEvents.push({ type: "text_start", contentIndex: output.content.length, partial: output });
								currentTextIndex = output.content.length;
								output.content.push({ type: "text", text: "" });
								endedText = false;
							} else if (currentTextIndex < 0) {
								currentTextIndex = output.content.length;
								output.content.push({ type: "text", text: "" });
								streamEvents.push({ type: "text_start", contentIndex: currentTextIndex, partial: output });
							}
							const block = output.content[currentTextIndex];
							if (block?.type === "text") block.text += delta;
							streamEvents.push({ type: "text_delta", contentIndex: currentTextIndex, delta, partial: output });
						}
					}
					const calls = collectToolCalls(inner);
					for (const call of calls) {
						if (output.content.some((part) => part.type === "toolCall" && part.id === call.id)) continue;
						if (currentTextIndex >= 0 && !endedText) {
							const block = output.content[currentTextIndex];
							if (block?.type === "text") streamEvents.push({ type: "text_end", contentIndex: currentTextIndex, content: block.text, partial: output });
							endedText = true;
						}
						const index = output.content.length;
						output.content.push(call);
						streamEvents.push({ type: "toolcall_start", contentIndex: index, partial: output });
						streamEvents.push({ type: "toolcall_delta", contentIndex: index, delta: JSON.stringify(call.arguments), partial: output });
						streamEvents.push({ type: "toolcall_end", contentIndex: index, toolCall: call, partial: output });
						sawTool = true;
					}
					const usage = usageFrom(inner);
					if (usage) {
						output.usage.input = usage.input;
						output.usage.output = usage.output;
						output.usage.totalTokens = usage.totalTokens;
					}
					const finish = finishReason(inner);
					if (finish) output.rawStopReason = finish;
				}
				newline = buffer.indexOf("\n");
			}
			if (done) break;
		}
		buffer += decoder.decode();
		const finalValue = parseJsonLine(buffer);
		const finalInner = finalValue === undefined ? undefined : parseInner(finalValue);
		if (finalInner !== undefined) {
			const text = longestCandidateText(finalInner);
			if (text !== undefined && text.length > emitted.length) {
				const delta = text.slice(emitted.length);
				if (currentTextIndex >= 0 && output.content[currentTextIndex]?.type === "text") {
					(output.content[currentTextIndex] as TextContent).text += delta;
					streamEvents.push({ type: "text_delta", contentIndex: currentTextIndex, delta, partial: output });
				}
			}
		}

		if (currentTextIndex >= 0 && !endedText) {
			const block = output.content[currentTextIndex];
			if (block?.type === "text") streamEvents.push({ type: "text_end", contentIndex: currentTextIndex, content: block.text, partial: output });
		}
		output.stopReason = sawTool ? "toolUse" : "stop";
		streamEvents.push({ type: "done", reason: output.stopReason, message: output });
	} catch (error) {
		output.stopReason = options?.signal?.aborted ? "aborted" : "error";
		output.errorMessage = formatProviderError(error);
		streamEvents.push({ type: "error", reason: output.stopReason, error: output });
	}
}
