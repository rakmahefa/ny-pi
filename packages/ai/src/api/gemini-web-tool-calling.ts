import { createHash } from "node:crypto";
import type {
	AssistantMessage,
	Context,
	Model,
	SimpleStreamOptions,
	StreamFunction,
	TextContent,
	Tool,
	ToolCall,
} from "../types.ts";
import { AssistantMessageEventStream } from "../utils/event-stream.ts";
import { stream as baseStream, streamSimple as baseStreamSimple, type GeminiWebOptions } from "./gemini-web.ts";

const MAX_TOOL_SCHEMA_BYTES = 256 * 1024;
const MAX_TOOL_CALL_BLOCK_BYTES = 1024 * 1024;
const MAX_TOOL_NAME_BYTES = 128;
const TOOL_CALL_RE = /```(?:function_call|tool_call)\s*\r?\n([\s\S]*?)\r?\n\s*```/g;

const TOOL_USE_INSTRUCTION = [
	"# Tool Use",
	"",
	"You can call the following tools to help accomplish tasks. These tools connect to the user's local environment and will execute when called.",
	"",
	"Call format (use this exact format):",
	"```function_call",
	'{"name": "<tool_name>", "args": {<arguments>}}',
	"```",
	"",
	"When calling tools:",
	"- Output ONLY the function_call block(s), nothing else",
	"- You may call multiple tools with multiple blocks",
	"- After receiving a [tool:<tool_name>] result, use that data to answer the user",
	"",
	"Available tools:",
].join("\n");

interface ParsedFunctionCall {
	name: string;
	arguments: Record<string, any>;
}

function cloneJson<T>(value: T): T {
	return JSON.parse(JSON.stringify(value)) as T;
}

function serializeToolSchemas(tools: Tool[]): string {
	const definitions = tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: cloneJson(tool.parameters) }));
	const json = JSON.stringify(definitions, null, 2);
	if (Buffer.byteLength(json, "utf8") > MAX_TOOL_SCHEMA_BYTES) throw new Error(`Gemini Web tool schema exceeds the safety limit of ${MAX_TOOL_SCHEMA_BYTES} bytes`);
	return json;
}

function toolCallBlock(call: ToolCall): string {
	return `\`\`\`function_call\n${JSON.stringify({ name: call.name, args: call.arguments })}\n\`\`\``;
}

function projectAssistantToolCalls(message: AssistantMessage): AssistantMessage {
	if (!message.content.some((part) => part.type === "toolCall")) return message;
	const content: TextContent[] = [];
	for (const part of message.content) {
		if (part.type === "toolCall") content.push({ type: "text", text: toolCallBlock(part) });
		else if (part.type === "text") content.push({ type: "text", text: part.text });
		else if (part.type === "thinking") content.push({ type: "text", text: `<thinking>\n${part.thinking}\n</thinking>` });
	}
	return { ...message, content };
}

function augmentContext(context: Context): Context {
	if (!context.tools?.length) return context;
	const toolInstruction = `${TOOL_USE_INSTRUCTION}\n${serializeToolSchemas(context.tools)}`;
	const systemPrompt = context.systemPrompt?.trim() ? `${context.systemPrompt.trim()}\n\n${toolInstruction}` : toolInstruction;
	const messages = context.messages.map((message) => (message.role === "assistant" ? projectAssistantToolCalls(message) : message));
	return { ...context, systemPrompt, messages };
}

function parseJsonObject(value: unknown, label: string): Record<string, any> {
	if (typeof value === "string") {
		let parsed: unknown;
		try {
			parsed = JSON.parse(value) as unknown;
		} catch {
			throw new Error(`Gemini Web ${label} contains invalid JSON`);
		}
		return parseJsonObject(parsed, label);
	}
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(`Gemini Web ${label} must be a JSON object`);
	}
	return value as Record<string, any>;
}

function normalizeFunctionCallObject(object: Record<string, unknown>): ParsedFunctionCall {
	const name = typeof object.name === "string" ? object.name.trim() : "";
	if (!name) throw new Error("Gemini Web function_call is missing a tool name");
	if (Buffer.byteLength(name, "utf8") > MAX_TOOL_NAME_BYTES) {
		throw new Error(`Gemini Web function_call tool name exceeds ${MAX_TOOL_NAME_BYTES} bytes`);
	}

	const explicitArguments = object.args ?? object.arguments ?? object.parameters;
	if (explicitArguments !== undefined) {
		return { name, arguments: parseJsonObject(explicitArguments, `function_call arguments for ${name}`) };
	}

	// Gemini Web sometimes emits a flattened tool call such as
	// {"name":"write","content":"..."} instead of the canonical
	// {"name":"write","args":{"content":"..."}}.
	// Normalize only when no explicit argument container is present. This
	// preserves arbitrary large values (including multi-line file contents)
	// without reparsing or truncating them.
	const flattened = Object.create(null) as Record<string, any>;
	for (const [key, value] of Object.entries(object)) {
		if (key === "name") continue;
		flattened[key] = value;
	}
	return { name, arguments: flattened };
}

function parseFunctionCallBlocks(text: string): ParsedFunctionCall[] {
	const calls: ParsedFunctionCall[] = [];
	for (const match of text.matchAll(TOOL_CALL_RE)) {
		const raw = match[1]?.trim() ?? "";
		if (!raw) continue;
		if (Buffer.byteLength(raw, "utf8") > MAX_TOOL_CALL_BLOCK_BYTES) throw new Error(`Gemini Web function_call block exceeds ${MAX_TOOL_CALL_BLOCK_BYTES} bytes`);
		let parsed: unknown;
		try {
			parsed = JSON.parse(raw) as unknown;
		} catch {
			throw new Error("Gemini Web returned an invalid function_call JSON block");
		}
		if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Gemini Web function_call payload must be a JSON object");
		calls.push(normalizeFunctionCallObject(parsed as Record<string, unknown>));
	}
	return calls;
}

function canonicalCallKey(call: ParsedFunctionCall): string {
	return createHash("sha256").update(`${call.name}\n${JSON.stringify(call.arguments)}`).digest("hex");
}

function convertAssistantMessage(message: AssistantMessage): AssistantMessage {
	const originalText = message.content.filter((part): part is TextContent => part.type === "text").map((part) => part.text).join("\n");
	if (!originalText.includes("```function_call") && !originalText.includes("```tool_call")) return message;
	const calls = parseFunctionCallBlocks(originalText);
	if (!calls.length) return message;
	const seen = new Set<string>();
	const toolCalls: ToolCall[] = [];
	for (let index = 0; index < calls.length; index++) {
		const call = calls[index]!;
		const key = canonicalCallKey(call);
		if (seen.has(key)) continue;
		seen.add(key);
		toolCalls.push({ type: "toolCall", id: `gemini_web_${key.slice(0, 16)}_${index}`, name: call.name, arguments: call.arguments });
	}
	const cleanedText = originalText.replace(TOOL_CALL_RE, "").trim();
	const content: AssistantMessage["content"] = [];
	if (cleanedText) content.push({ type: "text", text: cleanedText });
	content.push(...toolCalls);
	return { ...message, content, stopReason: "toolUse" };
}

function wrapStream(base: AssistantMessageEventStream): AssistantMessageEventStream {
	const output = new AssistantMessageEventStream();
	void (async () => {
		try {
			for await (const event of base) {
				if (event.type === "done") {
					const final = convertAssistantMessage(event.message);
					output.push(final.stopReason === "toolUse" ? { type: "done", reason: "toolUse", message: final } : { type: "done", reason: event.reason, message: final });
				} else output.push(event);
			}
		} catch (error) {
			const message: AssistantMessage = {
				role: "assistant", content: [], api: "gemini-web", provider: "gemini-web", model: "unknown",
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
				stopReason: "error", errorMessage: error instanceof Error ? error.message : String(error), timestamp: Date.now(),
			};
			output.push({ type: "error", reason: "error", error: message });
		}
	})();
	return output;
}

export function geminiWebToolCallingStream(model: Model<"gemini-web">, context: Context, options?: GeminiWebOptions): AssistantMessageEventStream {
	return wrapStream(baseStream(model, augmentContext(context), options));
}

export function geminiWebToolCallingStreamSimple(model: Model<"gemini-web">, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	return wrapStream(baseStreamSimple(model, augmentContext(context), options));
}

export function geminiWebToolCallingApi(): { stream: StreamFunction<"gemini-web", GeminiWebOptions>; streamSimple: StreamFunction<"gemini-web", SimpleStreamOptions> } {
	return { stream: geminiWebToolCallingStream, streamSimple: geminiWebToolCallingStreamSimple };
}

export const __geminiWebToolCallingTestables = { augmentContext, parseFunctionCallBlocks, convertAssistantMessage, serializeToolSchemas, normalizeFunctionCallObject };
