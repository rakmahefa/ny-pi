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
const TOOL_CALL_START_RE = /```(?:function_call|tool_call)[ \t]*\r?\n/g;

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

interface FunctionCallBlock {
	raw: string;
	start: number;
	end: number;
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
			try {
				parsed = JSON.parse(escapeLiteralJsonStringControls(value)) as unknown;
			} catch {
				throw new Error(`Gemini Web ${label} must be a JSON object; received an invalid JSON string`);
			}
		}
		return parseJsonObject(parsed, label);
	}
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		throw new Error(`Gemini Web ${label} must be a JSON object`);
	}
	return value as Record<string, any>;
}

/** Escape only control characters that a model may emit literally inside JSON strings. */
function escapeLiteralJsonStringControls(value: string): string {
	let output = "";
	let inString = false;
	let escaped = false;
	for (const char of value) {
		if (inString) {
			if (escaped) {
				output += char;
				escaped = false;
				continue;
			}
			if (char === "\\") {
				output += char;
				escaped = true;
				continue;
			}
			if (char === '"') {
				output += char;
				inString = false;
				continue;
			}
			const code = char.charCodeAt(0);
			if (code < 0x20) {
				output += `\\u${code.toString(16).padStart(4, "0")}`;
				continue;
			}
			output += char;
			continue;
		}
		if (char === '"') inString = true;
		output += char;
	}
	if (inString) throw new Error("Gemini Web function_call payload contains an unterminated JSON string");
	return output;
}

function parseFunctionCallJson(raw: string): Record<string, unknown> {
	let parsed: unknown;
	try {
		parsed = JSON.parse(raw) as unknown;
	} catch {
		try {
			parsed = JSON.parse(escapeLiteralJsonStringControls(raw)) as unknown;
		} catch {
			throw new Error("Gemini Web returned an invalid function_call JSON block");
		}
	}
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("Gemini Web function_call payload must be a JSON object");
	}
	return parsed as Record<string, unknown>;
}

function normalizeFunctionCallObject(object: Record<string, unknown>): ParsedFunctionCall {
	const name = typeof object.name === "string" ? object.name.trim() : "";
	if (!name) throw new Error("Gemini Web function_call is missing a tool name");
	if (Buffer.byteLength(name, "utf8") > MAX_TOOL_NAME_BYTES) {
		throw new Error(`Gemini Web function_call tool name exceeds ${MAX_TOOL_NAME_BYTES} bytes`);
	}

	const explicitKey = ["args", "arguments", "parameters"].find((key) => Object.prototype.hasOwnProperty.call(object, key));
	if (explicitKey) {
		return {
			name,
			arguments: parseJsonObject(object[explicitKey], `function_call arguments for ${name}`),
		};
	}

	const flattened = Object.create(null) as Record<string, any>;
	for (const [key, value] of Object.entries(object)) {
		if (key !== "name") flattened[key] = value;
	}
	return { name, arguments: flattened };
}

function findJsonObjectEnd(text: string, start: number): number {
	if (text[start] !== "{") throw new Error("Gemini Web function_call payload must be a JSON object");
	let depth = 0;
	let inString = false;
	let escaped = false;
	for (let index = start; index < text.length; index++) {
		const char = text[index]!;
		if (inString) {
			if (escaped) escaped = false;
			else if (char === "\\") escaped = true;
			else if (char === '"') inString = false;
			continue;
		}
		if (char === '"') {
			inString = true;
			continue;
		}
		if (char === "{") depth++;
		else if (char === "}" && --depth === 0) return index + 1;
	}
	throw new Error("Gemini Web returned an unterminated function_call JSON block");
}

function extractFunctionCallBlocks(text: string): FunctionCallBlock[] {
	const blocks: FunctionCallBlock[] = [];
	for (const match of text.matchAll(TOOL_CALL_START_RE)) {
		const blockStart = match.index ?? 0;
		let payloadStart = blockStart + match[0].length;
		while (payloadStart < text.length && /[ \t\r\n]/.test(text[payloadStart]!)) payloadStart++;
		const payloadEnd = findJsonObjectEnd(text, payloadStart);
		const raw = text.slice(payloadStart, payloadEnd).trim();
		if (Buffer.byteLength(raw, "utf8") > MAX_TOOL_CALL_BLOCK_BYTES) throw new Error(`Gemini Web function_call block exceeds ${MAX_TOOL_CALL_BLOCK_BYTES} bytes`);
		let fenceStart = payloadEnd;
		while (fenceStart < text.length && /[ \t\r\n]/.test(text[fenceStart]!)) fenceStart++;
		if (text.slice(fenceStart, fenceStart + 3) !== "```") throw new Error("Gemini Web function_call block is missing its closing fence");
		blocks.push({ raw, start: blockStart, end: fenceStart + 3 });
	}
	return blocks;
}

function parseFunctionCallBlocks(text: string): ParsedFunctionCall[] {
	return extractFunctionCallBlocks(text).map((block) => normalizeFunctionCallObject(parseFunctionCallJson(block.raw)));
}

/** Remove only protocol fences, including one line break on each side, preserving surrounding text exactly. */
function removeFunctionCallBlocks(text: string, blocks: FunctionCallBlock[]): string {
	let cleaned = text;
	for (const block of [...blocks].reverse()) {
		let start = block.start;
		let end = block.end;
		if (start > 0) {
			if (cleaned.slice(start - 2, start) === "\r\n") start -= 2;
			else if (cleaned[start - 1] === "\n" || cleaned[start - 1] === "\r") start -= 1;
		}
		if (end < cleaned.length) {
			if (cleaned.slice(end, end + 2) === "\r\n") end += 2;
			else if (cleaned[end] === "\n" || cleaned[end] === "\r") end += 1;
		}
		cleaned = cleaned.slice(0, start) + "\n" + cleaned.slice(end);
		if (!cleaned.slice(0, start).trim()) cleaned = cleaned.slice(1);
	}
	return cleaned.trim();
}

function canonicalCallKey(call: ParsedFunctionCall): string {
	return createHash("sha256").update(`${call.name}\n${JSON.stringify(call.arguments)}`).digest("hex");
}

function convertAssistantMessage(message: AssistantMessage): AssistantMessage {
	const content: AssistantMessage["content"] = [];
	const seen = new Set<string>();
	let foundProtocol = false;

	for (const part of message.content) {
		if (part.type === "text") {
			if (!part.text.includes("```function_call") && !part.text.includes("```tool_call")) {
				content.push(part);
				continue;
			}
			const blocks = extractFunctionCallBlocks(part.text);
			const calls = blocks.map((block) => normalizeFunctionCallObject(parseFunctionCallJson(block.raw)));
			const cleanedText = removeFunctionCallBlocks(part.text, blocks);
			if (cleanedText) content.push({ type: "text", text: cleanedText });
			for (const call of calls) {
				const key = canonicalCallKey(call);
				if (seen.has(key)) continue;
				seen.add(key);
				const toolIndex = content.filter((entry) => entry.type === "toolCall").length;
				content.push({ type: "toolCall", id: `gemini_web_${key.slice(0, 16)}_${toolIndex}`, name: call.name, arguments: call.arguments });
			}
			foundProtocol = true;
			continue;
		}
		if (part.type === "toolCall") {
			const key = canonicalCallKey({ name: part.name, arguments: part.arguments });
			if (seen.has(key)) continue;
			seen.add(key);
		}
		content.push(part);
	}

	if (!foundProtocol) {
		return message.content.some((part) => part.type === "toolCall") ? { ...message, stopReason: "toolUse" } : message;
	}
	return { ...message, content, stopReason: "toolUse" };
}

function emitFinalContentEvents(output: AssistantMessageEventStream, message: AssistantMessage): void {
	for (let index = 0; index < message.content.length; index++) {
		const part = message.content[index]!;
		if (part.type === "text") {
			if (!part.text) continue;
			output.push({ type: "text_start", contentIndex: index, partial: message });
			output.push({ type: "text_delta", contentIndex: index, delta: part.text, partial: message });
			output.push({ type: "text_end", contentIndex: index, content: part.text, partial: message });
		} else if (part.type === "thinking") {
			if (!part.thinking) continue;
			output.push({ type: "thinking_start", contentIndex: index, partial: message });
			output.push({ type: "thinking_delta", contentIndex: index, delta: part.thinking, partial: message });
			output.push({ type: "thinking_end", contentIndex: index, content: part.thinking, partial: message });
		} else if (part.type === "toolCall") {
			output.push({ type: "toolcall_start", contentIndex: index, partial: message });
			output.push({ type: "toolcall_delta", contentIndex: index, delta: JSON.stringify(part.arguments), partial: message });
			output.push({ type: "toolcall_end", contentIndex: index, toolCall: part, partial: message });
		}
	}
}

function wrapStream(base: AssistantMessageEventStream, bufferToolProtocol = false): AssistantMessageEventStream {
	const output = new AssistantMessageEventStream();
	void (async () => {
		try {
			for await (const event of base) {
				if (event.type === "done") {
					const final = convertAssistantMessage(event.message);
					if (bufferToolProtocol) emitFinalContentEvents(output, final);
					output.push(final.stopReason === "toolUse" ? { type: "done", reason: "toolUse", message: final } : { type: "done", reason: event.reason, message: final });
				} else if (event.type === "error") {
					output.push(event);
				} else if (!bufferToolProtocol || event.type === "start") {
					output.push(event);
				}
			}
		} catch (error) {
			const errorMessage: AssistantMessage = {
				role: "assistant", content: [], api: "gemini-web", provider: "gemini-web", model: "unknown",
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "error", errorMessage: error instanceof Error ? error.message : String(error), timestamp: Date.now(),
			};
			output.push({ type: "error", reason: "error", error: errorMessage });
		}
	})();
	return output;
}

export function geminiWebToolCallingStream(model: Model<"gemini-web">, context: Context, options?: GeminiWebOptions): AssistantMessageEventStream {
	return wrapStream(baseStream(model, augmentContext(context), options), Boolean(context.tools?.length));
}

export function geminiWebToolCallingStreamSimple(model: Model<"gemini-web">, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	return wrapStream(baseStreamSimple(model, augmentContext(context), options), Boolean(context.tools?.length));
}

export function geminiWebToolCallingApi(): { stream: StreamFunction<"gemini-web", GeminiWebOptions>; streamSimple: StreamFunction<"gemini-web", SimpleStreamOptions> } {
	return { stream: geminiWebToolCallingStream, streamSimple: geminiWebToolCallingStreamSimple };
}

export const __geminiWebToolCallingTestables = {
	augmentContext,
	parseFunctionCallBlocks,
	convertAssistantMessage,
	serializeToolSchemas,
	normalizeFunctionCallObject,
	extractFunctionCallBlocks,
	wrapStream,
};
