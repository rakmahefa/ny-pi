import { describe, expect, it } from "vitest";
import type { AssistantMessage, Context, Tool, ToolCall } from "../src/types.ts";
import { AssistantMessageEventStream } from "../src/utils/event-stream.ts";
import { __geminiWebToolCallingTestables as testables } from "../src/api/gemini-web-tool-calling.ts";

const tool: Tool = {
	name: "read",
	description: "Read a file from the workspace.",
	parameters: {
		type: "object",
		properties: {
			path: { type: "string", description: "Path to read" },
		},
		required: ["path"],
	},
};

function assistant(content: AssistantMessage["content"]): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "gemini-web",
		provider: "gemini-web",
		model: "gemini-3.6-flash",
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 1,
	};
}

describe("Gemini Web tool-calling bridge", () => {
	it("serializes Pi tool schemas without mutating the definitions", () => {
		const schema = testables.serializeToolSchemas([tool]);
		const parsed = JSON.parse(schema) as Array<Record<string, unknown>>;
		expect(parsed).toEqual([
			{
				name: "read",
				description: "Read a file from the workspace.",
				parameters: tool.parameters,
			},
		]);
		expect(tool.name).toBe("read");
	});

	it("adds the tool protocol and schema to the Gemini conversation", () => {
		const context: Context = {
			systemPrompt: "You are a coding agent.",
			messages: [
				{ role: "user", content: "Read README.md", timestamp: 1 },
			],
			tools: [tool],
		};
		const augmented = testables.augmentContext(context);
		expect(augmented.systemPrompt).toContain("```function_call");
		expect(augmented.systemPrompt).toContain('"name": "read"');
		expect(augmented.systemPrompt).toContain('"path"');
		expect(augmented.messages).toHaveLength(1);
		expect(augmented.messages[0]).toEqual(context.messages[0]);
	});

	it("converts canonical Gemini function_call blocks into structured Pi tool calls", () => {
		const message = assistant([
			{
				type: "text",
				text: '```function_call\n{"name":"read","args":{"path":"README.md"}}\n```',
			},
		]);
		const converted = testables.convertAssistantMessage(message);
		const call = converted.content.find((part): part is ToolCall => part.type === "toolCall");
		expect(converted.stopReason).toBe("toolUse");
		expect(call).toMatchObject({
			type: "toolCall",
			name: "read",
			arguments: { path: "README.md" },
		});
		expect(converted.content.some((part) => part.type === "text")).toBe(false);
	});

	it("normalizes flattened arguments emitted by Gemini Web", () => {
		const calls = testables.parseFunctionCallBlocks(
			'```function_call\n{"name":"write","path":"README.md","content":"# Arcade Pro Python\\n\\n## Games Included\\n\\n1. Snake\\n2. Pong"}\n```',
		);
		expect(calls).toEqual([
			{
				name: "write",
				arguments: {
					path: "README.md",
					content: "# Arcade Pro Python\n\n## Games Included\n\n1. Snake\n2. Pong",
				},
			},
		]);
	});

	it("accepts Gemini's non-strict multiline strings in flattened calls", () => {
		const text = [
			"```function_call",
			'{"name":"write","content":"[project] ',
			'name = \\"gemini-oauth-bridge\\"',
			'version = \\"0.1.0\\"',
			'dependencies = [',
			'  \\"fastapi>=0.141.1\\",',
			']",',
			'"path":"gemini-oauth-bridge/pyproject.toml"}',
			"```",
		].join("\n");
		const calls = testables.parseFunctionCallBlocks(text);
		expect(calls).toEqual([
			{
				name: "write",
				arguments: {
					content: '[project] \nname = "gemini-oauth-bridge"\nversion = "0.1.0"\ndependencies = [\n  "fastapi>=0.141.1",\n]',
					path: "gemini-oauth-bridge/pyproject.toml",
				},
			},
		]);
	});

	it("preserves large multi-line flattened content without truncation", () => {
		const content = Array.from({ length: 2_000 }, (_, index) => `line-${index}: ${"x".repeat(80)}`).join("\n");
		const raw = JSON.stringify({ name: "write", path: "README.md", content });
		const calls = testables.parseFunctionCallBlocks("```function_call\n" + raw + "\n```");
		expect(calls).toHaveLength(1);
		expect(calls[0]?.name).toBe("write");
		expect(calls[0]?.arguments.content).toBe(content);
		expect(calls[0]?.arguments.content.length).toBe(content.length);
	});

	it("does not truncate Markdown code fences contained inside a write payload", () => {
		const content = "# Title\n\n```python\nprint(\"hello {world}\")\n```\n\nafter";
		const raw = JSON.stringify({ name: "write", path: "README.md", content });
		const calls = testables.parseFunctionCallBlocks("```function_call\n" + raw + "\n```");
		expect(calls).toEqual([{ name: "write", arguments: { path: "README.md", content } }]);
	});

	it("accepts CRLF function_call fences", () => {
		const calls = testables.parseFunctionCallBlocks(
			'```function_call\r\n{"name":"read","args":{"path":"README.md"}}\r\n```',
		);
		expect(calls).toEqual([{ name: "read", arguments: { path: "README.md" } }]);
	});

	it("accepts explicit argument containers as well as their JSON-string form", () => {
		const objectForm = testables.parseFunctionCallBlocks(
			'```function_call\n{"name":"read","arguments":{"path":"a.txt"}}\n```',
		);
		const stringForm = testables.parseFunctionCallBlocks(
			'```tool_call\n{"name":"read","arguments":"{\\"path\\":\\"b.txt\\"}"}\n```',
		);
		expect(objectForm).toEqual([{ name: "read", arguments: { path: "a.txt" } }]);
		expect(stringForm).toEqual([{ name: "read", arguments: { path: "b.txt" } }]);
	});

	it("supports multiple function_call blocks in source order", () => {
		const calls = testables.parseFunctionCallBlocks([
			"```function_call",
			'{"name":"read","args":{"path":"a.txt"}}',
			"```",
			"",
			"```function_call",
			'{"name":"read","args":{"path":"b.txt"}}',
			"```",
		].join("\n"));
		expect(calls).toEqual([
			{ name: "read", arguments: { path: "a.txt" } },
			{ name: "read", arguments: { path: "b.txt" } },
		]);
	});

	it("replays previous assistant tool calls in the protocol format", () => {
		const call: ToolCall = {
			type: "toolCall",
			id: "gemini_web_call_1",
			name: "read",
			arguments: { path: "README.md" },
		};
		const context: Context = {
			messages: [
				assistant([call]),
				{
					role: "toolResult",
					toolCallId: call.id,
					toolName: call.name,
					content: [{ type: "text", text: "hello" }],
					isError: false,
					timestamp: 2,
				},
			],
			tools: [tool],
		};
		const augmented = testables.augmentContext(context);
		const replayed = augmented.messages[0];
		expect(replayed.role).toBe("assistant");
		if (replayed.role === "assistant") {
			expect(replayed.content).toEqual([
				{
					type: "text",
					text: '```function_call\n{"name":"read","args":{"path":"README.md"}}\n```',
				},
			]);
		}
	});

	it("rejects malformed function_call JSON instead of executing it", () => {
		expect(() =>
			testables.parseFunctionCallBlocks('```function_call\n{"name":"read","args":}\n```'),
		).toThrow(/invalid function_call JSON/i);
	});

	it("rejects a non-object explicit argument container", () => {
		expect(() =>
			testables.parseFunctionCallBlocks('```function_call\n{"name":"read","args":"hello"}\n```'),
		).toThrow(/function_call arguments.*JSON object/i);
	});

	it("rejects an explicit null argument container instead of treating it as flattened", () => {
		expect(() =>
			testables.parseFunctionCallBlocks('```function_call\n{"name":"read","args":null,"path":"README.md"}\n```'),
		).toThrow(/function_call arguments.*JSON object/i);
	});

	it("accepts the real write-style flattened payload shape", () => {
		const message = assistant([
			{
				type: "text",
				text: '```function_call\n{"name":"write","content":"# Arcade Pro Python\\n\\nA professional arcade platform featuring 10 fully functional retro-style games built with Pygame and managed with uv.","path":"gemini-oauth-bridge/README.md"}\n```',
			},
		]);
		const converted = testables.convertAssistantMessage(message);
		const call = converted.content.find((part): part is ToolCall => part.type === "toolCall");
		expect(call).toMatchObject({
			name: "write",
			arguments: {
				path: "gemini-oauth-bridge/README.md",
				content: expect.stringContaining("Arcade Pro Python"),
			},
		});
	});

	it("preserves thinking and native tool calls while converting textual calls", () => {
		const nativeCall: ToolCall = {
			type: "toolCall",
			id: "native-1",
			name: "read",
			arguments: { path: "native.txt" },
		};
		const message = assistant([
			{ type: "thinking", thinking: "Need to inspect both files." },
			{ type: "text", text: "before\n```function_call\n{\"name\":\"read\",\"path\":\"README.md\"}\n```\nafter" },
			nativeCall,
		]);
		const converted = testables.convertAssistantMessage(message);
		expect(converted.content[0]).toEqual({ type: "thinking", thinking: "Need to inspect both files." });
		expect(converted.content.some((part) => part.type === "text" && part.text === "before\nafter")).toBe(true);
		expect(converted.content.filter((part) => part.type === "toolCall")).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ id: "native-1", name: "read" }),
				expect.objectContaining({ name: "read", arguments: { path: "README.md" } }),
			]),
		);
	});

	it("does not leak raw textual function_call protocol when tool mode is enabled", async () => {
		const base = new AssistantMessageEventStream();
		const wrapped = testables.wrapStream(base, true);
		const collected: Array<{ type: string; [key: string]: unknown }> = [];
		const consume = (async () => {
			for await (const event of wrapped) collected.push(event as { type: string; [key: string]: unknown });
		})();

		const message = assistant([
			{ type: "text", text: '```function_call\n{"name":"read","path":"README.md"}\n```' },
		]);
		base.push({ type: "start", partial: assistant([]) });
		base.push({
			type: "text_start",
			contentIndex: 0,
			partial: message,
		});
		base.push({
			type: "text_delta",
			contentIndex: 0,
			delta: '```function_call\n{"name":"read","path":"README.md"}\n```',
			partial: message,
		});
		base.push({ type: "text_end", contentIndex: 0, content: message.content[0]!.type === "text" ? message.content[0]!.text : "", partial: message });
		base.push({ type: "done", reason: "stop", message });

		await consume;
		expect(collected.some((event) => event.type === "text_delta" && String(event.delta).includes("function_call"))).toBe(false);
		expect(collected.some((event) => event.type === "toolcall_end")).toBe(true);
		expect(collected.at(-1)?.type).toBe("done");
	});
});
