import { describe, expect, it } from "vitest";
import type { AssistantMessage, Context, Tool, ToolCall } from "../src/types.ts";
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

	it("converts Gemini function_call blocks into structured Pi tool calls", () => {
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
});
