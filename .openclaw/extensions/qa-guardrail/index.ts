/**
 * QA Guardrail OpenClaw Plugin
 *
 * Implements "agents propose, validators decide" at the tool boundary.
 *
 * Intercepts tool calls via before_tool_call hook and runs QA guardrail
 * validation before allowing execution.
 *
 * Configuration (in .openclaw/config.json):
 * {
 *   "extensions": {
 *     "qa-guardrail": {
 *       "enabled": true,
 *       "pythonPath": "python",
 *       "guardrailPath": "qa_alphageometry_ptolemy/qa_guardrail/qa_guardrail.py",
 *       "policy": {
 *         "deny": ["tool.exec", "tool.gmail.*"],
 *         "require_verified_ic_cert": false,
 *         "required_capability": null
 *       },
 *       "capabilities": ["READ", "WRITE"],
 *       "logDenials": true
 *     }
 *   }
 * }
 */

import { spawn } from "child_process";
import * as path from "path";

// Types matching GUARDRAIL_REQUEST.v1 and GUARDRAIL_RESULT.v1
interface GuardrailRequest {
  planned_move: string;
  context: {
    active_generators?: string[];
    policy?: Record<string, unknown>;
    capabilities?: string[];
    instruction_content_cert?: {
      schema_id: string;
      verified?: boolean;
      instruction_domain?: string[];
      content_domain?: string[];
    };
    trace_tail?: Array<{
      move: string;
      fail_type: string | null;
      invariant_diff: Record<string, unknown>;
    }>;
  };
}

interface GuardrailResult {
  ok: boolean;
  result: "ALLOW" | "DENY";
  checks: string[];
  fail_record?: {
    move: string;
    fail_type: string;
    invariant_diff: Record<string, unknown>;
    detail?: string;
    timestamp_utc?: string;
  };
  error?: string;
}

interface PluginConfig {
  enabled: boolean;
  pythonPath: string;
  guardrailPath: string;
  policy: {
    deny?: string[];
    allow?: string[];
    require_verified_ic_cert?: boolean;
    required_capability?: string;
  };
  capabilities: string[];
  activeGenerators: string[];
  logDenials: boolean;
}

// Default configuration
const DEFAULT_CONFIG: PluginConfig = {
  enabled: true,
  pythonPath: "python",
  guardrailPath: "qa_alphageometry_ptolemy/qa_guardrail/qa_guardrail.py",
  policy: {
    deny: [],
    require_verified_ic_cert: false,
  },
  capabilities: [],
  activeGenerators: ["sigma", "mu", "lambda", "nu"],
  logDenials: true,
};

/**
 * Convert OpenClaw tool call to QA planned_move format.
 *
 * Examples:
 *   { toolName: "bash", params: { command: "ls" } } -> "tool.bash({command: 'ls'})"
 *   { toolName: "web_search", params: { query: "..." } } -> "tool.web_search({query: '...'})"
 */
function toolCallToPlannedMove(toolName: string, params: Record<string, unknown>): string {
  // Truncate long param values for readability
  const truncatedParams: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === "string" && value.length > 100) {
      truncatedParams[key] = value.slice(0, 100) + "...";
    } else {
      truncatedParams[key] = value;
    }
  }
  const paramsStr = JSON.stringify(truncatedParams);
  return `tool.${toolName}(${paramsStr})`;
}

/**
 * Run the Python guardrail via subprocess.
 */
async function runGuardrail(
  request: GuardrailRequest,
  config: PluginConfig,
  workspaceRoot: string
): Promise<GuardrailResult> {
  return new Promise((resolve, reject) => {
    const guardrailPath = path.resolve(workspaceRoot, config.guardrailPath);
    const proc = spawn(config.pythonPath, [guardrailPath, "--guard"], {
      cwd: workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Guardrail process exited with code ${code}: ${stderr}`));
        return;
      }
      try {
        const result: GuardrailResult = JSON.parse(stdout);
        resolve(result);
      } catch (e) {
        reject(new Error(`Failed to parse guardrail output: ${stdout}`));
      }
    });

    proc.on("error", (err) => {
      reject(err);
    });

    // Send request to stdin
    proc.stdin.write(JSON.stringify(request));
    proc.stdin.end();
  });
}

/**
 * Map tool name to capability requirement.
 */
function toolToCapability(toolName: string): string | null {
  const capabilityMap: Record<string, string> = {
    bash: "EXEC",
    exec: "EXEC",
    shell: "EXEC",
    write: "WRITE",
    edit: "WRITE",
    file_write: "WRITE",
    gmail: "NET_CREDENTIAL",
    google_search: "NET",
    web_search: "NET",
    web_fetch: "NET",
    browser: "NET_CREDENTIAL",
  };
  return capabilityMap[toolName.toLowerCase()] || null;
}

/**
 * Check if a tool matches a pattern (supports glob-like wildcards).
 */
function matchesPattern(toolName: string, pattern: string): boolean {
  if (pattern.endsWith(".*")) {
    const prefix = pattern.slice(0, -2);
    return toolName.startsWith(prefix);
  }
  return toolName === pattern || `tool.${toolName}` === pattern;
}

// ============================================================================
// OPENCLAW PLUGIN HOOKS
// ============================================================================

export interface ToolCallContext {
  toolName: string;
  params: Record<string, unknown>;
  sessionId?: string;
  messageId?: string;
}

export interface PluginHooks {
  before_tool_call?: (
    ctx: ToolCallContext,
    config: PluginConfig,
    workspaceRoot: string
  ) => Promise<{ allow: boolean; reason?: string; failRecord?: GuardrailResult["fail_record"] }>;
  after_tool_call?: (
    ctx: ToolCallContext,
    result: unknown,
    config: PluginConfig
  ) => Promise<void>;
}

/**
 * Main plugin export.
 */
export const hooks: PluginHooks = {
  /**
   * before_tool_call: The authoritative gate.
   *
   * Converts tool call to QA planned_move, runs guardrail, blocks if DENY.
   */
  async before_tool_call(ctx, config, workspaceRoot) {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    if (!cfg.enabled) {
      return { allow: true };
    }

    const plannedMove = toolCallToPlannedMove(ctx.toolName, ctx.params);

    // Build context for guardrail
    const guardrailContext: GuardrailRequest["context"] = {
      active_generators: cfg.activeGenerators,
      policy: cfg.policy,
      capabilities: cfg.capabilities,
    };

    // Check if tool requires elevated capability
    const requiredCap = toolToCapability(ctx.toolName);
    if (requiredCap && !cfg.capabilities.includes(requiredCap)) {
      // Add capability requirement to policy for this call
      guardrailContext.policy = {
        ...guardrailContext.policy,
        required_capability: requiredCap,
      };
    }

    const request: GuardrailRequest = {
      planned_move: plannedMove,
      context: guardrailContext,
    };

    try {
      const result = await runGuardrail(request, cfg, workspaceRoot);

      if (result.ok && result.result === "ALLOW") {
        return { allow: true };
      }

      // DENY case
      if (cfg.logDenials) {
        console.error(`[QA-GUARDRAIL] DENY: ${plannedMove}`);
        console.error(`[QA-GUARDRAIL] Checks: ${result.checks.join(", ")}`);
        if (result.fail_record) {
          console.error(
            `[QA-GUARDRAIL] Fail: ${result.fail_record.fail_type} - ${result.fail_record.detail}`
          );
        }
      }

      return {
        allow: false,
        reason: result.fail_record?.detail || "Guardrail denied",
        failRecord: result.fail_record,
      };
    } catch (err) {
      // On guardrail error, default to DENY (fail-closed)
      console.error(`[QA-GUARDRAIL] Error running guardrail: ${err}`);
      return {
        allow: false,
        reason: `Guardrail error: ${err}`,
      };
    }
  },

  /**
   * after_tool_call: Log successful calls for audit trail.
   */
  async after_tool_call(ctx, result, config) {
    // Optional: append to trace for future context
    // This could write to a trace file or emit to a logging service
  },
};

/**
 * Plugin metadata for OpenClaw discovery.
 */
export const metadata = {
  name: "qa-guardrail",
  version: "1.0.0",
  description: "QA Guardrail - agents propose, validators decide",
  author: "QA Team",
  hooks: ["before_tool_call", "after_tool_call"],
};

export default { hooks, metadata };
