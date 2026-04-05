#!/usr/bin/env python3
"""
QA Terminal Agent - Network Chuck Inspired Multi-AI Orchestration
Persistent context for QA research with Claude Code, Codex, Gemini, and QALM
"""

import argparse
import json
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class QATerminalAgent:
    """Terminal-based multi-AI agent orchestrator for QA research"""

    def __init__(self, context_file: str, verbose: bool = False):
        self.context_file = Path(context_file)
        self.verbose = verbose
        self.context = self.load_context()

        # Resolve base_dir - find signal_experiments directory
        self.base_dir = Path(__file__).parent.parent.resolve()

        # Verify qa_lab exists, otherwise try to find it
        if not (self.base_dir / 'qa_lab').exists():
            # Try current working directory
            if (Path.cwd() / 'qa_lab').exists():
                self.base_dir = Path.cwd()
            # Try going up one level from cwd
            elif (Path.cwd().parent / 'qa_lab').exists():
                self.base_dir = Path.cwd().parent

        # AI providers with CLI commands
        self.providers = {
            'claude': {
                'call': self.call_claude,
                'description': 'Claude Code (current session)',
                'available': True
            },
            'codex': {
                'call': self.call_codex,
                'description': 'OpenAI Codex (code generation)',
                'available': self._check_command_exists('codex')
            },
            'gemini': {
                'call': self.call_gemini,
                'description': 'Google Gemini (analysis)',
                'available': self._check_command_exists('gemini')
            },
            'qalm': {
                'call': self.call_qalm,
                'description': 'QA Local Model (specialized)',
                'available': True  # Always available locally
            }
        }

    def _check_command_exists(self, command: str) -> bool:
        """Check if a command-line tool exists"""
        try:
            result = subprocess.run(
                ['which', command],
                capture_output=True,
                text=True,
                timeout=1
            )
            return result.returncode == 0
        except:
            return False

    def load_context(self) -> Dict:
        """Load persistent QA context from YAML file"""
        if self.context_file.exists():
            with open(self.context_file, 'r') as f:
                ctx = yaml.safe_load(f)
                if self.verbose:
                    print(f"📂 Loaded context from {self.context_file}")
                return ctx

        # Create default context if file doesn't exist
        default_context = {
            'project_name': 'QA Research Session',
            'created': datetime.now().isoformat(),
            'modulus_outer': 24,
            'modulus_inner': 9,
            'active_tuples': [],
            'experiments': [],
            'chat_history': [],
            'mcp_servers': [
                'qa-right-triangle',
                'qa-resonance',
                'qa-hgd-optimizer'
            ],
            'qa_invariants': {
                'tuple_structure': '(b, e, d, a) where d = b+e, a = b+2e',
                'core_invariants': ['J = b·d', 'K = d·a', 'X = e·d'],
                'modular_arithmetic': True
            }
        }

        # Save default context
        self.context_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.context_file, 'w') as f:
            yaml.dump(default_context, f, default_flow_style=False)

        if self.verbose:
            print(f"📝 Created new context: {self.context_file}")

        return default_context

    def save_context(self):
        """Save persistent context to YAML file"""
        self.context['last_updated'] = datetime.now().isoformat()
        with open(self.context_file, 'w') as f:
            yaml.dump(self.context, f, default_flow_style=False)

        if self.verbose:
            print(f"💾 Saved context to {self.context_file}")

    def build_system_prompt(self) -> str:
        """Build QA-aware system prompt for AI providers"""
        recent_experiments = self.context.get('experiments', [])[-3:]

        system_prompt = f"""You are a QA (Quantum Arithmetic) research assistant.

🔬 Active Project: {self.context['project_name']}
📐 Modular Arithmetic: mod-{self.context['modulus_outer']} (outer), mod-{self.context['modulus_inner']} (inner)

⚙️ QA CORE INVARIANTS (NEVER violate):
"""

        for key, value in self.context.get('qa_invariants', {}).items():
            if isinstance(value, list):
                system_prompt += f"  • {key}: {', '.join(value)}\n"
            else:
                system_prompt += f"  • {key}: {value}\n"

        system_prompt += f"""
🔌 Available MCP Tools:
  {chr(10).join(f"  • {server}" for server in self.context['mcp_servers'])}

"""

        if recent_experiments:
            system_prompt += "📊 Recent Experiments:\n"
            for exp in recent_experiments:
                system_prompt += f"  • {exp.get('name', 'Unnamed')}: {exp.get('status', 'unknown')}\n"

        system_prompt += """
⚡ Always preserve QA mathematical rigor and cite MCP tools when performing calculations.
🎯 Use precise QA terminology and respect modular arithmetic constraints.
"""

        return system_prompt

    def call_claude(self, prompt: str) -> str:
        """
        Call Claude Code (current session).
        Note: In actual implementation, this would integrate with the running Claude Code session.
        For now, returns instruction for manual execution.
        """
        return f"""[Claude Code Response - Execute in current session]

System Context:
{self.build_system_prompt()}

User Request:
{prompt}

---
NOTE: As this is a terminal agent, please manually process this request in your current Claude Code session, then add the response to the context file.
"""

    def call_codex(self, prompt: str) -> str:
        """Call OpenAI Codex via CLI (if available)"""
        if not self.providers['codex']['available']:
            return "[Codex CLI not installed. Install via: pip install openai-codex-cli]"

        try:
            full_prompt = f"{self.build_system_prompt()}\n\nUser: {prompt}"

            result = subprocess.run(
                ['codex', full_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"[Codex error: {result.stderr}]"
        except subprocess.TimeoutExpired:
            return "[Codex timeout - request took too long]"
        except Exception as e:
            return f"[Codex error: {e}]"

    def call_gemini(self, prompt: str) -> str:
        """Call Google Gemini via CLI (if available)"""
        if not self.providers['gemini']['available']:
            return "[Gemini CLI not installed. Install via: npm install -g @google/generative-ai-cli]"

        try:
            full_prompt = f"{self.build_system_prompt()}\n\nUser: {prompt}"

            result = subprocess.run(
                ['gemini', prompt],
                input=self.build_system_prompt(),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"[Gemini error: {result.stderr}]"
        except subprocess.TimeoutExpired:
            return "[Gemini timeout - request took too long]"
        except Exception as e:
            return f"[Gemini error: {e}]"

    def call_qalm(self, prompt: str) -> str:
        """Call QALM (QA Local Model) - local specialized model"""
        try:
            qalm_script = self.base_dir / 'qa_lab/qa_agents/cli/qalm.py'

            if not qalm_script.exists():
                return "[QALM not found. Expected at: qa_lab/qa_agents/cli/qalm.py]"

            # Build enhanced prompt with QA context
            enhanced_prompt = f"""{self.build_system_prompt()}

Query: {prompt}
"""

            result = subprocess.run(
                ['python3', str(qalm_script), '--interactive'],
                input=enhanced_prompt,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"[QALM error: {result.stderr}]"
        except subprocess.TimeoutExpired:
            return "[QALM timeout - request took too long]"
        except Exception as e:
            return f"[QALM error: {e}]"

    def call_mcp_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call MCP tool directly (bypasses AI providers)"""
        # Add qa_lab directory to Python path
        qa_lab_path = str(self.base_dir / 'qa_lab')
        if qa_lab_path not in sys.path:
            sys.path.insert(0, qa_lab_path)

        try:
            # Import MCP client from qa_lab
            from qa_agents.cli.mcp_client import MCPClient

            # Map tool names to servers
            server_map = {
                'qa_compute_triangle': 'qa-right-triangle',
                'qa_scan_resonance': 'qa-resonance',
                'qa_optimize_hgd': 'qa-hgd-optimizer'
            }

            server = server_map.get(tool_name, 'qa-right-triangle')
            server_path = self.base_dir / f'qa_lab/qa_mcp_servers/{server}/server.py'

            if not server_path.exists():
                return {'error': f'MCP server not found: {server}'}

            # Use context manager for MCP client
            with MCPClient(['python3', str(server_path)]) as client:
                result = client.call_tool(tool_name, arguments)
                return result

        except ImportError as e:
            return {'error': f'Failed to import MCP client: {e}. Make sure qa_lab is installed.'}
        except Exception as e:
            return {'error': str(e)}

    def orchestrate(self, user_message: str, provider: str = 'claude') -> str:
        """Orchestrate AI response with persistent context"""
        if provider not in self.providers:
            return f"[Error: Unknown provider '{provider}'. Available: {', '.join(self.providers.keys())}]"

        if not self.providers[provider]['available']:
            return f"[Error: Provider '{provider}' is not available]"

        if self.verbose:
            print(f"🤖 Calling {provider}...")

        # Call the selected provider
        response = self.providers[provider]['call'](user_message)

        # Update context with interaction
        self.context['chat_history'].append({
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'user_message': user_message,
            'response': response[:500] + '...' if len(response) > 500 else response  # Truncate for context size
        })

        # Save context
        self.save_context()

        return response

    def list_providers(self):
        """List all available AI providers"""
        print("\n🤖 Available AI Providers:")
        for name, info in self.providers.items():
            status = "✅" if info['available'] else "❌"
            print(f"  {status} {name}: {info['description']}")
        print()

    def show_context_summary(self):
        """Display current context summary"""
        print("\n📊 Current QA Research Context:")
        print(f"  Project: {self.context['project_name']}")
        print(f"  Modulus: mod-{self.context['modulus_outer']} / mod-{self.context['modulus_inner']}")
        print(f"  Active Tuples: {len(self.context.get('active_tuples', []))}")
        print(f"  Experiments: {len(self.context.get('experiments', []))}")
        print(f"  Chat History: {len(self.context.get('chat_history', []))} exchanges")
        print(f"  Context File: {self.context_file}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='QA Terminal Agent - Multi-AI Orchestration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with Claude
  %(prog)s

  # Single query to Gemini
  %(prog)s -p gemini "Analyze QA tuple (3, 5, 8, 13)"

  # Use specific context file
  %(prog)s -c qa_contexts/proton_radius.yaml

  # Call MCP tool directly
  %(prog)s --mcp qa_compute_triangle '{"b": 1, "e": 1}'
        """
    )

    parser.add_argument('message', nargs='*', help='Message to send to AI provider')
    parser.add_argument('-c', '--context',
                       default='qa_lab/qa_contexts/base_context.yaml',
                       help='Path to context YAML file')
    parser.add_argument('-p', '--provider',
                       default='claude',
                       choices=['claude', 'codex', 'gemini', 'qalm'],
                       help='AI provider to use')
    parser.add_argument('-v', '--verbose',
                       action='store_true',
                       help='Verbose output')
    parser.add_argument('--list-providers',
                       action='store_true',
                       help='List available AI providers')
    parser.add_argument('--show-context',
                       action='store_true',
                       help='Show current context summary')
    parser.add_argument('--mcp',
                       metavar='TOOL',
                       help='Call MCP tool directly (provide tool name)')
    parser.add_argument('--mcp-args',
                       metavar='JSON',
                       default='{}',
                       help='Arguments for MCP tool (JSON string)')

    args = parser.parse_args()

    # Create agent instance
    agent = QATerminalAgent(args.context, verbose=args.verbose)

    # Handle special commands
    if args.list_providers:
        agent.list_providers()
        return

    if args.show_context:
        agent.show_context_summary()
        return

    # Handle MCP tool calls
    if args.mcp:
        try:
            mcp_args = json.loads(args.mcp_args)
            result = agent.call_mcp_tool(args.mcp, mcp_args)
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in --mcp-args: {e}")
        return

    # Handle single message mode
    if args.message:
        message = ' '.join(args.message)
        response = agent.orchestrate(message, args.provider)
        print(f"\n{response}\n")
        return

    # Interactive mode
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   QA Terminal Agent - Network Chuck Inspired Multi-AI    ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"\n📁 Context: {args.context}")
    print(f"🤖 Provider: {args.provider}")
    print("\nCommands:")
    print("  /switch <provider>  - Switch AI provider")
    print("  /context            - Show context summary")
    print("  /providers          - List available providers")
    print("  /mcp <tool> <args>  - Call MCP tool")
    print("  /quit or /exit      - Exit agent")
    print()

    current_provider = args.provider

    while True:
        try:
            message = input(f"{current_provider}> ")

            if not message.strip():
                continue

            # Handle special commands
            if message.startswith('/'):
                cmd_parts = message[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                elif cmd == 'switch' and len(cmd_parts) == 2:
                    new_provider = cmd_parts[1]
                    if new_provider in agent.providers:
                        current_provider = new_provider
                        print(f"✅ Switched to {current_provider}")
                    else:
                        print(f"❌ Unknown provider: {new_provider}")
                    continue

                elif cmd == 'context':
                    agent.show_context_summary()
                    continue

                elif cmd == 'providers':
                    agent.list_providers()
                    continue

                elif cmd == 'mcp' and len(cmd_parts) == 2:
                    try:
                        tool_and_args = cmd_parts[1].split(maxsplit=1)
                        tool = tool_and_args[0]
                        tool_args = json.loads(tool_and_args[1]) if len(tool_and_args) > 1 else {}

                        result = agent.call_mcp_tool(tool, tool_args)
                        print(json.dumps(result, indent=2))
                    except Exception as e:
                        print(f"❌ MCP error: {e}")
                    continue

                else:
                    print(f"❌ Unknown command: /{cmd}")
                    continue

            # Normal message - send to current provider
            response = agent.orchestrate(message, current_provider)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
