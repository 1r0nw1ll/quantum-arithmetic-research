#!/usr/bin/env python3
"""
OpenCode Integration Agent
Allows Claude Code to converse with OpenCode and retrieve work status.
Also provides integration with Codex CLI for code generation.
"""

import subprocess
import json
import sys
import time
import importlib
from pathlib import Path
from typing import Optional, Dict, List

# GraphRAG availability check
GRAPHRAG_AVAILABLE = True  # Assume available, handle import errors in class

class CodexAgent:
    """Agent for communicating with Codex CLI"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()

    def exec_prompt(self, prompt: str, timeout: int = 60) -> Dict:
        """
        Execute a Codex prompt non-interactively.

        Args:
            prompt: The code generation prompt
            timeout: Timeout in seconds

        Returns:
            Dictionary with response data
        """
        try:
            print(f"[CodexAgent] Executing: {prompt[:100]}...", file=sys.stderr)

            result = subprocess.run(
                ["codex", "exec", prompt],
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Codex timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def generate_code(self, description: str) -> str:
        """Generate code from natural language description"""
        response = self.exec_prompt(f"Generate Python code: {description}")
        if response['success']:
            return response['output']
        return f"Error: {response.get('error', 'Unknown error')}"


class GraphRAGAgentWrapper:
    """Wrapper for GraphRAG agent integration"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.agent = None
        try:
            graphrag_module = importlib.import_module('graphrag_agent')
            GraphRAGAgentClass = getattr(graphrag_module, 'GraphRAGAgent')
            self.agent = GraphRAGAgentClass()
        except Exception as e:
            print(f"[GraphRAGAgentWrapper] Failed to initialize: {e}", file=sys.stderr)

    def query(self, query_str: str, top_k: int = 5) -> Dict:
        """Query the GraphRAG knowledge graph"""
        if not self.agent:
            return {
                'success': False,
                'error': 'GraphRAG not available',
                'results': []
            }

        try:
            response = self.agent.query(query_str, top_k=top_k)
            return {
                'success': True,
                'response': response,
                'results': response.get('results', [])
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': []
            }

    def get_entity_details(self, entity_name: str) -> Dict:
        """Get details for a specific entity"""
        if not self.agent:
            return {'success': False, 'error': 'GraphRAG not available'}

        try:
            details = self.agent.get_entity_details(entity_name)
            return {'success': True, 'details': details}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_stats(self) -> Dict:
        """Get GraphRAG statistics"""
        if not self.agent:
            return {'success': False, 'error': 'GraphRAG not available'}

        try:
            stats = self.agent.get_stats()
            return {'success': True, 'stats': stats}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class OpenCodeAgent:
    """Agent for communicating with OpenCode CLI"""

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.session_id: Optional[str] = None

    def query(self, message: str, continue_session: bool = False,
              format_type: str = "json") -> Dict:
        """
        Send a query to OpenCode and get response.

        Args:
            message: The question/prompt to send
            continue_session: Whether to continue the last session
            format_type: 'json' for structured output, 'default' for formatted

        Returns:
            Dictionary with response data
        """
        cmd = ["opencode", "run"]

        if continue_session and self.session_id:
            cmd.extend(["-s", self.session_id])
        elif continue_session:
            cmd.append("-c")

        cmd.extend(["--format", format_type])

        # Add message as positional arguments
        cmd.extend(message.split())

        try:
            print(f"[OpenCodeAgent] Sending: {message[:100]}...", file=sys.stderr)

            result = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                capture_output=True,
                text=True,
                timeout=60
            )

            error_text = None
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                error_text = stderr if stderr else stdout if stdout else None

            if format_type == "json":
                # Parse JSON events
                events = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                # Extract session ID from events
                for event in events:
                    if event.get('type') == 'session_start':
                        self.session_id = event.get('sessionId')

                return {
                    'success': result.returncode == 0,
                    'events': events,
                    'session_id': self.session_id,
                    'raw_output': result.stdout,
                    'error': error_text
                }
            else:
                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': error_text
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Query timed out after 60 seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_status(self) -> Dict:
        """Quick status check - what is OpenCode working on?"""
        return self.query(
            "What files have you been working on recently? Give a brief summary.",
            format_type="default"
        )

    def list_recent_work(self) -> Dict:
        """Get detailed list of recent work"""
        return self.query(
            "List all files you've created or modified recently with brief descriptions.",
            format_type="default"
        )

    def get_file_details(self, filepath: str) -> Dict:
        """Ask OpenCode about a specific file"""
        return self.query(
            f"Explain what {filepath} does and its current status.",
            format_type="default"
        )

    def export_sessions(self) -> List[Dict]:
        """Export OpenCode session data"""
        try:
            result = subprocess.run(
                ["opencode", "export"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            return []

        except Exception as e:
            print(f"[OpenCodeAgent] Export error: {e}", file=sys.stderr)
            return []

    def parse_response(self, response: Dict) -> str:
        """Parse OpenCode response into readable text"""
        if not response['success']:
            return f"❌ Error: {response.get('error', 'Unknown error')}"

        if 'events' in response:
            # Parse JSON events
            messages = []
            for event in response['events']:
                event_type = event.get('type')

                if event_type == 'message_chunk':
                    content = event.get('content', '')
                    if content:
                        messages.append(content)

                elif event_type == 'tool_call':
                    tool_name = event.get('tool', {}).get('name', 'unknown')
                    messages.append(f"\n[Tool: {tool_name}]\n")

                elif event_type == 'tool_result':
                    result = event.get('result', '')
                    messages.append(f"Result: {result}\n")

            return ''.join(messages)

        elif 'output' in response:
            return response['output']

        return "No response content"


def main():
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='OpenCode Integration Agent')
    parser.add_argument('query', nargs='*', help='Query to send to OpenCode')
    parser.add_argument('--continue', '-c', action='store_true',
                       dest='continue_session',
                       help='Continue last session')
    parser.add_argument('--status', action='store_true',
                       help='Get status of recent work')
    parser.add_argument('--list', action='store_true',
                       help='List recent work in detail')
    parser.add_argument('--project', default='.',
                       help='Project path')

    # GraphRAG options
    parser.add_argument('--graphrag', '-g', action='store_true',
                       help='Use GraphRAG instead of OpenCode')
    parser.add_argument('--graphrag-entity', help='Get GraphRAG entity details')
    parser.add_argument('--graphrag-stats', action='store_true',
                       help='Get GraphRAG statistics')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                       help='Number of GraphRAG results (default: 5)')

    args = parser.parse_args()

    # Handle GraphRAG commands
    if args.graphrag or args.graphrag_entity or args.graphrag_stats:
        graphrag_agent = GraphRAGAgentWrapper(args.project)

        if args.graphrag_stats:
            print("\n📊 GraphRAG Statistics:\n")
            stats_response = graphrag_agent.get_stats()
            if stats_response['success']:
                stats = stats_response['stats']
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Error: {stats_response.get('error', 'Unknown error')}")

        elif args.graphrag_entity:
            print(f"\n🔍 GraphRAG Entity Details: {args.graphrag_entity}\n")
            entity_response = graphrag_agent.get_entity_details(args.graphrag_entity)
            if entity_response['success']:
                details = entity_response['details']
                print(f"Entity: {details.get('entity', 'Unknown')}")
                print(f"QA Tuple: {details.get('qa_tuple', 'N/A')}")
                print(f"E8 Alignment: {details.get('e8_alignment', 'N/A')}")
                print(f"Definition: {details.get('definition', 'N/A')}")
                print(f"Relationships: {len(details.get('relationships', []))}")
            else:
                print(f"Error: {entity_response.get('error', 'Unknown error')}")

        elif args.query:
            query_str = ' '.join(args.query)
            print(f"\n🧠 Querying GraphRAG: {query_str}\n")
            query_response = graphrag_agent.query(query_str, top_k=args.top_k)

            if query_response['success']:
                response = query_response['response']
                print(f"Found {len(response['results'])} results in {response['processing_time']:.3f}s")
                print(f"Cache hit: {response['cache_hit']}")

                for i, result in enumerate(response['results'], 1):
                    print(f"\n{i}. {result['entity']}")
                    print(f"   QA Tuple: {result['qa_tuple']}")
                    print(f"   E8 Alignment: {result['e8_alignment']:.3f}")
                    if 'similarity' in result:
                        print(f"   Similarity: {result['similarity']:.3f}")
                    if result['definition']:
                        print(f"   Definition: {result['definition'][:150]}...")
            else:
                print(f"Error: {query_response.get('error', 'Unknown error')}")

        else:
            print("Use --graphrag with a query, or --graphrag-entity, or --graphrag-stats")
        return

    # Handle OpenCode commands
    agent = OpenCodeAgent(args.project)

    if args.status:
        print("\n📊 Getting OpenCode status...\n")
        response = agent.get_status()
    elif args.list:
        print("\n📋 Listing recent OpenCode work...\n")
        response = agent.list_recent_work()
    elif args.query:
        query = ' '.join(args.query)
        print(f"\n🤖 Querying OpenCode: {query}\n")
        response = agent.query(query, continue_session=args.continue_session)
    else:
        parser.print_help()
        return

    print("=" * 70)
    print(agent.parse_response(response))
    print("=" * 70)

    if response.get('session_id'):
        print(f"\n📝 Session ID: {response['session_id']}")


if __name__ == '__main__':
    main()
