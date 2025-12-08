#!/usr/bin/env python3
"""
Test GraphRAG integration with multi-agent system
"""

import subprocess
import sys
import json

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_graphrag_cli():
    """Test GraphRAG CLI integration"""
    print("Testing GraphRAG CLI integration...")

    # Test stats
    success, stdout, stderr = run_command("./opencode_cli.sh graphrag-stats")
    if not success:
        print(f"❌ GraphRAG stats failed: {stderr}")
        return False
    if "nodes:" not in stdout:
        print(f"❌ Stats output malformed: {stdout}")
        return False
    print("✅ GraphRAG stats working")

    # Test Harmonic Index query
    success, stdout, stderr = run_command('./opencode_cli.sh graphrag "What is Harmonic Index?"')
    if not success:
        print(f"❌ Harmonic Index query failed: {stderr}")
        return False
    if "Harmonic Index" not in stdout:
        print(f"❌ Harmonic Index not found in results: {stdout}")
        return False
    print("✅ Harmonic Index query working")

    # Test Bell test query
    success, stdout, stderr = run_command('./opencode_cli.sh graphrag "Find Bell test experiments"')
    if not success:
        print(f"❌ Bell test query failed: {stderr}")
        return False
    bell_tests = ["CHSH", "I₃₃₂₂", "Platonic"]
    found_tests = sum(1 for test in bell_tests if test in stdout)
    if found_tests < 2:
        print(f"❌ Not enough Bell tests found ({found_tests}/3): {stdout}")
        return False
    print("✅ Bell test query working")

    return True

def test_graphrag_agent():
    """Test GraphRAG agent directly"""
    print("Testing GraphRAG agent directly...")

    try:
        from graphrag_agent import GraphRAGAgent
        agent = GraphRAGAgent()

        # Test query
        response = agent.query("Harmonic Index", top_k=3)
        if 'error' in response:
            print(f"❌ Agent query failed: {response['error']}")
            return False

        results = response.get('results', [])
        if not results:
            print("❌ No results returned")
            return False

        if results[0]['entity'] != 'Harmonic Index':
            print(f"❌ Top result not Harmonic Index: {results[0]['entity']}")
            return False

        print("✅ GraphRAG agent working")

        # Test stats
        stats = agent.get_stats()
        if 'error' in stats:
            print(f"❌ Stats failed: {stats['error']}")
            return False

        print("✅ GraphRAG stats working")

        return True

    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_opencode_integration():
    """Test OpenCode agent integration"""
    print("Testing OpenCode agent GraphRAG integration...")

    try:
        from opencode_agent import GraphRAGAgentWrapper
        wrapper = GraphRAGAgentWrapper()

        # Test query
        response = wrapper.query("QA tuple", top_k=3)
        if not response['success']:
            print(f"❌ OpenCode wrapper failed: {response.get('error', 'Unknown error')}")
            return False

        results = response.get('results', [])
        if not results:
            print("❌ No results from wrapper")
            return False

        print("✅ OpenCode GraphRAG integration working")
        return True

    except Exception as e:
        print(f"❌ OpenCode integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 Testing GraphRAG Integration\n")

    tests = [
        ("GraphRAG CLI", test_graphrag_cli),
        ("GraphRAG Agent", test_graphrag_agent),
        ("OpenCode Integration", test_opencode_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All GraphRAG integration tests PASSED!")
        return 0
    else:
        print("⚠️  Some tests failed - check integration")
        return 1

if __name__ == '__main__':
    sys.exit(main())