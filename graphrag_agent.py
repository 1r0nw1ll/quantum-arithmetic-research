#!/usr/bin/env python3
"""
GraphRAG Agent for QA-based Knowledge Graph Integration
Provides context retrieval capabilities to the multi-agent research lab.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx

class GraphRAGAgent:
    """Agent for querying QA-based knowledge graph"""

    def __init__(self, graph_path: str = "qa_knowledge_graph.graphml",
                 entities_path: str = "qa_entities_merged.json",
                 encodings_path: str = "qa_entity_encodings.json"):
        """
        Initialize GraphRAG agent with knowledge graph

        Args:
            graph_path: Path to NetworkX graph file
            entities_path: Path to entities JSON
            encodings_path: Path to encodings JSON
        """
        self.graph_path = Path(graph_path)
        self.entities_path = Path(entities_path)
        self.encodings_path = Path(encodings_path)

        self.graph: Optional[nx.DiGraph] = None
        # Map entity name -> metadata (definition, symbols, ...)
        self.entities: Dict[str, Dict] = {}
        # Map entity name -> encoding record
        self.encodings: Dict[str, Dict] = {}

        self.load_data()

        # Caching for performance
        self.query_cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600  # 1 hour

    def load_data(self) -> bool:
        """Load graph, entities, and encodings"""
        try:
            # Load graph
            if self.graph_path.exists():
                self.graph = nx.read_graphml(self.graph_path)
                print(f"[GraphRAG] Loaded graph with {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
            else:
                print(f"[GraphRAG] Warning: Graph file not found: {self.graph_path}")
                return False

            # Load entities (support merged payload)
            if self.entities_path.exists():
                with open(self.entities_path, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and 'entities' in payload:
                    self.entities = {e.get('name'): e for e in payload.get('entities', []) if e.get('name')}
                elif isinstance(payload, dict):
                    self.entities = payload  # already a map
                else:
                    self.entities = {}
                print(f"[GraphRAG] Loaded {len(self.entities)} entities from {self.entities_path}")
            else:
                print(f"[GraphRAG] Warning: Entities file not found: {self.entities_path}")

            # Load encodings (list payload -> name map)
            if self.encodings_path.exists():
                with open(self.encodings_path, 'r', encoding='utf-8') as f:
                    enc_payload = json.load(f)
                if isinstance(enc_payload, dict) and 'encodings' in enc_payload:
                    self.encodings = {rec.get('name'): rec for rec in enc_payload.get('encodings', []) if rec.get('name')}
                elif isinstance(enc_payload, dict):
                    self.encodings = enc_payload
                else:
                    self.encodings = {}
                print(f"[GraphRAG] Loaded {len(self.encodings)} encodings from {self.encodings_path}")
            else:
                print(f"[GraphRAG] Warning: Encodings file not found: {self.encodings_path}")

            return True

        except Exception as e:
            print(f"[GraphRAG] Error loading data: {e}")
            return False

    def query(self, query_str: str, top_k: int = 5,
              include_context: bool = True, method: str = "hybrid") -> Dict:
        """
        Query the knowledge graph

        Args:
            query_str: Natural language query
            top_k: Number of top results to return
            include_context: Include full context chunks
            method: Query method ('tuple', 'traversal', 'ppr', 'hybrid', 'ppr_hybrid')

        Returns:
            Dictionary with query results
        """
        start_time = time.time()

        # Check cache first
        cache_key = f"{query_str}_{top_k}_{method}"
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                cached['cache_hit'] = True
                cached['processing_time'] = time.time() - start_time
                return cached

        try:
            # Use the existing qa_graph_query.py functionality
            # For now, implement basic tuple-based similarity
            results = self._query_by_tuple_similarity(query_str, top_k)

            response = {
                'query': query_str,
                'results': results,
                'processing_time': time.time() - start_time,
                'cache_hit': False,
                'method': 'tuple_similarity',
                'graph_size': {
                    'nodes': len(self.graph.nodes()) if self.graph else 0,
                    'edges': len(self.graph.edges()) if self.graph else 0
                }
            }

            # Cache result
            self.query_cache[cache_key] = {
                **response,
                'timestamp': time.time()
            }

            return response

        except Exception as e:
            return {
                'query': query_str,
                'error': str(e),
                'results': [],
                'processing_time': time.time() - start_time,
                'cache_hit': False
            }

    def _query_by_tuple_similarity(self, query_str: str, top_k: int) -> List[Dict]:
        """Hybrid query implementation: name matching + tuple similarity"""
        if not self.graph:
            return []

        results = []

        # First, try direct name matching (case-insensitive)
        query_lower = query_str.lower()
        name_matches = []

        for node in self.graph.nodes():
            node_lower = node.lower()
            if query_lower in node_lower or any(word in node_lower for word in query_lower.split()):
                # Found name match - high priority
                node_data = self.graph.nodes[node]

                # Get node QA tuple
                node_tuple = self._get_node_tuple(node_data)
                if node_tuple:
                    definition = self.entities.get(node, {}).get('definition') or str(node_data.get('definition', ''))

                    name_matches.append({
                        'entity': node,
                        'qa_tuple': node_tuple,
                        'e8_alignment': float(node_data.get('e8_alignment', 0.0)),
                        'definition': definition,
                        'similarity': 1.0,  # Perfect name match
                        'match_type': 'name',
                        'relationships': []
                    })

        # If we have name matches, prioritize them
        if name_matches:
            results.extend(name_matches[:top_k])
            remaining_slots = top_k - len(results)
            if remaining_slots > 0:
                # Add tuple similarity results for remaining slots
                tuple_results = self._get_tuple_similarity_results(query_str, remaining_slots)
                results.extend(tuple_results)
        else:
            # No name matches, fall back to tuple similarity
            results = self._get_tuple_similarity_results(query_str, top_k)

        return results[:top_k]

    def _get_node_tuple(self, node_data):
        """Extract QA tuple from node data"""
        if 'qa_tuple' in node_data:
            qa_tuple = node_data['qa_tuple']
            # Handle string-encoded tuples from GraphML
            if isinstance(qa_tuple, str):
                try:
                    return tuple(map(int, qa_tuple.strip('()').split(',')))
                except:
                    return None
            else:
                return tuple(qa_tuple) if hasattr(qa_tuple, '__iter__') else None
        return None

    def _get_tuple_similarity_results(self, query_str: str, top_k: int) -> List[Dict]:
        """Get results based on tuple similarity"""
        # Generate query tuple
        import hashlib
        h = hashlib.sha256(query_str.encode()).digest()
        query_b = int.from_bytes(h[0:4], 'big') % 24
        query_e = int.from_bytes(h[4:8], 'big') % 24
        query_d = (query_b + query_e) % 24
        query_a = (query_b + 2*query_e) % 24
        query_tuple = (query_b, query_e, query_d, query_a)

        similarities = []

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_tuple = self._get_node_tuple(node_data)

            if node_tuple and len(node_tuple) == 4:
                # Calculate tuple similarity (Euclidean distance)
                distance = sum((a - b) ** 2 for a, b in zip(query_tuple, node_tuple)) ** 0.5
                similarity = 1.0 / (1.0 + distance)  # Convert to similarity score

                definition = self.entities.get(node, {}).get('definition') or str(node_data.get('definition', ''))

                similarities.append({
                    'entity': node,
                    'qa_tuple': node_tuple,
                    'e8_alignment': float(node_data.get('e8_alignment', 0.0)),
                    'definition': definition,
                    'similarity': similarity,
                    'match_type': 'tuple',
                    'relationships': []
                })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def get_entity_details(self, entity_name: str) -> Dict:
        """Get detailed information about a specific entity"""
        if not self.graph or entity_name not in self.graph.nodes():
            return {'error': f'Entity not found: {entity_name}'}

        node_data = self.graph.nodes[entity_name]

        # Get relationships
        relationships = []
        for source, target, edge_data in self.graph.in_edges(entity_name, data=True):
            relationships.append({
                'from': source,
                'type': edge_data.get('relationship', 'related'),
                'strength': float(edge_data.get('strength', 0.0))
            })
        for source, target, edge_data in self.graph.out_edges(entity_name, data=True):
            relationships.append({
                'to': target,
                'type': edge_data.get('relationship', 'related'),
                'strength': float(edge_data.get('strength', 0.0))
            })

        return {
            'entity': entity_name,
            'qa_tuple': node_data.get('qa_tuple'),
            'e8_alignment': float(node_data.get('e8_alignment', 0.0)),
            'definition': self.entities.get(entity_name, {}).get('definition') or str(node_data.get('definition', '')),
            'relationships': relationships,
            'metadata': self.entities.get(entity_name, {})
        }

    def find_related_entities(self, entity_name: str, relationship_type: str = None,
                             top_k: int = 10) -> List[Dict]:
        """Find entities related to given entity"""
        if not self.graph or entity_name not in self.graph.nodes():
            return []

        related = []

        # Get outgoing edges
        for _, target, edge_data in self.graph.out_edges(entity_name, data=True):
            rel_type = edge_data.get('relationship', 'related')
            if relationship_type is None or rel_type == relationship_type:
                related.append({
                    'entity': target,
                    'relationship': rel_type,
                    'strength': float(edge_data.get('strength', 0.0)),
                    'direction': 'outgoing'
                })

        # Get incoming edges
        for source, _, edge_data in self.graph.in_edges(entity_name, data=True):
            rel_type = edge_data.get('relationship', 'related')
            if relationship_type is None or rel_type == relationship_type:
                related.append({
                    'entity': source,
                    'relationship': rel_type,
                    'strength': float(edge_data.get('strength', 0.0)),
                    'direction': 'incoming'
                })

        # Sort by strength and return top-k
        related.sort(key=lambda x: x['strength'], reverse=True)
        return related[:top_k]

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        if not self.graph:
            return {'error': 'Graph not loaded'}

        return {
            'nodes': len(self.graph.nodes()),
            'edges': len(self.graph.edges()),
            'entities_loaded': len(self.entities),
            'encodings_loaded': len(self.encodings),
            'cache_size': len(self.query_cache),
            'graph_path': str(self.graph_path),
            'entities_path': str(self.entities_path),
            'encodings_path': str(self.encodings_path)
        }

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        print(f"[GraphRAG] Cache cleared")

    def reload_data(self) -> bool:
        """Reload all data from disk"""
        self.clear_cache()
        return self.load_data()


def main():
    """CLI interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='GraphRAG Agent')
    parser.add_argument('query', nargs='*', help='Query to send to GraphRAG')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='Number of results')
    parser.add_argument('--entity', '-e', help='Get details for specific entity')
    parser.add_argument('--related', '-r', help='Find related entities')
    parser.add_argument('--stats', '-s', action='store_true', help='Show graph statistics')
    parser.add_argument('--reload', action='store_true', help='Reload data from disk')
    parser.add_argument('--graph', default='qa_knowledge_graph.graphml', help='Graph file path')
    parser.add_argument('--entities', default='qa_entities_merged.json', help='Entities JSON path')
    parser.add_argument('--encodings', default='qa_entity_encodings.json', help='Encodings JSON path')
    parser.add_argument('--method', default='hybrid', choices=['tuple','traversal','ppr','ppr_hybrid','hybrid'], help='Query ranking method')

    args = parser.parse_args()

    agent = GraphRAGAgent(graph_path=args.graph, entities_path=args.entities, encodings_path=args.encodings)

    if args.reload:
        print("Reloading data...")
        agent.reload_data()
        return

    if args.stats:
        stats = agent.get_stats()
        print("GraphRAG Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    if args.entity:
        details = agent.get_entity_details(args.entity)
        if 'error' in details:
            print(f"Error: {details['error']}")
        else:
            print(f"Entity: {details['entity']}")
            print(f"QA Tuple: {details['qa_tuple']}")
            print(f"E8 Alignment: {details['e8_alignment']:.3f}")
            print(f"Definition: {details['definition']}")
            print(f"Relationships: {len(details['relationships'])}")
        return

    if args.related:
        related = agent.find_related_entities(args.related, top_k=args.top_k)
        print(f"Entities related to '{args.related}':")
        for item in related:
            print(f"  {item['entity']} ({item['relationship']}, strength: {item['strength']:.3f})")
        return

    if args.query:
        query_str = ' '.join(args.query)
        print(f"Querying GraphRAG: {query_str}")
        response = agent.query(query_str, top_k=args.top_k, method=args.method)

        if 'error' in response:
            print(f"Error: {response['error']}")
            return

        print(f"Found {len(response['results'])} results in {response['processing_time']:.3f}s")
        print(f"Cache hit: {response['cache_hit']}")

        for i, result in enumerate(response['results'], 1):
            print(f"\n{i}. {result['entity']}")
            print(f"   QA Tuple: {result['qa_tuple']}")
            print(f"   E8 Alignment: {result['e8_alignment']:.3f}")
            print(f"   Similarity: {result.get('similarity', 0):.3f}")
            if result['definition']:
                print(f"   Definition: {result['definition'][:100]}...")
            if result['relationships']:
                print(f"   Relationships: {result['relationships']}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
