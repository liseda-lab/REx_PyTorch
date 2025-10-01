#!/usr/bin/env python3
"""
test_beam_parser.py
- parses test beam output with all scoring methods
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from itertools import combinations

import networkx as nx
from rdflib import URIRef


class TestBeamParser:
    """Parser for test beam output with flexible scoring methods."""
    
    def __init__(self, 
                 nodes_tsv: Path,
                 edges_tsv: Path,
                 enable_lca: bool = False,
                 ncit_pickle: Optional[Path] = None,
                 chebi_pickle: Optional[Path] = None,
                 ontology_labels: Optional[Path] = None):
        """Initialize parser with label mappings and optional LCA support."""
        self.node_labels = self._load_tsv_labels(nodes_tsv)
        self.node_types = self._load_graph_types_labels(nodes_tsv)
        self.edge_labels = self._load_tsv_labels(edges_tsv)
        
        self.enable_lca = enable_lca
        if enable_lca:
            if not all([ncit_pickle, chebi_pickle, ontology_labels]):
                raise ValueError("LCA requires ncit_pickle, chebi_pickle, and ontology_labels paths")
            self.lca_finder = LCAFinder(ncit_pickle, chebi_pickle, nodes_tsv, ontology_labels)
        else:
            self.lca_finder = None
    
    def _load_tsv_labels(self, tsv_path: Path) -> Dict[str, str]:
        """Load TSV of <code>\t<label> into a dict."""
        mapping = {}
        with tsv_path.open(encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        return mapping
    

    def _load_graph_types_labels(self, tsv_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load TSV of <id>\t<label>\t<type> into two dicts."""
        labels = {}
        types = {}
        with tsv_path.open(encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    node_id, label, node_type = parts[0], parts[1], parts[2]
                    labels[node_id] = label
                    types[node_id] = node_type
                elif len(parts) >= 2:
                    # Fallback for files without type column
                    labels[parts[0]] = parts[1]
        return  types
    
    
    # def _parse_score_and_method(self, method_line: str) -> Tuple[float, str, Dict[str, float]]:
    #     """
    #     Parse the score method line to extract score value, method type, and sub-scores.
    #     Returns: (final_score, method_type, sub_scores_dict)
    #     """
    #     sub_scores = {}
        
    #     if method_line.strip().lower() == "llm":
    #         return None, "llm", {}
        
    #     if '(llm)' in method_line.lower():
    #         # Parse format like "0.6667 (llm) v:3 c_conv:5 r:3"
    #         match = re.match(r'([\d.]+)', method_line.strip())
    #         if match:
    #             score_value = float(match.group(1))
                
    #             # Extract sub-scores
    #             # v:X for validity
    #             v_match = re.search(r'v:(\d+)', method_line)
    #             if v_match:
    #                 sub_scores['validity'] = float(v_match.group(1))
                
    #             # c_conv:X for completeness
    #             c_match = re.search(r'c_conv:(\d+)', method_line)
    #             if c_match:
    #                 sub_scores['completeness'] = float(c_match.group(1))
                
    #             # r:X for relevance
    #             r_match = re.search(r'r:(\d+)', method_line)
    #             if r_match:
    #                 sub_scores['relevance'] = float(r_match.group(1))
                
    #             return score_value, "llm", sub_scores
    #         return 0.0, "llm", {}
        
    #     # Parse standard format like "0.1000 (low_fixed)"
    #     match = re.match(r'([\d.]+)\s*\(([\w_]+)\)', method_line.strip())
    #     if match:
    #         score_value = float(match.group(1))
    #         method = match.group(2)
    #         return score_value, method, {}
        
    #     return 0.0, "none", {}
    def _parse_score_and_method(self, method_line: str) -> Tuple[float, str, Dict[str, float]]:
        """
        Parse the score method line to extract score value, method type, and sub-scores.
        Returns: (final_score, method_type, sub_scores_dict)
        """
        sub_scores = {}
        
        # Extract ic_avg if present
        ic_avg_match = re.search(r'ic_avg:([\d.]+)', method_line)
        if ic_avg_match:
            sub_scores['ic_avg'] = float(ic_avg_match.group(1))
        
        if method_line.strip().lower() == "llm":
            return None, "llm", sub_scores
        
        if '(llm)' in method_line.lower():
            # Parse format like "0.4167 (llm) ic_avg:0.577 v:2 c_conv:3 r:3"
            match = re.match(r'([\d.]+)', method_line.strip())
            if match:
                score_value = float(match.group(1))
                
                # Extract other sub-scores
                v_match = re.search(r'v:(\d+)', method_line)
                if v_match:
                    sub_scores['validity'] = float(v_match.group(1))
                
                c_match = re.search(r'c_conv:(\d+)', method_line)
                if c_match:
                    sub_scores['completeness'] = float(c_match.group(1))
                
                r_match = re.search(r'r:(\d+)', method_line)
                if r_match:
                    sub_scores['relevance'] = float(r_match.group(1))
                
                return score_value, "llm", sub_scores
            return 0.0, "llm", sub_scores
        
        # Parse standard format like "0.1000 (low_fixed) ic_avg:0.388"
        match = re.match(r'([\d.]+)\s*\(([\w_]+)\)', method_line.strip())
        if match:
            score_value = float(match.group(1))
            method = match.group(2)
            return score_value, method, sub_scores
        
        return 0.0, "none", sub_scores
    
    def _create_score_object(self, score_value: float, method: str, sub_scores: Dict[str, float]) -> Dict[str, Any]:
        """Create score object based on method type and available sub-scores."""
        if method == "llm" and sub_scores:
            score_obj = {
                "scientific_validity": sub_scores.get('validity'),
                "completeness": sub_scores.get('completeness'),
                "relevance": sub_scores.get('relevance'),
                "final_score": score_value if score_value is not None else 0.0
            }
        else:
            # For non-llm or llm without sub-scores
            score_obj = {
                "scientific_validity": None,
                "completeness": None,
                "relevance": None,
                "final_score": score_value if score_value is not None else 0.0
            }
        
        # Add ic_avg if it exists in sub_scores (for both llm and non-llm)
        if 'ic_avg' in sub_scores:
            score_obj['ic_avg'] = sub_scores['ic_avg']
        
        return score_obj
    def _parse_block(self, block: str) -> Optional[Tuple[Optional[str], Optional[Dict]]]:
        """
        Parse a single block from the test beam output.
        Returns: (pair_id, path_obj) or (None, None)
        """
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        
        pair_id = None
        
        # Handle 8-line format with pair separator and header
        if len(lines) == 8 and lines[0] == "#####################":
            # Line 0: #####################
            # Line 1: Pair ID (Compound::XX  Disease::YY)
            # Line 2: Reward:1
            # Lines 3-7: Actual 5-line path data
            pair_id = lines[1]
            lines = lines[3:8]
        elif len(lines) != 5:
            return None, None
        
        # Parse the standard 5-line format
        route_line = lines[0]
        edges_line = lines[1]
        label_line = lines[2]
        negative_score_line = lines[3]
        method_line = lines[4]
        
        # Parse label (reward)
        try:
            label = int(label_line)
        except ValueError:
            return None, None
        
        # Only process positive rewards
        if label != 1:
            return None, None
        
        # Parse score, method, and sub-scores
        score_value, method_type, sub_scores = self._parse_score_and_method(method_line)
        
        # Parse nodes and edges
        raw_nodes = route_line.split('\t')
        raw_edges = [e for e in edges_line.split('\t') if e and e != 'NO_OP']
        
        # Create nodes and edges for JSON output
        nodes = []
        for node_code in raw_nodes:
            nodes.append({
                "id": self.node_labels.get(node_code, node_code),
                "type": self._infer_node_type(node_code)
            })
        
        edges = []
        for i in range(len(raw_edges)):
            if i < len(raw_nodes) - 1:
                edges.append({
                    "source": self.node_labels.get(raw_nodes[i], raw_nodes[i]),
                    "target": self.node_labels.get(raw_nodes[i+1], raw_nodes[i+1]),
                    "label": self.edge_labels.get(raw_edges[i], raw_edges[i])
                })
        
        path_obj = {
            "score": self._create_score_object(score_value, method_type, sub_scores),
            "reasoning": f"Path extracted using {method_type} method",
            "nodes": nodes,
            "edges": edges
        }
        
        if self.enable_lca and self.lca_finder:
            lcas = self.lca_finder.compute_lcas_for_path(raw_nodes)
            if lcas:
                path_obj["lowest_common_ancestors"] = lcas
        
        return pair_id, path_obj
    
    def _infer_node_type(self, node_code: str) -> str:
        """Get node type from loaded types dictionary."""
        return self.node_types.get(node_code, "Unknown")
    
    def parse_file(self, 
                input_path: Path,
                chunk_separator: str = "___",
                pair_separator: str = "#####################") -> Dict[str, Any]:
        """Parse the entire test beam file."""
        raw_content = input_path.read_text(encoding='utf-8').strip()
        
        # First split by pair separator to ensure proper boundaries
        pair_blocks = raw_content.split(pair_separator)
        
        # Group paths by pair
        pairs_dict = defaultdict(list)
        path_id_counter = 1
        
        for pair_block in pair_blocks:
            if not pair_block.strip():
                continue
                
            # Within each pair block, find the pair ID and process chunks
            lines = pair_block.strip().split('\n')
            if len(lines) < 2:
                continue
                
            # First line should be the pair ID
            pair_id = lines[0].strip()
            if not pair_id or '\t' not in pair_id or '::' not in pair_id:
                continue
                
            # Rest of the block contains paths separated by chunk_separator
            pair_content = '\n'.join(lines[1:])
            chunks = pair_content.split(chunk_separator)
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                # Parse each path chunk
                chunk_lines = [ln.strip() for ln in chunk.strip().splitlines() if ln.strip()]
                
                # Skip if it starts with "Reward:" (pair header)
                if chunk_lines and chunk_lines[0].startswith("Reward:"):
                    chunk_lines = chunk_lines[1:]  # Skip the Reward line
                
                if len(chunk_lines) != 5:
                    continue
                
                # Parse the standard 5-line format
                route_line = chunk_lines[0]
                edges_line = chunk_lines[1]
                label_line = chunk_lines[2]
                negative_score_line = chunk_lines[3]
                method_line = chunk_lines[4]
                
                # Parse label (reward)
                try:
                    label = int(label_line)
                except ValueError:
                    continue
                
                # Only process positive rewards
                if label != 1:
                    continue
                
                # Parse score, method, and sub-scores
                score_value, method_type, sub_scores = self._parse_score_and_method(method_line)
                
                # Parse nodes and edges
                raw_nodes = route_line.split('\t')
                raw_edges = [e for e in edges_line.split('\t') if e and e != 'NO_OP']
                
                # Verify first node matches the pair
                # This is crucial - ensures paths belong to the right pair
                expected_source = pair_id.split('\t')[0]
                if raw_nodes[0] != expected_source:
                    continue  # Skip paths that don't start with the right compound
                
                # Create nodes and edges for JSON output
                nodes = []
                for node_code in raw_nodes:
                    nodes.append({
                        "id": self.node_labels.get(node_code, node_code),
                        "type": self._infer_node_type(node_code)
                    })
                
                edges = []
                for i in range(len(raw_edges)):
                    if i < len(raw_nodes) - 1:
                        edges.append({
                            "source": self.node_labels.get(raw_nodes[i], raw_nodes[i]),
                            "target": self.node_labels.get(raw_nodes[i+1], raw_nodes[i+1]),
                            "label": self.edge_labels.get(raw_edges[i], raw_edges[i])
                        })
                
                path_obj = {
                    "id": f"path_{path_id_counter}",
                    "score": self._create_score_object(score_value, method_type, sub_scores),
                    "reasoning": f"Path extracted using {method_type} method",
                    "nodes": nodes,
                    "edges": edges
                }
                
                if self.enable_lca and self.lca_finder:
                    lcas = self.lca_finder.compute_lcas_for_path(raw_nodes)
                    if lcas:
                        path_obj["lowest_common_ancestors"] = lcas
                
                path_id_counter += 1
                pairs_dict[pair_id].append(path_obj)
        
        # Convert to final format
        results = {"pairs": []}
        for pair_id, paths in pairs_dict.items():
            results["pairs"].append({
                "id": pair_id,
                "paths": paths
            })
        
        return results

class LCAFinder:
    """Compute lowest common ancestors for node pairs in ontology DAGs."""
    
    def __init__(self, ncit_pickle: Path, chebi_pickle: Path, 
                 nodes_tsv: Path, ontology_labels: Path):
        self.dag_ncit = self._load_pickle(ncit_pickle)
        self.dag_chebi = self._load_pickle(chebi_pickle)
        self.node_labels = self._load_tsv_labels(nodes_tsv)
        self.onto_labels = self._load_json(ontology_labels)
        
        self.too_general = {
            'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7057',
            'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C1909',
            'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C1908'
        }
    
    def _load_pickle(self, path: Path) -> nx.DiGraph:
        with path.open('rb') as f:
            return pickle.load(f)
    
    def _load_json(self, path: Path) -> Dict[str, str]:
        with path.open('r') as f:
            return json.load(f)
    
    def _load_tsv_labels(self, tsv_path: Path) -> Dict[str, str]:
        labels = {}
        prefix = "http://onto/"
        with tsv_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    uri, label = parts[0], parts[1]
                    labels[f"{prefix}{uri}"] = label
        return labels
    
    def compute_lcas_for_path(self, node_codes: List[str]) -> Dict[str, List[str]]:
        """Compute LCAs for all pairs in a path."""
        # Convert node codes to URIRefs
        nodes = [URIRef(f"http://onto/{code}") for code in node_codes]
        
        # Separate NCIT and CHEBI nodes
        nodes_ncit = [n for n in nodes if n in self.dag_ncit]
        nodes_chebi = [n for n in nodes if n in self.dag_chebi]
        
        # Compute pairs
        pairs_ncit = list(combinations(nodes_ncit, 2))
        pairs_chebi = list(combinations(nodes_chebi, 2))
        
        # Get LCAs
        lcas = defaultdict(set)
        
        if pairs_ncit:
            # Convert generator to dictionary
            ncit_lcas = dict(nx.all_pairs_lowest_common_ancestor(self.dag_ncit, pairs_ncit))
            for pair, anc in ncit_lcas.items():
                if anc and str(anc) not in self.too_general:
                    lcas[pair].add(anc)
        
        if pairs_chebi:
            # Convert generator to dictionary
            chebi_lcas = dict(nx.all_pairs_lowest_common_ancestor(self.dag_chebi, pairs_chebi))
            for pair, anc in chebi_lcas.items():
                if anc and str(anc) not in self.too_general:
                    lcas[pair].add(anc)
        
        # Convert to labeled format
        result = {}
        for (n1, n2), anc_set in lcas.items():
            # Get labels for the nodes
            n1_label = self.node_labels.get(str(n1), str(n1))
            n2_label = self.node_labels.get(str(n2), str(n2))
            key = f"{n1_label},{n2_label}"
            
            # Get labels for ancestors
            result[key] = [
                self.onto_labels.get(str(anc), str(anc))
                for anc in anc_set
            ]
        
        return result


# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Parse test beam output files")
#     parser.add_argument("input", type=Path, help="Input test beam file")
#     parser.add_argument("--nodes-tsv", type=Path, required=True, help="Node labels TSV")
#     parser.add_argument("--edges-tsv", type=Path, required=True, help="Edge labels TSV")
#     parser.add_argument("--enable-lca", action="store_true", help="Enable LCA computation")
#     parser.add_argument("--ncit-pickle", type=Path, help="NCIT DAG pickle")
#     parser.add_argument("--chebi-pickle", type=Path, help="CHEBI DAG pickle")
#     parser.add_argument("--onto-labels", type=Path, help="Ontology labels JSON")
#     parser.add_argument("--output", type=Path, help="Output JSON file")
    
#     args = parser.parse_args()
    
#     beam_parser = TestBeamParser(
#         nodes_tsv=args.nodes_tsv,
#         edges_tsv=args.edges_tsv,
#         enable_lca=args.enable_lca,
#         ncit_pickle=args.ncit_pickle,
#         chebi_pickle=args.chebi_pickle,
#         ontology_labels=args.onto_labels
#     )
    
#     print(f"Parsing {args.input}...")
#     results = beam_parser.parse_file(args.input)
    
#     # Count methods
#     method_counts = defaultdict(int)
#     for pair in results['pairs']:
#         for path in pair['paths']:
#             if 'reasoning' in path:
#                 method = path['reasoning'].replace('Path extracted using ', '').replace(' method', '')
#                 method_counts[method] += 1
    
#     print(f"\nParsing complete:")
#     print(f"  Total pairs: {len(results['pairs'])}")
#     print(f"  Total paths: {sum(len(p['paths']) for p in results['pairs'])}")
#     print(f"\nMethod distribution:")
#     for method, count in sorted(method_counts.items()):
#         print(f"  {method}: {count}")
    
#     output_path = args.output or args.input.with_suffix('.json')
#     with output_path.open('w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
    
#     print(f"\nOutput saved to: {output_path}")




def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse test beam output files")
    parser.add_argument("input", type=Path, help="Input test beam file")
    parser.add_argument("--nodes-tsv", type=Path, required=True, help="Node labels TSV")
    parser.add_argument("--edges-tsv", type=Path, required=True, help="Edge labels TSV")
    parser.add_argument("--enable-lca", action="store_true", help="Enable LCA computation")
    parser.add_argument("--ncit-pickle", type=Path, help="NCIT DAG pickle")
    parser.add_argument("--chebi-pickle", type=Path, help="CHEBI DAG pickle")
    parser.add_argument("--onto-labels", type=Path, help="Ontology labels JSON")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    
    args = parser.parse_args()
    
    beam_parser = TestBeamParser(
        nodes_tsv=args.nodes_tsv,
        edges_tsv=args.edges_tsv,
        enable_lca=args.enable_lca,
        ncit_pickle=args.ncit_pickle,
        chebi_pickle=args.chebi_pickle,
        ontology_labels=args.onto_labels
    )
    
    print(f"Parsing {args.input}...")
    results = beam_parser.parse_file(args.input)
    
    # Count methods
    method_counts = defaultdict(int)
    for pair in results['pairs']:
        for path in pair['paths']:
            if 'reasoning' in path:
                method = path['reasoning'].replace('Path extracted using ', '').replace(' method', '')
                method_counts[method] += 1
    
    print(f"\nParsing complete:")
    print(f"  Total pairs: {len(results['pairs'])}")
    print(f"  Total paths: {sum(len(p['paths']) for p in results['pairs'])}")
    print(f"\nMethod distribution:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")
    
    # Save full results
    output_path = args.output or args.input.with_suffix('.json')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull output saved to: {output_path}")
    
    # Filter and process LLM-only results
    llm_pairs_dict = {}
    llm_results = {"pairs": []}
    
    for pair_data in results['pairs']:
        pair_id = pair_data['id']
        llm_paths = []
        
        # Filter for LLM paths only
        for path in pair_data['paths']:
            if 'llm' in path.get('reasoning', '').lower():
                score = path['score']['final_score']
                # Create path string representation
                path_nodes = [node['id'] for node in path['nodes']]
                path_str = ' -> '.join(path_nodes)
                llm_paths.append((path_str, score, path))
        
        # If this pair has LLM paths, add to results
        if llm_paths:
            # Sort by score (descending)
            llm_paths.sort(key=lambda x: x[1], reverse=True)
            
            # Add to dictionary (without the full path object)
            llm_pairs_dict[pair_id] = [(path_str, score) for path_str, score, _ in llm_paths]
            
            # Add to JSON results (with full path objects)
            llm_pair_data = {
                "id": pair_id,
                "paths": [path_obj for _, _, path_obj in llm_paths]
            }
            llm_results["pairs"].append(llm_pair_data)
    
    # Save LLM-only results
    if llm_pairs_dict:
        # Save dictionary as JSON
        llm_dict_path = args.output.parent / f"{args.output.stem}_llm_dict.json" if args.output else args.input.with_suffix('_llm_dict.json')
        with llm_dict_path.open('w', encoding='utf-8') as f:
            json.dump(llm_pairs_dict, f, indent=2, ensure_ascii=False)
        print(f"\nLLM dictionary saved to: {llm_dict_path}")
        
        # Save as text file
        llm_txt_path = args.output.parent / f"{args.output.stem}_llm.txt" if args.output else args.input.with_suffix('_llm.txt')
        with llm_txt_path.open('w', encoding='utf-8') as f:
            for pair_id, path_scores in llm_pairs_dict.items():
                f.write(f"{pair_id}\n")
                for path_str, score in path_scores:
                    f.write(f"{score:.4f}\t{path_str}\n")
                f.write("\n")
        print(f"LLM text file saved to: {llm_txt_path}")
        
        # Save full LLM JSON in same format as original
        llm_json_path = args.output.parent / f"{args.output.stem}_llm_only.json" if args.output else args.input.with_suffix('_llm_only.json')
        with llm_json_path.open('w', encoding='utf-8') as f:
            json.dump(llm_results, f, indent=2, ensure_ascii=False)
        print(f"LLM-only JSON saved to: {llm_json_path}")
        
        print(f"\nLLM-only statistics:")
        print(f"  Pairs with LLM paths: {len(llm_pairs_dict)}")
        print(f"  Total LLM paths: {sum(len(paths) for paths in llm_pairs_dict.values())}")
    else:
        print("\nNo LLM paths found in the results.")

if __name__ == "__main__":
    main()
