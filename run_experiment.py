
import argparse
import json
import os
import random
import re
import heapq
from collections import Counter, defaultdict, deque

# ============================================================
# Graph
# ============================================================
class Graph:
    def __init__(self):
        self.adj = defaultdict(set)
        self.edge_set = set()
        self.nodes_set = set()

    def add_edge(self, u, v):
        if u == v:
            self.nodes_set.add(u)
            return
        self.nodes_set.add(u)
        self.nodes_set.add(v)
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.edge_set.add(tuple(sorted((u, v))))

    def nodes(self):
        return list(self.nodes_set)

    def edges(self):
        return list(self.edge_set)

    def neighbors(self, u):
        return self.adj.get(u, set())

    def degree(self, u):
        return len(self.adj.get(u, set()))

    def induced_edges(self, node_subset):
        node_subset = set(node_subset)
        return [e for e in self.edge_set if e[0] in node_subset and e[1] in node_subset]

    def shortest_path(self, src, dst):
        if src == dst:
            return [src]
        q = deque([src])
        parent = {src: None}
        while q:
            u = q.popleft()
            for v in self.adj.get(u, []):
                if v not in parent:
                    parent[v] = u
                    if v == dst:
                        path = [dst]
                        cur = dst
                        while parent[cur] is not None:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        return path
                    q.append(v)
        return []

# ============================================================
# Loading
# ============================================================
def normalize_uri_token(tok):
    tok = tok.strip().rstrip(".")
    if tok.startswith("<") and tok.endswith(">"):
        tok = tok[1:-1]
    prefix_map = {
        "http://dbpedia.org/ontology/": "dbpo:",
        "https://dbpedia.org/ontology/": "dbpo:",
        "http://dbpedia.org/property/": "dbprop:",
        "https://dbpedia.org/property/": "dbprop:",
        "http://dbpedia.org/resource/": "dbpr:",
        "https://dbpedia.org/resource/": "dbpr:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "https://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
        "https://www.w3.org/2000/01/rdf-schema#": "rdfs:",
        "http://www.w3.org/2002/07/owl#": "owl:",
        "https://www.w3.org/2002/07/owl#": "owl:",
        "http://purl.org/dc/terms/": "dct:",
        "https://purl.org/dc/terms/": "dct:",
        "http://xmlns.com/foaf/0.1/": "foaf:",
        "https://xmlns.com/foaf/0.1/": "foaf:",
        "http://www.w3.org/2003/01/geo/wgs84_pos#": "pos:",
        "https://www.w3.org/2003/01/geo/wgs84_pos#": "pos:",
        "http://www.wikidata.org/entity/": "wde:",
        "https://www.wikidata.org/entity/": "wde:",
        "http://www.opengis.net/ont/geosparql#": "gsp:",
        "https://www.opengis.net/ont/geosparql#": "gsp:",
    }
    for base, pref in prefix_map.items():
        if tok.startswith(base):
            return pref + tok[len(base):]
    return tok


def is_literal_token(token):
    if not token:
        return False
    token = token.strip()
    if token.startswith('"') or token.startswith("'"):
        return True
    if token.lower() in {"true", "false"}:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def is_blank_node(token):
    return bool(token) and token.startswith("_:")


def is_system_token(token):
    if not token:
        return False
    return token.startswith(("rdf:", "rdfs:", "owl:", "xsd:", "foaf:", "dct:", "pos:", "gsp:", "geo:", "skos:", "bif:"))


def is_schema_node(token, allow_system_nodes=False):
    if not token:
        return False
    if is_literal_token(token) or is_blank_node(token):
        return False
    if token.startswith(("http://", "https://")):
        return False
    if token.startswith(("dbpo:", "dbo:", "dbprop:", "dbpr:")):
        return True
    if allow_system_nodes and is_system_token(token):
        return True
    return False


def parse_schema_line(line):
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) >= 3:
        subj = normalize_uri_token(parts[0])
        pred = normalize_uri_token(parts[1])
        obj = normalize_uri_token(" ".join(parts[2:]).strip())
        obj = normalize_uri_token(obj)
        return subj, pred, obj
    if len(parts) == 2:
        u = normalize_uri_token(parts[0])
        v = normalize_uri_token(parts[1])
        return u, None, v
    return None


def load_schema(path, clean_schema=True, keep_predicate_nodes=False, allow_system_nodes=False):
    g = Graph()
    skipped_lines = 0
    skipped_non_node = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            triple = parse_schema_line(line)
            if triple is None:
                skipped_lines += 1
                continue
            u, pred, v = triple
            if pred is None:
                if is_schema_node(u, allow_system_nodes=allow_system_nodes) and is_schema_node(v, allow_system_nodes=allow_system_nodes):
                    g.add_edge(u, v)
                else:
                    skipped_non_node += 1
                continue
            if not clean_schema:
                g.nodes_set.add(u)
                if pred is not None:
                    g.nodes_set.add(pred)
                g.nodes_set.add(v)
                if pred is not None:
                    g.add_edge(u, pred)
                    g.add_edge(pred, v)
                else:
                    g.add_edge(u, v)
                continue
            subj_ok = is_schema_node(u, allow_system_nodes=allow_system_nodes)
            obj_ok = is_schema_node(v, allow_system_nodes=allow_system_nodes)
            if not (subj_ok and obj_ok):
                skipped_non_node += 1
                continue
            g.add_edge(u, v)
            if keep_predicate_nodes and pred and is_schema_node(pred, allow_system_nodes=allow_system_nodes):
                g.add_edge(u, pred)
                g.add_edge(pred, v)
    g.schema_stats = {
        "skipped_lines": skipped_lines,
        "skipped_non_node": skipped_non_node,
        "clean_schema": clean_schema,
        "keep_predicate_nodes": keep_predicate_nodes,
        "allow_system_nodes": allow_system_nodes,
    }
    return g

# ============================================================
# Query model
# ============================================================
def query_from_nodes(nodes, qtype, pattern, weight=1.0, contains_rare=False):
    return {
        "nodes": set(nodes),
        "type": qtype,
        "pattern": pattern,
        "weight": float(weight),
        "contains_rare": bool(contains_rare),
    }

# ============================================================
# Workload generation
# ============================================================
def generate_simple_workload(nodes, n=100, qsize=3, seed=42):
    rng = random.Random(seed)
    nodes = list(nodes)
    workload = []
    if not nodes:
        return workload
    for _ in range(n):
        size = min(qsize, len(nodes))
        chosen = rng.sample(nodes, size)
        pattern = "SELECT * WHERE { " + " . ".join(f"?x relatedTo {c}" for c in chosen) + " }"
        workload.append(query_from_nodes(chosen, "simple", pattern))
    return workload

def random_walk_path(graph, start, length, rng):
    path = [start]
    current = start
    visited = {start}
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        unseen = [n for n in neighbors if n not in visited]
        nxt = rng.choice(unseen if unseen else neighbors)
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path

def path_to_pattern(path, rare_nodes=None):
    rare_nodes = set(rare_nodes or [])
    if not path:
        return "SELECT * WHERE { }"
    triples = []
    for i, node in enumerate(path):
        tag = " [RARE]" if node in rare_nodes else ""
        if i == 0:
            triples.append(f"{node}{tag}")
        else:
            triples.append(f"--p{i}--> {node}{tag}")
    return "PATH { " + " ".join(triples) + " }"

def generate_structured_workload(graph, n=100, min_len=2, max_len=4, seed=42, forced_start_nodes=None, rare_nodes=None, rare_weight=1.0):
    rng = random.Random(seed)
    nodes = graph.nodes()
    workload = []
    if not nodes:
        return workload
    forced_start_nodes = list(forced_start_nodes or [])
    rare_nodes = set(rare_nodes or [])
    for i in range(n):
        if forced_start_nodes:
            start = forced_start_nodes[i % len(forced_start_nodes)]
        else:
            start = rng.choice(nodes)
        plen = rng.randint(min_len, max_len)
        path = random_walk_path(graph, start, plen, rng)
        contains_rare = any(n in rare_nodes for n in path)
        qtype = "rare_structured" if contains_rare else "structured"
        weight = rare_weight if contains_rare and rare_weight > 1.0 else 1.0
        workload.append(query_from_nodes(path, qtype, path_to_pattern(path, rare_nodes=rare_nodes), weight=weight, contains_rare=contains_rare))
    return workload

def generate_mixed_workload(graph, n=100, qsize=3, min_len=2, max_len=4, structured_ratio=0.5, seed=42, forced_rare_queries=0, rare_nodes=None, rare_weight=1.0):
    rng = random.Random(seed)
    rare_nodes = list(rare_nodes or [])
    forced_rare_queries = max(0, min(n, int(forced_rare_queries)))
    normal_n = n - forced_rare_queries
    simple_n = int(round(normal_n * (1.0 - structured_ratio)))
    structured_n = normal_n - simple_n
    simple = generate_simple_workload(graph.nodes(), n=simple_n, qsize=qsize, seed=seed)
    structured = generate_structured_workload(graph, n=structured_n, min_len=min_len, max_len=max_len, seed=seed + 1000)
    forced = []
    if forced_rare_queries > 0 and rare_nodes:
        forced_structured_n = max(1, int(round(forced_rare_queries * structured_ratio)))
        forced_simple_n = forced_rare_queries - forced_structured_n
        forced.extend(generate_structured_workload(
            graph,
            n=forced_structured_n,
            min_len=min_len,
            max_len=max_len,
            seed=seed + 2000,
            forced_start_nodes=rare_nodes,
            rare_nodes=rare_nodes,
            rare_weight=rare_weight,
        ))
        for i in range(forced_simple_n):
            chosen = set()
            chosen.add(rare_nodes[i % len(rare_nodes)])
            target_size = min(qsize, len(list(graph.nodes())))
            candidates = [n for n in graph.nodes() if n not in chosen]
            local_rng = random.Random(seed + 3000 + i)
            while len(chosen) < target_size and candidates:
                nxt = local_rng.choice(candidates)
                chosen.add(nxt)
                candidates.remove(nxt)
            chosen = sorted(chosen)
            contains_rare = any(n in set(rare_nodes) for n in chosen)
            pattern = "SELECT * WHERE { " + " . ".join(
                f"?x relatedTo {c}{' [RARE]' if c in set(rare_nodes) else ''}" for c in chosen
            ) + " }"
            forced.append(query_from_nodes(chosen, "rare_simple", pattern, weight=rare_weight, contains_rare=contains_rare))
    workload = simple + structured + forced
    rng.shuffle(workload)
    return workload

def identify_rare_nodes(graph, rare_fraction=0.10, rare_degree_threshold=2):
    nodes = graph.nodes()
    if not nodes:
        return set()
    degree_pairs = sorted(((n, graph.degree(n)) for n in nodes), key=lambda t: (t[1], t[0]))
    fraction_count = max(1, int(round(len(nodes) * rare_fraction)))
    fraction_rare = {n for n, _ in degree_pairs[:fraction_count]}
    threshold_rare = {n for n, d in degree_pairs if d <= rare_degree_threshold}
    rare_nodes = fraction_rare | threshold_rare
    if not rare_nodes:
        rare_nodes = fraction_rare
    return rare_nodes

def generate_rare_aware_workload(
    graph,
    n=100,
    qsize=3,
    min_len=2,
    max_len=4,
    structured_ratio=0.7,
    rare_query_ratio=0.25,
    rare_fraction=0.10,
    rare_degree_threshold=2,
    rare_weight=2.0,
    seed=42
):
    """
    Creates a workload with a controlled fraction of rare-node queries.
    - rare nodes are identified primarily by low degree
    - some queries are forced to include at least one rare node
    - those queries can receive higher weight
    """
    rng = random.Random(seed)
    nodes = graph.nodes()
    if not nodes:
        return [], set()

    rare_nodes = identify_rare_nodes(graph, rare_fraction=rare_fraction, rare_degree_threshold=rare_degree_threshold)
    common_nodes = list(set(nodes) - set(rare_nodes))
    rare_nodes_list = list(rare_nodes)

    rare_count = int(round(n * rare_query_ratio))
    normal_count = max(0, n - rare_count)

    workload = []

    # Normal portion
    if normal_count > 0:
        workload.extend(
            generate_mixed_workload(
                graph,
                n=normal_count,
                qsize=qsize,
                min_len=min_len,
                max_len=max_len,
                structured_ratio=structured_ratio,
                seed=seed
            )
        )

    # Rare-focused portion
    for _ in range(rare_count):
        use_structured = rng.random() < structured_ratio and bool(rare_nodes_list)
        if use_structured:
            start = rng.choice(rare_nodes_list)
            plen = rng.randint(min_len, max_len)
            path = random_walk_path(graph, start, plen, rng)
            # if walk was too short, enrich with common nodes when possible
            if len(path) < 2 and common_nodes:
                path.append(rng.choice(common_nodes))
            contains_rare = any(n in rare_nodes for n in path)
            pattern = path_to_pattern(path, rare_nodes=rare_nodes)
            workload.append(query_from_nodes(
                path,
                "rare_structured",
                pattern,
                weight=rare_weight,
                contains_rare=contains_rare
            ))
        else:
            chosen = set()
            if rare_nodes_list:
                chosen.add(rng.choice(rare_nodes_list))
            target_size = min(qsize, len(nodes))
            candidates = list(set(nodes) - chosen)
            while len(chosen) < target_size and candidates:
                nxt = rng.choice(candidates)
                chosen.add(nxt)
                candidates.remove(nxt)
            contains_rare = any(n in rare_nodes for n in chosen)
            pattern = "SELECT * WHERE { " + " . ".join(
                f"?x relatedTo {c}{' [RARE]' if c in rare_nodes else ''}" for c in sorted(chosen)
            ) + " }"
            workload.append(query_from_nodes(
                chosen,
                "rare_simple",
                pattern,
                weight=rare_weight,
                contains_rare=contains_rare
            ))

    rng.shuffle(workload)
    return workload, rare_nodes



def extract_nodes_from_sparql(query_text, include_variables=False):
    """
    Extract graph-like tokens from the SPARQL query body only.

    Important fixes:
    - ignores PREFIX declaration lines so namespace stubs like dbpo: and rdf:
      are not treated as workload nodes
    - keeps dotted DBpedia-style resource names intact, e.g. dbpr:S.C._Nampula
    - normalizes full IRIs to the same prefixed style used by the schema loader
    - rejects bare namespace prefixes such as rdf: or dbpo:
    """
    tokens = set()
    if not query_text:
        return tokens

    # Remove PREFIX declarations entirely before token extraction.
    body = re.sub(
        r"(?im)^\s*PREFIX\s+[A-Za-z_][\w-]*:\s*<[^>]+>\s*$",
        "",
        query_text,
    )

    # Prefixed names used in the query body. Allow dots and other common
    # DBpedia resource-name characters after the colon.
    prefixed = re.findall(
        r"\b[A-Za-z_][\w-]*:[:A-Za-z0-9._()%+\-]+\b",
        body,
    )

    for t in prefixed:
        # Reject bare namespace declarations like rdf: or dbpo:
        if re.fullmatch(r"[A-Za-z_][\w-]*:", t):
            continue
        tokens.add(t)

    # Full IRIs inside angle brackets.
    uris = re.findall(r"<([^>]+)>", body)
    for u in uris:
        norm = normalize_uri_token(u)
        if re.fullmatch(r"[A-Za-z_][\w-]*:", norm):
            continue
        tokens.add(norm)

    if include_variables:
        tokens.update(re.findall(r"\?[A-Za-z_][\w]*", body))

    banned = {"rdf:", "rdfs:", "owl:", "foaf:", "dct:", "dbpo:", "dbpr:", "dbprop:", "pos:", "bif:"}
    return {
        t for t in tokens
        if t
        and t not in banned
        and not re.fullmatch(r"[A-Za-z_][\w-]*:", t)
        and not t.lower().startswith("http://www.w3.org/2001/xmlschema#")
    }


def _iter_sparql_log_entries(data):
    if isinstance(data, dict):
        bindings = data.get('results', {}).get('bindings', [])
        for entry in bindings:
            if isinstance(entry, dict):
                text_obj = entry.get('text')
                if isinstance(text_obj, dict) and 'value' in text_obj:
                    yield str(text_obj['value'])
                elif isinstance(text_obj, str):
                    yield text_obj
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                if 'text' in item and isinstance(item['text'], str):
                    yield item['text']
                elif isinstance(item.get('text'), dict) and 'value' in item['text']:
                    yield str(item['text']['value'])
                elif 'query' in item:
                    yield str(item['query'])


def load_sparql_json_workload(
    path,
    graph=None,
    include_variables=False,
    min_nodes=2,
    require_graph_overlap=False,
    drop_generic_queries=False,
    graph_overlap_weight=2.0,
):
    """
    Load a real SPARQL log from JSON and convert each query into the internal
    workload representation used by WBSum / GA.

    If a graph is provided, tokens can be filtered to those present in the graph.
    This is usually the safest mode because synthetic query variables and
    unrelated vocabulary will otherwise dominate the workload.
    """
    workload = []
    graph_nodes = set(graph.nodes()) if graph is not None else None

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)

    for query_text in _iter_sparql_log_entries(data):
        raw_nodes = extract_nodes_from_sparql(query_text, include_variables=include_variables)
        raw_nodes = {n for n in raw_nodes if is_schema_node(n)}
        nodes = set(raw_nodes)

        if graph_nodes is not None:
            overlap = {n for n in (raw_nodes & graph_nodes) if is_schema_node(n)}
            if overlap:
                nodes = overlap
            elif require_graph_overlap:
                continue

        qnorm = ' '.join(query_text.split()).lower()
        is_generic = qnorm in {
            'select * where { ?s ?p ?o }',
            'select * where { ?subject ?predicate ?object }',
            'select * where { ?x ?y ?z }',
            'select * where { ?a ?b ?c }',
        }
        if drop_generic_queries and is_generic:
            continue

        if len(nodes) < max(1, int(min_nodes)):
            continue

        weight = float(max(1, len(nodes)))
        contains_rare = False
        if graph_nodes is not None and raw_nodes & graph_nodes:
            weight *= float(graph_overlap_weight)

        workload.append(query_from_nodes(
            nodes,
            'real_log',
            query_text.strip(),
            weight=weight,
            contains_rare=contains_rare,
        ))

    return workload


def deduplicate_workload(workload):
    """Merge repeated queries with the same node signature by summing weights."""
    merged = {}
    order = []
    for q in workload:
        key = tuple(sorted(q["nodes"]))
        if key not in merged:
            merged[key] = {
                "nodes": set(q["nodes"]),
                "type": q.get("type", "loaded"),
                "pattern": q.get("pattern", ""),
                "weight": float(q.get("weight", 1.0)),
                "contains_rare": bool(q.get("contains_rare", False)),
            }
            order.append(key)
        else:
            merged[key]["weight"] += float(q.get("weight", 1.0))
            merged[key]["contains_rare"] = merged[key]["contains_rare"] or bool(q.get("contains_rare", False))
    return [merged[k] for k in order]


def connected_components_all(graph):
    return connected_components(graph, graph.nodes())


def node_to_component_map(graph):
    comps = connected_components_all(graph)
    mapping = {}
    for idx, comp in enumerate(comps):
        for n in comp:
            mapping[n] = idx
    return mapping, comps


def component_subgraph_nodes_for_query(qnodes, node_component):
    buckets = defaultdict(set)
    for n in qnodes:
        cid = node_component.get(n)
        if cid is not None:
            buckets[cid].add(n)
    return buckets


def restrict_workload_to_component(workload, component_nodes, node_component=None, component_id=None):
    component_nodes = set(component_nodes)
    restricted = []
    for q in workload:
        qnodes = set(q["nodes"])
        if component_id is not None and node_component is not None:
            buckets = component_subgraph_nodes_for_query(qnodes, node_component)
            kept = buckets.get(component_id, set())
        else:
            kept = qnodes & component_nodes
        if not kept:
            continue
        ratio = len(kept) / max(1, len(qnodes))
        new_q = dict(q)
        new_q["nodes"] = kept
        new_q["weight"] = float(q.get("weight", 1.0)) * ratio
        restricted.append(new_q)
    return deduplicate_workload(restricted)



def induced_edge_count(graph, nodes):
    return len(graph.induced_edges(nodes))


def connectivity_penalty(graph, nodes):
    return max(0, len(connected_components(graph, nodes)) - 1)


def workload_node_set(workload):
    nodes = set()
    for q in workload:
        nodes.update(q.get("nodes", set()))
    return nodes


def node_workload_weight(workload):
    weights = Counter()
    for q in workload:
        w = float(q.get("weight", 1.0))
        for n in q.get("nodes", set()):
            weights[n] += w
    return weights


def dominant_workload_nodes(workload, frac=0.15, min_weight=0.0):
    weights = node_workload_weight(workload)
    if not weights:
        return set()
    max_w = max(weights.values())
    threshold = max(float(min_weight), float(frac) * max_w)
    return {n for n, w in weights.items() if w >= threshold}


def node_domain_penalty_details(nodes, workload, frac=0.15, min_weight=0.0):
    nodes = set(nodes)
    dominant = dominant_workload_nodes(workload, frac=frac, min_weight=min_weight)
    if not dominant:
        return {
            "dominant_nodes": set(),
            "off_domain_nodes": set(),
            "dominant_ratio": 0.0,
            "weighted_drift": 0.0,
        }
    weights = node_workload_weight(workload)
    max_w = max(weights.values()) if weights else 1.0
    off_domain = {n for n in nodes if n not in dominant}
    weighted_drift = 0.0
    for n in off_domain:
        weighted_drift += max(0.0, 1.0 - (weights.get(n, 0.0) / max(1e-9, max_w)))
    return {
        "dominant_nodes": dominant,
        "off_domain_nodes": off_domain,
        "dominant_ratio": len([n for n in nodes if n in dominant]) / max(1, len(nodes)),
        "weighted_drift": weighted_drift,
    }


def relevance_score(nodes, workload):
    nodes = set(nodes)
    covered = 0.0
    total = 0.0
    for q in workload:
        weight = float(q.get("weight", 1.0))
        total += weight
        if set(q.get("nodes", set())).issubset(nodes):
            covered += weight
    return covered / total if total else 0.0


def useless_node_penalty(nodes, workload):
    workload_nodes = workload_node_set(workload)
    return len([n for n in set(nodes) if n not in workload_nodes])


def domain_penalty(nodes, workload, dominant_frac=0.15, min_node_workload_weight=0.0):
    details = node_domain_penalty_details(
        nodes,
        workload,
        frac=dominant_frac,
        min_weight=min_node_workload_weight,
    )
    return len(details["off_domain_nodes"])


def summary_objectives(
    graph,
    nodes,
    workload,
    budget,
    structured_bonus=0.1,
    seed_nodes=None,
    dominant_frac=0.15,
    min_node_workload_weight=0.0,
):
    nodes = set(nodes)
    coverage = evaluate_coverage(nodes, workload, structured_bonus=structured_bonus)
    rare_coverage = evaluate_rare_query_coverage(nodes, workload)
    relevance = relevance_score(nodes, workload)
    edge_density = induced_edge_count(graph, nodes) / max(1, len(nodes))
    overflow_ratio = max(0, len(nodes) - budget) / max(1, budget)
    connector_ratio = 0.0
    if seed_nodes is not None:
        connector_ratio = max(0, len(nodes) - len(set(seed_nodes))) / max(1, budget)
    component_pen = connectivity_penalty(graph, nodes)
    useless_pen = useless_node_penalty(nodes, workload)
    domain_details = node_domain_penalty_details(
        nodes,
        workload,
        frac=dominant_frac,
        min_weight=min_node_workload_weight,
    )
    domain_pen = len(domain_details["off_domain_nodes"])
    return {
        "coverage": coverage,
        "rare_coverage": rare_coverage,
        "relevance": relevance,
        "edge_density": edge_density,
        "overflow_ratio": overflow_ratio,
        "connector_ratio": connector_ratio,
        "component_penalty": component_pen,
        "useless_penalty": useless_pen,
        "domain_penalty": domain_pen,
        "dominant_ratio": domain_details["dominant_ratio"],
        "weighted_domain_drift": domain_details["weighted_drift"],
        "off_domain_nodes": sorted(domain_details["off_domain_nodes"]),
        "dominant_nodes_count": len(domain_details["dominant_nodes"]),
    }


def summary_fitness(
    graph,
    nodes,
    workload,
    budget,
    structured_bonus=0.1,
    rare_bonus=0.15,
    overflow_weight=0.15,
    connector_weight=0.20,
    edge_bonus_weight=0.05,
    component_penalty_weight=0.5,
    relevance_weight=0.75,
    useless_penalty_weight=0.10,
    domain_penalty_weight=0.45,
    dominant_frac=0.15,
    min_node_workload_weight=0.0,
    strict_connectivity=False,
    connectivity_hard_penalty=2.0,
    seed_nodes=None,
    return_parts=False,
):
    parts = summary_objectives(
        graph,
        nodes,
        workload,
        budget,
        structured_bonus=structured_bonus,
        seed_nodes=seed_nodes,
        dominant_frac=dominant_frac,
        min_node_workload_weight=min_node_workload_weight,
    )
    normalized_domain_penalty = parts["domain_penalty"] / max(1, len(set(nodes)))
    disconnected = parts["component_penalty"] > 0
    if strict_connectivity and disconnected:
        score = -1_000_000_000.0
    else:
        score = (
            2.50 * parts["coverage"]
            + relevance_weight * parts["relevance"]
            + 0.60 * parts["rare_coverage"]
            + edge_bonus_weight * parts["edge_density"]
            - overflow_weight * parts["overflow_ratio"]
            - connector_weight * parts["connector_ratio"]
            - component_penalty_weight * parts["component_penalty"]
            - useless_penalty_weight * parts["useless_penalty"]
            - domain_penalty_weight * normalized_domain_penalty
        )
    if return_parts:
        out = dict(parts)
        out["normalized_domain_penalty"] = normalized_domain_penalty
        out["strict_connectivity"] = bool(strict_connectivity)
        out["disconnected"] = bool(disconnected)
        out["score"] = score
        return out
    return score


def pareto_dominates(a, b):
    keys = ["coverage", "relevance", "rare_coverage", "edge_density"]
    penalty_keys = ["overflow_ratio", "connector_ratio", "component_penalty", "useless_penalty", "domain_penalty"]
    better_or_equal = all(a[k] >= b[k] for k in keys) and all(a[k] <= b[k] for k in penalty_keys)
    strictly_better = any(a[k] > b[k] for k in keys) or any(a[k] < b[k] for k in penalty_keys)
    return better_or_equal and strictly_better


def pareto_frontier(scored_items):
    frontier = []
    for item in scored_items:
        dominated = False
        survivors = []
        for existing in frontier:
            if pareto_dominates(existing["objectives"], item["objectives"]):
                dominated = True
                break
            if not pareto_dominates(item["objectives"], existing["objectives"]):
                survivors.append(existing)
        if not dominated:
            survivors.append(item)
            frontier = survivors
    return frontier


def crowding_distance(front):
    if not front:
        return
    for item in front:
        item["crowding_distance"] = 0.0
    maximize_keys = ["coverage", "relevance", "rare_coverage", "edge_density", "dominant_ratio"]
    minimize_keys = ["overflow_ratio", "connector_ratio", "component_penalty", "useless_penalty", "domain_penalty", "weighted_domain_drift"]
    for key in maximize_keys:
        ordered = sorted(front, key=lambda it: it["objectives"].get(key, 0.0))
        if len(ordered) == 1:
            ordered[0]["crowding_distance"] = float("inf")
            continue
        lo = ordered[0]["objectives"].get(key, 0.0)
        hi = ordered[-1]["objectives"].get(key, 0.0)
        ordered[0]["crowding_distance"] = float("inf")
        ordered[-1]["crowding_distance"] = float("inf")
        denom = hi - lo
        if denom <= 0:
            continue
        for i in range(1, len(ordered) - 1):
            prev_v = ordered[i - 1]["objectives"].get(key, 0.0)
            next_v = ordered[i + 1]["objectives"].get(key, 0.0)
            if ordered[i]["crowding_distance"] != float("inf"):
                ordered[i]["crowding_distance"] += (next_v - prev_v) / denom
    for key in minimize_keys:
        ordered = sorted(front, key=lambda it: it["objectives"].get(key, 0.0))
        if len(ordered) == 1:
            ordered[0]["crowding_distance"] = float("inf")
            continue
        lo = ordered[0]["objectives"].get(key, 0.0)
        hi = ordered[-1]["objectives"].get(key, 0.0)
        ordered[0]["crowding_distance"] = float("inf")
        ordered[-1]["crowding_distance"] = float("inf")
        denom = hi - lo
        if denom <= 0:
            continue
        for i in range(1, len(ordered) - 1):
            prev_v = ordered[i - 1]["objectives"].get(key, 0.0)
            next_v = ordered[i + 1]["objectives"].get(key, 0.0)
            if ordered[i]["crowding_distance"] != float("inf"):
                ordered[i]["crowding_distance"] += (next_v - prev_v) / denom


def fast_non_dominated_sort(items):
    items = list(items)
    dominates = {}
    dominated_count = {}
    fronts = []
    first_front = []
    for i, p in enumerate(items):
        dominates[i] = []
        dominated_count[i] = 0
        for j, q in enumerate(items):
            if i == j:
                continue
            if pareto_dominates(p["objectives"], q["objectives"]):
                dominates[i].append(j)
            elif pareto_dominates(q["objectives"], p["objectives"]):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            p["pareto_rank"] = 0
            first_front.append(i)
    current = first_front
    rank = 0
    while current:
        front = [items[i] for i in current]
        crowding_distance(front)
        fronts.append(front)
        next_front = []
        for i in current:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    items[j]["pareto_rank"] = rank + 1
                    next_front.append(j)
        rank += 1
        current = next_front
    return fronts


def nsga2_select(scored_population, target_size):
    items = []
    for ind, objectives in scored_population:
        items.append({
            "seed": set(ind["seed"]),
            "connected": set(ind["connected"]),
            "component_id": ind["component_id"],
            "objectives": objectives,
        })
    fronts = fast_non_dominated_sort(items)
    selected = []
    for front in fronts:
        crowding_distance(front)
        front.sort(key=lambda it: (it.get("pareto_rank", 10**9), -it.get("crowding_distance", 0.0), -it["objectives"].get("score", float("-inf"))))
        if len(selected) + len(front) <= target_size:
            selected.extend(front)
        else:
            needed = target_size - len(selected)
            selected.extend(front[:needed])
            break
    return selected, fronts


def load_workload_auto(
    path,
    graph=None,
    include_variables=False,
    min_nodes=2,
    require_graph_overlap=False,
    drop_generic_queries=False,
    graph_overlap_weight=2.0,
):
    lower = path.lower()
    if lower.endswith('.json'):
        return load_sparql_json_workload(
            path,
            graph=graph,
            include_variables=include_variables,
            min_nodes=min_nodes,
            require_graph_overlap=require_graph_overlap,
            drop_generic_queries=drop_generic_queries,
            graph_overlap_weight=graph_overlap_weight,
        )
    return load_workload(path)

def load_workload(path):
    workload = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            nodes = s.split()
            pattern = "SELECT * WHERE { " + " . ".join(f"?x relatedTo {c}" for c in nodes) + " }"
            workload.append(query_from_nodes(nodes, "loaded", pattern))
    return workload

# ============================================================
# Metrics
# ============================================================
def evaluate_coverage(summary_nodes, workload, structured_bonus=0.1):
    if not workload:
        return 0.0
    summary_nodes = set(summary_nodes)
    total = 0.0
    counted_weight = 0.0
    for q in workload:
        qnodes = q["nodes"] if isinstance(q, dict) else set(q)
        if not qnodes:
            continue
        weight = q.get("weight", 1.0) if isinstance(q, dict) else 1.0
        counted_weight += weight
        score = len(qnodes & summary_nodes) / len(qnodes)
        if isinstance(q, dict) and "structured" in q.get("type", "") and qnodes.issubset(summary_nodes):
            score = min(1.0, score + structured_bonus)
        total += weight * min(1.0, score)
    return total / counted_weight if counted_weight else 0.0

def evaluate_rare_query_coverage(summary_nodes, workload):
    rare_queries = [q for q in workload if isinstance(q, dict) and q.get("contains_rare", False)]
    if not rare_queries:
        return 0.0
    return evaluate_coverage(summary_nodes, rare_queries, structured_bonus=0.0)

def connected_components(graph, nodes):
    nodes = set(nodes)
    comps = []
    seen = set()
    for s in list(nodes):
        if s in seen:
            continue
        comp = set()
        q = deque([s])
        seen.add(s)
        while q:
            u = q.popleft()
            comp.add(u)
            for v in graph.neighbors(u):
                if v in nodes and v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps

def is_connected_subset(graph, nodes):
    nodes = set(nodes)
    if not nodes:
        return True
    return len(connected_components(graph, nodes)) == 1




def filter_ranked_nodes_by_support(ranked_nodes, freq_map, dominant_nodes=None, min_node_workload_weight=0.0):
    dominant_nodes = set(dominant_nodes or [])
    filtered = []
    for n in ranked_nodes:
        if freq_map.get(n, 0.0) < float(min_node_workload_weight):
            continue
        if dominant_nodes and n not in dominant_nodes:
            continue
        filtered.append(n)
    return filtered


DEFAULT_BAD_CONNECTOR_PATTERNS = [
    r":Mountain$",
    r":Island$",
    r":HistoricBuilding$",
    r":TelevisionEpisode$",
    r":Film$",
    r":MusicalArtist$",
    r":Actor$",
    r":Writer$",
    r":Director$",
    r":starring$",
    r":writer$",
    r":director$",
]


def connector_is_banned(node, banned_patterns=None):
    patterns = list(banned_patterns or DEFAULT_BAD_CONNECTOR_PATTERNS)
    for pat in patterns:
        if re.search(pat, node):
            return True
    return False


def connector_node_cost(node, dominant_nodes=None, node_weights=None, off_domain_penalty=3.0, support_reward=0.35, banned_patterns=None):
    dominant_nodes = set(dominant_nodes or [])
    node_weights = dict(node_weights or {})
    base = 1.0
    if connector_is_banned(node, banned_patterns=banned_patterns):
        base += float(off_domain_penalty) * 0.75
    if dominant_nodes and node not in dominant_nodes:
        base += float(off_domain_penalty)
    if node_weights:
        max_w = max(node_weights.values()) if node_weights else 0.0
        if max_w > 0:
            base -= float(support_reward) * (node_weights.get(node, 0.0) / max_w)
    return max(0.05, base)


def weighted_bridge_path(graph, source_nodes, target_nodes, allowed_nodes=None, dominant_nodes=None, node_weights=None, off_domain_penalty=3.0, support_reward=0.35, banned_patterns=None):
    source_nodes = set(source_nodes)
    target_nodes = set(target_nodes)
    if not source_nodes or not target_nodes:
        return []
    allowed_nodes = set(allowed_nodes) if allowed_nodes is not None else None

    dist = {}
    parent = {}
    heap = []

    for s in source_nodes:
        if allowed_nodes is not None and s not in allowed_nodes:
            continue
        dist[s] = 0.0
        parent[s] = None
        heapq.heappush(heap, (0.0, s))

    seen = set()
    hit = None
    while heap:
        cost, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)

        if u in target_nodes:
            hit = u
            break

        for v in graph.neighbors(u):
            if allowed_nodes is not None and v not in allowed_nodes:
                continue
            step = connector_node_cost(
                v,
                dominant_nodes=dominant_nodes,
                node_weights=node_weights,
                off_domain_penalty=off_domain_penalty,
                support_reward=support_reward,
                banned_patterns=banned_patterns,
            )
            if step >= 1_000_000.0:
                continue
            new_cost = cost + step
            if new_cost < dist.get(v, float("inf")):
                dist[v] = new_cost
                parent[v] = u
                heapq.heappush(heap, (new_cost, v))

    if hit is None:
        return []

    path = [hit]
    cur = hit
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def path_bridge_cost(path, selected, dominant_nodes=None, node_weights=None, off_domain_penalty=3.0, support_reward=0.35, banned_patterns=None):
    if not path:
        return float("inf")
    selected = set(selected)
    cost = 0.0
    for node in path:
        if node in selected:
            continue
        cost += connector_node_cost(
            node,
            dominant_nodes=dominant_nodes,
            node_weights=node_weights,
            off_domain_penalty=off_domain_penalty,
            support_reward=support_reward,
            banned_patterns=banned_patterns,
        )
    connector_count = sum(1 for node in path if node not in selected)
    return cost + 0.05 * connector_count


def connector_acceptance_gain(workload, current_nodes, path_nodes):
    current_nodes = set(current_nodes)
    path_nodes = set(path_nodes)
    if not workload or not path_nodes:
        return 0.0
    before = 0.0
    after = 0.0
    current_plus = current_nodes | path_nodes
    for q in workload:
        qnodes = set(q.get("nodes", set()))
        w = float(q.get("weight", 1.0))
        if qnodes and qnodes.issubset(current_nodes):
            before += w
        if qnodes and qnodes.issubset(current_plus):
            after += w
    return after - before


def choose_best_bridge_path(
    graph,
    current,
    workload=None,
    allowed_nodes=None,
    fallback_allowed_nodes=None,
    dominant_nodes=None,
    node_weights=None,
    off_domain_penalty=3.0,
    support_reward=0.35,
    banned_patterns=None,
    min_gain_ratio=0.0,
):
    current = set(current)
    dominant_nodes = set(dominant_nodes or [])
    node_weights = dict(node_weights or {})
    comps = connected_components(graph, current)
    if len(comps) <= 1:
        return []

    candidate_sets = []
    if allowed_nodes is not None:
        candidate_sets.append(set(allowed_nodes))
    else:
        candidate_sets.append(None)
    if fallback_allowed_nodes is not None:
        fb = set(fallback_allowed_nodes)
        if not candidate_sets or candidate_sets[-1] != fb:
            candidate_sets.append(fb)
    if None not in candidate_sets:
        candidate_sets.append(None)

    best_path = []
    best_value = None
    for candidate_nodes in candidate_sets:
        local_best = None
        local_value = None
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                path = weighted_bridge_path(
                    graph,
                    comps[i],
                    comps[j],
                    allowed_nodes=candidate_nodes,
                    dominant_nodes=dominant_nodes,
                    node_weights=node_weights,
                    off_domain_penalty=off_domain_penalty,
                    support_reward=support_reward,
                    banned_patterns=banned_patterns,
                )
                if not path:
                    continue
                path_nodes = [n for n in path if n not in current]
                cost = path_bridge_cost(
                    path,
                    current,
                    dominant_nodes=dominant_nodes,
                    node_weights=node_weights,
                    off_domain_penalty=off_domain_penalty,
                    support_reward=support_reward,
                    banned_patterns=banned_patterns,
                )
                gain = connector_acceptance_gain(workload, current, path_nodes)
                ratio = (gain / max(cost, 1e-9)) if path_nodes else float('inf')
                value = (gain, -cost, -len(path))
                if gain > 0 and ratio >= min_gain_ratio:
                    if local_value is None or value > local_value:
                        local_value = value
                        local_best = path
                elif local_best is None and (local_value is None or value > local_value):
                    local_value = value
                    local_best = path
        if local_best:
            return local_best
        if local_value is not None and (best_value is None or local_value > best_value):
            best_value = local_value
            best_path = local_best
    return best_path or []


def connect_nodes_force(
    graph,
    selected,
    allowed_nodes=None,
    fallback_allowed_nodes=None,
    dominant_nodes=None,
    node_weights=None,
    off_domain_penalty=3.0,
    support_reward=0.35,
    banned_patterns=None,
    workload=None,
    min_gain_ratio=0.0,
):
    selected = set(selected)
    allowed_nodes = set(allowed_nodes) if allowed_nodes is not None else None
    fallback_allowed_nodes = set(fallback_allowed_nodes) if fallback_allowed_nodes is not None else None
    dominant_nodes = set(dominant_nodes or [])
    node_weights = dict(node_weights or {})
    if not selected:
        return selected

    def _connect_with(candidate_nodes, current, fallback_nodes=None):
        current = set(current)
        while True:
            comps = connected_components(graph, current)
            if len(comps) <= 1:
                break

            best_path = choose_best_bridge_path(
                graph,
                current,
                workload=workload,
                allowed_nodes=candidate_nodes,
                fallback_allowed_nodes=fallback_nodes,
                dominant_nodes=dominant_nodes,
                node_weights=node_weights,
                off_domain_penalty=off_domain_penalty,
                support_reward=support_reward,
                banned_patterns=banned_patterns,
                min_gain_ratio=min_gain_ratio,
            )

            if not best_path:
                break
            current.update(best_path)
        return current

    connected = _connect_with(allowed_nodes, selected, fallback_allowed_nodes)

    if len(connected_components(graph, connected)) > 1 and fallback_allowed_nodes != allowed_nodes:
        connected = _connect_with(fallback_allowed_nodes, connected)

    return connected


# ============================================================
# Export
# ============================================================
def write_node_list(path, nodes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("node\n")
        for n in sorted(nodes):
            f.write(f"{n}\n")

def write_edge_list(path, edges):
    with open(path, "w", encoding="utf-8") as f:
        f.write("source\ttarget\n")
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\n")

def write_workload_text(path, workload):
    with open(path, "w", encoding="utf-8") as f:
        f.write("id\ttype\tweight\tcontains_rare\tnodes\tpattern\n")
        for i, q in enumerate(workload, start=1):
            qnodes = sorted(q["nodes"])
            f.write(f"Q{i:03d}\t{q['type']}\t{q.get('weight',1.0)}\t{q.get('contains_rare',False)}\t{', '.join(qnodes)}\t{q['pattern']}\n")



def write_summary_text(path, name, seed_nodes, connected_nodes, edges, fitness_parts, requested_budget, connected, pareto_rank=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Algorithm: {name}\\n")
        f.write(f"Score: {fitness_parts.get('score', 0.0):.6f}\\n")
        f.write(f"Coverage: {fitness_parts.get('coverage', 0.0):.6f}\\n")
        f.write(f"Relevance: {fitness_parts.get('relevance', 0.0):.6f}\\n")
        f.write(f"Rare-query coverage: {fitness_parts.get('rare_coverage', 0.0):.6f}\\n")
        f.write(f"Edge density: {fitness_parts.get('edge_density', 0.0):.6f}\\n")
        f.write(f"Connector ratio: {fitness_parts.get('connector_ratio', 0.0):.6f}\\n")
        f.write(f"Overflow ratio: {fitness_parts.get('overflow_ratio', 0.0):.6f}\\n")
        f.write(f"Component penalty: {fitness_parts.get('component_penalty', 0.0)}\\n")
        f.write(f"Useless-node penalty: {fitness_parts.get('useless_penalty', 0.0)}\\n")
        f.write(f"Domain penalty: {fitness_parts.get('domain_penalty', 0.0)}\\n")
        f.write(f"Requested budget (seed/top-k size): {requested_budget}\\n")
        f.write(f"Seed nodes: {len(seed_nodes)}\\n")
        f.write(f"Final connected nodes: {len(connected_nodes)}\\n")
        f.write(f"Connector nodes added: {max(0, len(connected_nodes)-len(seed_nodes))}\\n")
        f.write(f"Overflow beyond budget: {max(0, len(connected_nodes)-requested_budget)}\\n")
        f.write(f"Edges: {len(edges)}\\n")
        f.write(f"Connected: {str(connected).lower()}\\n")
        if pareto_rank is not None:
            f.write(f"Pareto-rank: {pareto_rank}\\n")
        f.write("\\nSeed nodes:\\n")
        for n in sorted(seed_nodes):
            f.write(f"- {n}\\n")
        f.write("\\nFinal connected nodes:\\n")
        for n in sorted(connected_nodes):
            f.write(f"- {n}\\n")
        f.write("\\nSelected edges:\\n")
        for u, v in sorted(edges):
            f.write(f"- {u}\\t{v}\\n")

def save_algorithm_outputs(out_dir, prefix, graph, seed_nodes, connected_nodes, fitness_parts, requested_budget, pareto_front=None):
    os.makedirs(out_dir, exist_ok=True)
    edges = graph.induced_edges(connected_nodes)
    connected = is_connected_subset(graph, connected_nodes)
    write_summary_text(
        os.path.join(out_dir, f"{prefix}_summary.txt"),
        prefix,
        seed_nodes,
        connected_nodes,
        edges,
        fitness_parts,
        requested_budget,
        connected,
        pareto_rank=0 if pareto_front else None,
    )
    write_node_list(os.path.join(out_dir, f"{prefix}_seed_nodes.tsv"), seed_nodes)
    write_node_list(os.path.join(out_dir, f"{prefix}_nodes.tsv"), connected_nodes)
    write_edge_list(os.path.join(out_dir, f"{prefix}_edges.tsv"), edges)

    if pareto_front is not None:
        frontier_path = os.path.join(out_dir, f"{prefix}_pareto_front.json")
        serializable = []
        for item in pareto_front:
            serializable.append({
                "seed_nodes": sorted(item["seed"]),
                "connected_nodes": sorted(item["connected"]),
                "objectives": item["objectives"],
            })
        with open(frontier_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)






def write_ilp_formulation(path, graph, workload, budget):
    workload_nodes = workload_node_set(workload)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Research-grade ILP sketch for exact optimization\\n")
        f.write("# Variables:\\n")
        f.write("#   x_n in {0,1} for each workload-supported node n\\n")
        f.write("#   y_q in {0,1} for each query q, active when all q-nodes are selected\\n")
        f.write("# Objective (schematic): maximize coverage + relevance - penalties\\n\\n")
        f.write("max  sum_q weight_q * y_q\\n")
        f.write("   + lambda_rel * sum_q weight_q * y_q\\n")
        f.write("   + lambda_edge * sum_(u,v in E_W) z_uv\\n")
        f.write("   - lambda_dom * sum_(n notin W) x_n\\n")
        f.write("   - lambda_over * overflow\\n\\n")
        f.write("s.t.\\n")
        f.write(f"  sum_n x_n <= {budget} + overflow\\n")
        for idx, q in enumerate(workload, start=1):
            qnodes = sorted(set(q.get('nodes', set())))
            if not qnodes:
                continue
            for n in qnodes:
                f.write(f"  y_{idx} <= x[{n}]\\n")
            f.write("  ")
            f.write(" + ".join(f"x[{n}]" for n in qnodes))
            f.write(f" - {len(qnodes)} * y_{idx} >= 0\\n")
        f.write("\\n# Connectivity can be added with single-commodity flow or cut constraints.\\n")
        f.write("# This file is a solver-ready template skeleton, not a full MILP export.\\n")

# ============================================================
# WBSumFREQ
# ============================================================

class WBSumFreq:
    def __init__(
        self,
        graph,
        workload,
        budget,
        structured_bonus=0.1,
        rare_bonus=0.15,
        overflow_weight=0.15,
        connector_weight=0.20,
        edge_bonus_weight=0.05,
        relevance_weight=0.75,
        useless_penalty_weight=0.10,
        domain_penalty_weight=0.45,
        dominant_frac=0.15,
        min_node_workload_weight=0.0,
        restrict_connectors_to_workload=True,
        strict_connectivity=False,
        connectivity_hard_penalty=2.0,
        off_domain_connector_penalty=3.0,
        connector_support_reward=0.35,
        connector_banlist_enabled=False,
        min_connector_gain_ratio=0.0,
        coverage_priority=True,
        target_final_size=None,
        target_final_size_penalty=10.0,
        topk_seed_fraction=0.30,
        rare_seed_fraction=0.35,
        mutation_jump_rate=0.35,
        mutation_swap_count=2,
    ):
        self.graph = graph
        self.workload = workload
        self.budget = budget
        self.structured_bonus = structured_bonus
        self.rare_bonus = rare_bonus
        self.overflow_weight = overflow_weight
        self.connector_weight = connector_weight
        self.edge_bonus_weight = edge_bonus_weight
        self.relevance_weight = relevance_weight
        self.useless_penalty_weight = useless_penalty_weight
        self.domain_penalty_weight = domain_penalty_weight
        self.dominant_frac = dominant_frac
        self.min_node_workload_weight = min_node_workload_weight
        self.restrict_connectors_to_workload = restrict_connectors_to_workload
        self.strict_connectivity = strict_connectivity
        self.connectivity_hard_penalty = connectivity_hard_penalty
        self.off_domain_connector_penalty = off_domain_connector_penalty
        self.connector_support_reward = connector_support_reward
        self.connector_banlist_enabled = connector_banlist_enabled
        self.min_connector_gain_ratio = min_connector_gain_ratio
        self.coverage_priority = coverage_priority
        self.target_final_size = int(target_final_size) if target_final_size is not None else None
        self.target_final_size_penalty = float(target_final_size_penalty)
        self.topk_seed_fraction = topk_seed_fraction
        self.rare_seed_fraction = rare_seed_fraction
        self.mutation_jump_rate = mutation_jump_rate
        self.mutation_swap_count = mutation_swap_count
        self.topk_seed_fraction = topk_seed_fraction
        self.rare_seed_fraction = rare_seed_fraction
        self.mutation_jump_rate = mutation_jump_rate
        self.mutation_swap_count = mutation_swap_count
        self.allow_off_workload_connectors = not restrict_connectors_to_workload
        self.node_component, self.components = node_to_component_map(graph)

    def run(self):
        if not self.workload:
            return {"seed": set(), "connected": set(), "objectives": {}}

        global_freq = defaultdict(float)
        for q in self.workload:
            for n in q["nodes"]:
                global_freq[n] += q.get("weight", 1.0)

        best_result = {"seed": set(), "connected": set(), "objectives": {}}
        best_fit = float("-inf")

        for cid, comp_nodes in enumerate(self.components):
            comp_workload = restrict_workload_to_component(self.workload, comp_nodes, self.node_component, cid)
            if not comp_workload:
                continue

            comp_freq = defaultdict(float)
            for q in comp_workload:
                for n in q["nodes"]:
                    comp_freq[n] += q.get("weight", 1.0)

            ranked_pairs = sorted(comp_freq.items(), key=lambda x: (-x[1], -global_freq.get(x[0], 0.0), x[0]))
            if not ranked_pairs:
                continue

            dominant_nodes = dominant_workload_nodes(
                comp_workload,
                frac=self.dominant_frac,
                min_weight=self.min_node_workload_weight,
            )
            ranked = filter_ranked_nodes_by_support(
                [n for n, _ in ranked_pairs],
                comp_freq,
                dominant_nodes=dominant_nodes,
                min_node_workload_weight=self.min_node_workload_weight,
            ) or [n for n, _ in ranked_pairs]
            seeds = ranked[:self.budget]
            allowed_nodes = dominant_nodes if self.restrict_connectors_to_workload else None
            if self.restrict_connectors_to_workload and not allowed_nodes:
                allowed_nodes = workload_node_set(comp_workload)
            fallback_allowed_nodes = None if self.strict_connectivity else allowed_nodes
            connected = connect_nodes_force(
                self.graph,
                seeds,
                allowed_nodes=allowed_nodes,
                fallback_allowed_nodes=fallback_allowed_nodes,
                dominant_nodes=dominant_nodes,
                node_weights=comp_freq,
                off_domain_penalty=self.off_domain_connector_penalty,
                support_reward=self.connector_support_reward,
                banned_patterns=None if self.connector_banlist_enabled is False else DEFAULT_BAD_CONNECTOR_PATTERNS,
            )
            fit_parts = summary_fitness(
                self.graph,
                connected,
                comp_workload,
                budget=self.budget,
                structured_bonus=self.structured_bonus,
                rare_bonus=self.rare_bonus,
                overflow_weight=self.overflow_weight,
                connector_weight=self.connector_weight,
                edge_bonus_weight=self.edge_bonus_weight,
                relevance_weight=self.relevance_weight,
                useless_penalty_weight=self.useless_penalty_weight,
                domain_penalty_weight=self.domain_penalty_weight,
                dominant_frac=self.dominant_frac,
                min_node_workload_weight=self.min_node_workload_weight,
                strict_connectivity=self.strict_connectivity,
                connectivity_hard_penalty=self.connectivity_hard_penalty,
                seed_nodes=seeds,
                return_parts=True,
            )
            fit = fit_parts["score"]
            if fit > best_fit:
                best_fit = fit
                best_result = {"seed": set(seeds), "connected": set(connected), "objectives": fit_parts}

        return best_result


def select_best_ga_candidate(candidates, target_final_size=None, strict_connectivity=False):
    if not candidates:
        return None

    pool = list(candidates)
    if strict_connectivity:
        connected_pool = [item for item in pool if not item.get("objectives", {}).get("disconnected", False)]
        if connected_pool:
            pool = connected_pool

    if target_final_size is not None:
        exact = [item for item in pool if len(item.get("connected", set())) == int(target_final_size)]
        if exact:
            exact.sort(
                key=lambda item: (
                    -item["objectives"].get("coverage", 0.0),
                    -item["objectives"].get("rare_coverage", 0.0),
                    -item["objectives"].get("relevance", 0.0),
                    item["objectives"].get("domain_penalty", 0.0),
                    item["objectives"].get("connector_ratio", 0.0),
                    -item["objectives"].get("score", float("-inf")),
                )
            )
            return exact[0]

    pool.sort(
        key=lambda item: (
            abs(len(item.get("connected", set())) - int(target_final_size)) if target_final_size is not None else 0,
            -item["objectives"].get("coverage", 0.0),
            -item["objectives"].get("rare_coverage", 0.0),
            -item["objectives"].get("relevance", 0.0),
            item["objectives"].get("domain_penalty", 0.0),
            item["objectives"].get("connector_ratio", 0.0),
            -item["objectives"].get("score", float("-inf")),
        )
    )
    return pool[0]


# ============================================================
# GA
# ============================================================

class GASummarizer:
    def __init__(
        self,
        graph,
        workload,
        budget,
        pop=50,
        gen=50,
        seed=42,
        verbose=True,
        overflow_weight=0.15,
        connector_weight=0.20,
        structured_bonus=0.1,
        rare_bonus=0.15,
        edge_bonus_weight=0.05,
        relevance_weight=0.75,
        useless_penalty_weight=0.10,
        domain_penalty_weight=0.45,
        dominant_frac=0.15,
        min_node_workload_weight=0.0,
        restrict_connectors_to_workload=True,
        archive_limit=100,
        strict_connectivity=False,
        connectivity_hard_penalty=2.0,
        selection_mode="nsga2",
        off_domain_connector_penalty=3.0,
        connector_support_reward=0.35,
        connector_banlist_enabled=False,
        min_connector_gain_ratio=0.0,
        coverage_priority=True,
        target_final_size=None,
        target_final_size_penalty=12.0,
        topk_seed_fraction=0.30,
        rare_seed_fraction=0.35,
        mutation_jump_rate=0.35,
        mutation_swap_count=2,
    ):
        self.graph = graph
        self.workload = workload
        self.budget = budget
        self.population_size = pop
        self.generations = gen
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.overflow_weight = overflow_weight
        self.connector_weight = connector_weight
        self.structured_bonus = structured_bonus
        self.rare_bonus = rare_bonus
        self.edge_bonus_weight = edge_bonus_weight
        self.relevance_weight = relevance_weight
        self.useless_penalty_weight = useless_penalty_weight
        self.domain_penalty_weight = domain_penalty_weight
        self.dominant_frac = dominant_frac
        self.min_node_workload_weight = min_node_workload_weight
        self.restrict_connectors_to_workload = restrict_connectors_to_workload
        self.archive_limit = archive_limit
        self.strict_connectivity = strict_connectivity
        self.connectivity_hard_penalty = connectivity_hard_penalty
        self.selection_mode = selection_mode
        self.off_domain_connector_penalty = off_domain_connector_penalty
        self.connector_support_reward = connector_support_reward
        self.connector_banlist_enabled = connector_banlist_enabled
        self.min_connector_gain_ratio = min_connector_gain_ratio
        self.coverage_priority = coverage_priority
        self.target_final_size = int(target_final_size) if target_final_size is not None else None
        self.target_final_size_penalty = float(target_final_size_penalty)
        self.topk_seed_fraction = topk_seed_fraction
        self.rare_seed_fraction = rare_seed_fraction
        self.mutation_jump_rate = mutation_jump_rate
        self.mutation_swap_count = mutation_swap_count

        self.node_component, self.components = node_to_component_map(graph)
        self.component_data = []
        self.freq = defaultdict(float)
        for q in workload:
            for n in q["nodes"]:
                self.freq[n] += q.get("weight", 1.0)

        for cid, comp_nodes in enumerate(self.components):
            comp_workload = restrict_workload_to_component(self.workload, comp_nodes, self.node_component, cid)
            if not comp_workload:
                continue
            comp_freq = defaultdict(float)
            for q in comp_workload:
                for n in q["nodes"]:
                    comp_freq[n] += q.get("weight", 1.0)
            ranked_nodes_all = sorted(comp_freq, key=lambda n: (-comp_freq[n], -self.freq.get(n, 0.0), n))
            dominant_nodes = dominant_workload_nodes(
                comp_workload,
                frac=self.dominant_frac,
                min_weight=self.min_node_workload_weight,
            )
            ranked_nodes = filter_ranked_nodes_by_support(
                ranked_nodes_all,
                comp_freq,
                dominant_nodes=dominant_nodes,
                min_node_workload_weight=self.min_node_workload_weight,
            ) or ranked_nodes_all
            if ranked_nodes:
                self.component_data.append({
                    "id": cid,
                    "nodes": list(comp_nodes),
                    "workload": comp_workload,
                    "freq": comp_freq,
                    "ranked": ranked_nodes,
                    "dominant_nodes": dominant_nodes,
                    "allowed_nodes": workload_node_set(comp_workload) if self.restrict_connectors_to_workload else None,
                    "fallback_allowed_nodes": None if not self.strict_connectivity else None,
                    "node_weights": dict(comp_freq),
                    "rare_nodes": sorted({n for q in comp_workload if q.get("contains_rare", False) for n in q["nodes"]}),
                })

    def _seed_set(self, comp_info, size=None):
        ranked = comp_info["ranked"]
        if not ranked:
            return set()
        size = min(size if size is not None else self.budget, len(ranked))
        pool = ranked[:max(size * 3, size)]
        source = pool if len(pool) >= size else ranked
        return set(self.rng.sample(source, size))

    def _topk_seed(self, comp_info, size=None):
        ranked = comp_info["ranked"]
        if not ranked:
            return set()
        size = min(size if size is not None else self.budget, len(ranked))
        return set(ranked[:size])

    def _rare_injected_seed(self, comp_info, size=None):
        ranked = comp_info["ranked"]
        rare_nodes = [n for n in comp_info.get("rare_nodes", []) if n in ranked]
        if not ranked:
            return set()
        size = min(size if size is not None else self.budget, len(ranked))
        base = list(self._topk_seed(comp_info, size))
        if not base:
            return set()
        if not rare_nodes:
            return set(base)
        replace_count = max(1, min(len(base), int(round(size * self.rare_seed_fraction))))
        rare_pick = self.rng.sample(rare_nodes, min(len(rare_nodes), replace_count))
        for i, rn in enumerate(rare_pick):
            if i < len(base):
                base[-(i + 1)] = rn
        out = []
        seen = set()
        for n in base + ranked:
            if n not in seen:
                out.append(n)
                seen.add(n)
            if len(out) >= size:
                break
        return set(out)

    def make_individual(self, seed_nodes, comp_info):
        seed_nodes = set(seed_nodes)
        ranked = comp_info["ranked"]
        freq = comp_info["freq"]
        if len(seed_nodes) > self.budget:
            ranked_seed = sorted(seed_nodes, key=lambda n: (-freq.get(n, 0.0), n))
            seed_nodes = set(ranked_seed[:self.budget])
        if len(seed_nodes) < min(self.budget, len(ranked)):
            for cand in ranked:
                seed_nodes.add(cand)
                if len(seed_nodes) >= min(self.budget, len(ranked)):
                    break
        connected_nodes = connect_nodes_force(
            self.graph,
            seed_nodes,
            allowed_nodes=comp_info.get("allowed_nodes"),
            fallback_allowed_nodes=comp_info.get("fallback_allowed_nodes"),
            dominant_nodes=comp_info.get("dominant_nodes"),
            node_weights=comp_info.get("node_weights"),
            off_domain_penalty=self.off_domain_connector_penalty,
            support_reward=self.connector_support_reward,
            banned_patterns=None if self.connector_banlist_enabled is False else DEFAULT_BAD_CONNECTOR_PATTERNS,
            workload=comp_info.get("workload"),
            min_gain_ratio=self.min_connector_gain_ratio,
        )
        return {"seed": seed_nodes, "connected": connected_nodes, "component_id": comp_info["id"]}

    def random_individual(self):
        comp_info = self.rng.choice(self.component_data)
        r = self.rng.random()
        if r < self.topk_seed_fraction:
            seed = self._topk_seed(comp_info, self.budget)
        elif r < self.topk_seed_fraction + self.rare_seed_fraction:
            seed = self._rare_injected_seed(comp_info, self.budget)
        else:
            seed = self._seed_set(comp_info, self.budget)
        return self.make_individual(seed, comp_info)

    def objectives(self, individual):
        comp_info = next(c for c in self.component_data if c["id"] == individual["component_id"])
        parts = summary_fitness(
            self.graph,
            individual["connected"],
            comp_info["workload"],
            budget=self.budget,
            structured_bonus=self.structured_bonus,
            rare_bonus=self.rare_bonus,
            overflow_weight=self.overflow_weight,
            connector_weight=self.connector_weight,
            edge_bonus_weight=self.edge_bonus_weight,
            relevance_weight=self.relevance_weight,
            useless_penalty_weight=self.useless_penalty_weight,
            domain_penalty_weight=self.domain_penalty_weight,
            dominant_frac=self.dominant_frac,
            min_node_workload_weight=self.min_node_workload_weight,
            strict_connectivity=self.strict_connectivity,
            connectivity_hard_penalty=self.connectivity_hard_penalty,
            seed_nodes=individual["seed"],
            return_parts=True,
        )
        if self.coverage_priority:
            parts["score"] += 10.0 * parts.get("coverage", 0.0)
            parts["score"] += 3.0 * parts.get("rare_coverage", 0.0)
            parts["score"] += 0.75 * parts.get("relevance", 0.0)
            parts["score"] -= 0.10 * parts.get("connector_ratio", 0.0)
            parts["score"] -= 0.05 * parts.get("domain_penalty", 0.0)
        if self.target_final_size is not None:
            size_gap = abs(len(individual["connected"]) - self.target_final_size)
            parts["target_final_size"] = self.target_final_size
            parts["target_size_gap"] = size_gap
            parts["score"] -= self.target_final_size_penalty * float(size_gap)
        else:
            parts["target_final_size"] = None
            parts["target_size_gap"] = 0
        return parts

    def fitness(self, individual):
        return self.objectives(individual)["score"]

    def mutate(self, individual):
        comp_info = next(c for c in self.component_data if c["id"] == individual["component_id"])
        seeds = set(individual["seed"])
        ranked = comp_info["ranked"]
        freq = comp_info["freq"]
        rare_nodes = [n for n in comp_info.get("rare_nodes", []) if n in ranked]

        swap_count = max(1, int(self.mutation_swap_count))
        for _ in range(swap_count):
            if seeds:
                if self.rng.random() < self.mutation_jump_rate:
                    weakest = min(seeds, key=lambda n: (freq.get(n, 0.0), n))
                    seeds.remove(weakest)
                else:
                    seeds.remove(self.rng.choice(list(seeds)))

            if self.rng.random() < self.mutation_jump_rate:
                jump_pool = [n for n in ranked if n not in seeds]
                if rare_nodes and self.rng.random() < 0.6:
                    jump_pool = [n for n in rare_nodes if n not in seeds] or jump_pool
                if jump_pool:
                    seeds.add(self.rng.choice(jump_pool))
            else:
                candidates = [n for n in ranked if n not in seeds]
                if candidates:
                    seeds.add(self.rng.choice(candidates[:max(10, min(len(candidates), self.budget * 4))]))

        while len(seeds) < min(self.budget, len(ranked)):
            fill_pool = [n for n in rare_nodes if n not in seeds] if rare_nodes and self.rng.random() < 0.4 else [n for n in ranked if n not in seeds]
            if not fill_pool:
                break
            seeds.add(self.rng.choice(fill_pool))

        while len(seeds) > self.budget:
            weakest = min(seeds, key=lambda n: (freq.get(n, 0.0), n))
            seeds.remove(weakest)
        return self.make_individual(seeds, comp_info)

    def crossover(self, ind_a, ind_b):
        comp_info = next(c for c in self.component_data if c["id"] == ind_a["component_id"])
        freq = comp_info["freq"]
        a = sorted(ind_a["seed"], key=lambda n: (-freq.get(n, 0.0), n))
        b = sorted(ind_b["seed"], key=lambda n: (-freq.get(n, 0.0), n))
        half = max(1, self.budget // 2)
        child_seed = set(a[:half] + b[:half])
        for cand in comp_info["ranked"]:
            child_seed.add(cand)
            if len(child_seed) >= min(self.budget, len(comp_info["ranked"])):
                break
        while len(child_seed) > self.budget:
            weakest = min(child_seed, key=lambda n: (freq.get(n, 0.0), n))
            child_seed.remove(weakest)
        return self.make_individual(child_seed, comp_info)

    def run(self):
        if not self.component_data:
            return {"seed": set(), "connected": set(), "objectives": {}, "pareto_front": []}

        population = [self.random_individual() for _ in range(self.population_size)]
        best = None
        best_fit = float("-inf")
        archive = []

        for gen in range(1, self.generations + 1):
            scored = []
            for ind in population:
                objectives = self.objectives(ind)
                scored.append((objectives["score"], ind, objectives))

            scored.sort(key=lambda x: -x[0])

            connected_scored = [item for item in scored if not item[2].get("disconnected", False)]
            scored_for_best = connected_scored if (self.strict_connectivity and connected_scored) else scored
            if self.target_final_size is not None and scored_for_best:
                exact_scored = [item for item in scored_for_best if len(item[1].get("connected", set())) == self.target_final_size]
                if exact_scored:
                    exact_scored.sort(key=lambda item: (
                        -item[2].get("coverage", 0.0),
                        -item[2].get("rare_coverage", 0.0),
                        -item[2].get("relevance", 0.0),
                        item[2].get("domain_penalty", 0.0),
                        item[2].get("connector_ratio", 0.0),
                        -item[2].get("score", float("-inf")),
                    ))
                    scored_for_best = exact_scored

            if scored_for_best and scored_for_best[0][0] > best_fit:
                best_fit = scored_for_best[0][0]
                best = {
                    "seed": set(scored_for_best[0][1]["seed"]),
                    "connected": set(scored_for_best[0][1]["connected"]),
                    "component_id": scored_for_best[0][1]["component_id"],
                    "objectives": scored_for_best[0][2],
                }

            candidates = archive + [
                {
                    "seed": set(ind["seed"]),
                    "connected": set(ind["connected"]),
                    "component_id": ind["component_id"],
                    "objectives": objectives,
                }
                for _, ind, objectives in scored
            ]
            archive = pareto_frontier(candidates)
            if self.strict_connectivity:
                archive = [item for item in archive if not item["objectives"].get("disconnected", False)] or archive
            archive.sort(key=lambda item: (
                item["objectives"].get("target_size_gap", 0),
                -item["objectives"].get("coverage", 0.0),
                -item["objectives"].get("rare_coverage", 0.0),
                -item["objectives"].get("relevance", 0.0),
                item["objectives"].get("domain_penalty", 0.0),
                item["objectives"].get("connector_ratio", 0.0),
                -item["objectives"].get("score", float("-inf")),
            ))
            archive = archive[:self.archive_limit]

            if self.verbose and scored:
                top_fit, top, top_obj = scored[0]
                top_over = max(0, len(top["connected"]) - self.budget)
                top_conn = max(0, len(top["connected"]) - len(top["seed"]))
                avg_fit = sum(s for s, _, _ in scored) / len(scored)
                print(
                    f"[GA] Generation {gen:03d}/{self.generations} | "
                    f"best_fit={top_fit:.6f} | coverage={top_obj['coverage']:.6f} | "
                    f"relevance={top_obj['relevance']:.6f} | rare_cov={top_obj['rare_coverage']:.6f} | "
                    f"domain_pen={top_obj['domain_penalty']} | seed={len(top['seed'])} | final={len(top['connected'])} | "
                    f"connectors={top_conn} | overflow={top_over} | frontier={len(archive)} | avg_fit={avg_fit:.6f}"
                )

            scored_population = [(ind, objectives) for _, ind, objectives in scored]
            if self.strict_connectivity:
                scored_population = [
                    (ind, obj) for ind, obj in scored_population
                    if not obj.get("disconnected", False)
                ] or scored_population

            if self.strict_connectivity:
                connected_pool = [(ind, obj) for ind, obj in scored_population if not obj.get("disconnected", False)]
                if connected_pool:
                    scored_population = connected_pool

            if self.selection_mode == "nsga2":
                selected, fronts = nsga2_select(scored_population, max(2, self.population_size // 2))
                elites = [
                    {"seed": set(item["seed"]), "connected": set(item["connected"]), "component_id": item["component_id"]}
                    for item in selected[:max(2, self.population_size // 10)]
                ]
                parent_pool = [
                    {"seed": set(item["seed"]), "connected": set(item["connected"]), "component_id": item["component_id"]}
                    for item in selected
                ]
            else:
                elite_count = max(2, self.population_size // 10)
                elites = [{"seed": set(ind["seed"]), "connected": set(ind["connected"]), "component_id": ind["component_id"]} for _, ind, _ in scored[:elite_count]]
                parent_pool = [{"seed": set(ind["seed"]), "connected": set(ind["connected"]), "component_id": ind["component_id"]} for _, ind, _ in scored[:max(10, self.population_size // 2)]]

            by_comp = defaultdict(list)
            for ind in parent_pool:
                by_comp[ind["component_id"]].append(ind)
            comp_ids = list(by_comp) or [ind["component_id"] for ind in population]

            new_population = elites[:]
            while len(new_population) < self.population_size:
                cid = self.rng.choice(comp_ids)
                pool = by_comp.get(cid) or [ind for ind in population if ind["component_id"] == cid]
                if len(pool) >= 2:
                    p1, p2 = self.rng.sample(pool, 2)
                else:
                    p1 = p2 = pool[0]
                child = self.crossover(p1, p2)
                if self.rng.random() < 0.9:
                    child = self.mutate(child)
                new_population.append(child)
            population = new_population

        if best is None:
            return {"seed": set(), "connected": set(), "objectives": {}, "pareto_front": archive}

        selected_best = select_best_ga_candidate(
            archive or [best],
            target_final_size=self.target_final_size,
            strict_connectivity=self.strict_connectivity,
        )
        if selected_best is not None:
            best = {
                "seed": set(selected_best["seed"]),
                "connected": set(selected_best["connected"]),
                "component_id": selected_best["component_id"],
                "objectives": selected_best["objectives"],
            }

        if self.strict_connectivity and is_connected_subset(self.graph, best["connected"]) is False:
            connected_archive = [item for item in archive if is_connected_subset(self.graph, item["connected"])]
            if connected_archive:
                selected_best = select_best_ga_candidate(
                    connected_archive,
                    target_final_size=self.target_final_size,
                    strict_connectivity=True,
                )
                best = {
                    "seed": set(selected_best["seed"]),
                    "connected": set(selected_best["connected"]),
                    "component_id": selected_best["component_id"],
                    "objectives": selected_best["objectives"],
                }
            else:
                raise RuntimeError("Strict connectivity requested, but no connected solution was found. Re-run with --allow-off-workload-connectors or lower the dominant-domain thresholds.")

        best["pareto_front"] = archive
        return best


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, help="Path to schema edge list / triple file")
    parser.add_argument("--workload", default=None, help="Optional workload file; supports JSON SPARQL logs or one query per line as whitespace-separated nodes")
    parser.add_argument("--include-variables", action="store_true", help="When loading SPARQL JSON logs, also treat SPARQL variables like ?s as nodes")
    parser.add_argument("--min-query-nodes", type=int, default=2, help="Skip loaded queries with fewer than this many extracted schema nodes")
    parser.add_argument("--require-graph-overlap", action="store_true", help="When loading SPARQL JSON logs, keep only queries that share at least one token with the schema graph")
    parser.add_argument("--drop-generic-log-queries", action="store_true", help="Drop trivial log queries such as SELECT * WHERE { ?s ?p ?o }")
    parser.add_argument("--graph-overlap-weight", type=float, default=2.0, help="Extra weight for loaded SPARQL queries that overlap the schema graph")
    parser.add_argument("--synthetic-workload", action="store_true", help="Use generated workload")
    parser.add_argument("--workload-mode", choices=["simple", "structured", "mixed", "rare"], default="simple")
    parser.add_argument("--structured-ratio", type=float, default=0.5, help="Used in mixed/rare mode")
    parser.add_argument("--synthetic-queries", type=int, default=100)
    parser.add_argument("--query-size", type=int, default=3)
    parser.add_argument("--path-min-len", type=int, default=2)
    parser.add_argument("--path-max-len", type=int, default=4)

    parser.add_argument("--rare-query-ratio", type=float, default=0.25, help="Rare mode: fraction of queries forced to include rare nodes")
    parser.add_argument("--rare-fraction", type=float, default=0.10, help="Rare mode: fraction of lowest-degree nodes considered rare")
    parser.add_argument("--rare-degree-threshold", type=int, default=2, help="Rare mode: low-degree threshold")
    parser.add_argument("--rare-weight", type=float, default=2.0, help="Rare mode: weight assigned to rare-focused queries")
    parser.add_argument("--rare-bonus", type=float, default=0.15, help="GA fitness weight for rare-query coverage")

    parser.add_argument("--budget", type=int, default=20, help="Top-k / initial seed size; final connected summary may exceed this")
    parser.add_argument("--mode", choices=["ga", "wbsum", "both"], default="both")
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-seed-fraction", type=float, default=0.30, help="Fraction of GA initial population seeded directly from Top-K frequency nodes")
    parser.add_argument("--rare-seed-fraction", type=float, default=0.35, help="Fraction of GA initial population seeded with rare-node injection")
    parser.add_argument("--mutation-jump-rate", type=float, default=0.35, help="Probability of long-jump mutation away from the Top-K basin")
    parser.add_argument("--mutation-swap-count", type=int, default=2, help="Number of node replacement attempts per mutation")
    parser.add_argument("--overflow-weight", type=float, default=0.08)
    parser.add_argument("--connector-weight", type=float, default=0.10)
    parser.add_argument("--structured-bonus", type=float, default=0.10)
    parser.add_argument("--edge-bonus-weight", type=float, default=0.08, help="Small structural bonus after coverage")
    parser.add_argument("--relevance-weight", type=float, default=0.40, help="Secondary reward after coverage")
    parser.add_argument("--useless-penalty-weight", type=float, default=0.05, help="Soft penalty for nodes not present in the workload")
    parser.add_argument("--domain-penalty-weight", type=float, default=0.08, help="Soft penalty for domain drift outside the workload")
    parser.add_argument("--dominant-frac", type=float, default=0.15, help="Fraction of max node workload weight used to define dominant-domain nodes")
    parser.add_argument("--min-node-workload-weight", type=float, default=0.0, help="Minimum accumulated workload weight required for a node to count as dominant")
    parser.add_argument("--strict-connectivity", action="store_true", help="Reject disconnected solutions during selection and scoring")
    parser.add_argument("--connectivity-hard-penalty", type=float, default=2.0, help="Extra score penalty applied to disconnected solutions when strict connectivity is enabled")
    parser.add_argument("--selection-mode", choices=["elitist", "nsga2"], default="nsga2", help="GA parent/survivor selection strategy")
    parser.add_argument("--off-domain-connector-penalty", type=float, default=1.5, help="Soft extra cost for connector nodes outside the dominant domain during repair")
    parser.add_argument("--connector-support-reward", type=float, default=0.35, help="Reward for high-workload-support connector nodes during repair")
    parser.add_argument("--min-connector-gain-ratio", type=float, default=0.0, help="Minimum coverage-gain/cost ratio preferred when selecting bridge paths")
    parser.add_argument("--disable-connector-banlist", action="store_true", help="Use soft penalties instead of hard connector bans")
    parser.add_argument("--allow-off-workload-connectors", action="store_true", help="Allow connector nodes outside workload support when connectivity or coverage benefits")
    parser.add_argument("--coverage-priority", dest="coverage_priority", action="store_true", help="Make coverage the dominant GA objective while keeping connectivity hard")
    parser.add_argument("--no-coverage-priority", dest="coverage_priority", action="store_false", help="Disable coverage-first scoring")
    parser.set_defaults(coverage_priority=True)
    parser.add_argument("--archive-limit", type=int, default=100, help="Max Pareto archive size for research-grade GA")
    parser.add_argument("--out", default="results")
    parser.add_argument("--legacy-schema-loader", action="store_true", help="Use the older parser that counts predicates as nodes")
    parser.add_argument("--keep-predicate-nodes", action="store_true", help="In clean schema mode, also keep valid predicates as graph nodes")
    parser.add_argument("--allow-system-schema-nodes", action="store_true", help="Keep rdf/rdfs/owl-style system vocabulary as nodes in clean schema mode")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    graph = load_schema(
        args.schema,
        clean_schema=not args.legacy_schema_loader,
        keep_predicate_nodes=args.keep_predicate_nodes,
        allow_system_nodes=args.allow_system_schema_nodes,
    )
    print(f"Loaded schema: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    if getattr(graph, "schema_stats", None):
        print(
            f"Schema loader mode: {'legacy' if args.legacy_schema_loader else 'clean'} | "
            f"skipped_non_node={graph.schema_stats['skipped_non_node']} | "
            f"skipped_lines={graph.schema_stats['skipped_lines']}"
        )

    rare_nodes = set()

    if args.workload:
        workload = load_workload_auto(
            args.workload,
            graph=graph,
            include_variables=args.include_variables,
            min_nodes=args.min_query_nodes,
            require_graph_overlap=args.require_graph_overlap,
            drop_generic_queries=args.drop_generic_log_queries,
            graph_overlap_weight=args.graph_overlap_weight,
        )
        workload = deduplicate_workload(workload)
        workload_kind = "SPARQL JSON log" if args.workload.lower().endswith(".json") else "workload file"
        print(f"Loaded {workload_kind}: {len(workload)} deduplicated queries")
    elif args.synthetic_workload:
        if args.workload_mode == "simple":
            workload = generate_simple_workload(graph.nodes(), n=args.synthetic_queries, qsize=args.query_size, seed=args.seed)
            print(f"Generated simple synthetic workload: {len(workload)} queries")
        elif args.workload_mode in {"structured", "mixed", "rare"}:
            structured_ratio = 1.0 if args.workload_mode == "structured" else (args.structured_ratio if args.workload_mode == "mixed" else args.structured_ratio)
            workload, rare_nodes = generate_rare_aware_workload(
                graph=graph,
                n=args.synthetic_queries,
                qsize=args.query_size,
                min_len=args.path_min_len,
                max_len=args.path_max_len,
                structured_ratio=structured_ratio,
                rare_query_ratio=args.rare_query_ratio,
                rare_fraction=args.rare_fraction,
                rare_degree_threshold=args.rare_degree_threshold,
                rare_weight=args.rare_weight,
                seed=args.seed
            )
            label = "structured" if args.workload_mode == "structured" else ("mixed" if args.workload_mode == "mixed" else "rare-aware")
            forced_rare = sum(1 for q in workload if isinstance(q, dict) and q.get("contains_rare", False))
            print(f"Generated {label} synthetic workload: {len(workload)} queries")
            print(f"Identified rare schema nodes: {len(rare_nodes)}")
            print(f"Injected rare queries: {forced_rare}")
        else:
            workload, rare_nodes = generate_rare_aware_workload(
                graph=graph,
                n=args.synthetic_queries,
                qsize=args.query_size,
                min_len=args.path_min_len,
                max_len=args.path_max_len,
                structured_ratio=args.structured_ratio,
                rare_query_ratio=args.rare_query_ratio,
                rare_fraction=args.rare_fraction,
                rare_degree_threshold=args.rare_degree_threshold,
                rare_weight=args.rare_weight,
                seed=args.seed
            )
            print(f"Generated rare-aware synthetic workload: {len(workload)} queries")
            print(f"Identified rare schema nodes: {len(rare_nodes)}")
    else:
        workload = generate_simple_workload(graph.nodes(), n=args.synthetic_queries, qsize=args.query_size, seed=args.seed)
        print(f"No workload file given; generated simple synthetic workload: {len(workload)} queries")

    workload_path = os.path.join(args.out, "workload_generated.txt")
    write_workload_text(workload_path, workload)

    if rare_nodes:
        with open(os.path.join(args.out, "rare_nodes.txt"), "w", encoding="utf-8") as f:
            for n in sorted(rare_nodes):
                f.write(f"{n}\n")

    rare_query_count = sum(1 for q in workload if isinstance(q, dict) and q.get("contains_rare", False))

    results = {
        "workload": {
            "count": len(workload),
            "mode": args.workload_mode if not args.workload else "loaded",
            "file": workload_path,
            "rare_query_count": rare_query_count,
            "rare_nodes_count": len(rare_nodes),
            "rare_nodes_file": os.path.join(args.out, "rare_nodes.txt") if rare_nodes else None,
        },
        "settings": {
            "budget": args.budget,
            "budget_meaning": "top-k / initial seed size; connectivity is forced even if final summary grows larger",
            "overflow_weight": args.overflow_weight,
            "connector_weight": args.connector_weight,
            "structured_bonus": args.structured_bonus,
            "rare_bonus": args.rare_bonus,
            "edge_bonus_weight": args.edge_bonus_weight,
            "relevance_weight": args.relevance_weight,
            "useless_penalty_weight": args.useless_penalty_weight,
            "domain_penalty_weight": args.domain_penalty_weight,
            "dominant_frac": args.dominant_frac,
            "min_node_workload_weight": args.min_node_workload_weight,
            "strict_connectivity": args.strict_connectivity,
            "connectivity_hard_penalty": args.connectivity_hard_penalty,
            "selection_mode": args.selection_mode,
            "min_connector_gain_ratio": args.min_connector_gain_ratio,
            "coverage_priority": args.coverage_priority,
            "off_domain_connector_penalty": args.off_domain_connector_penalty,
            "connector_support_reward": args.connector_support_reward,
            "connector_banlist_enabled": not args.disable_connector_banlist,
            "restrict_connectors_to_workload": not args.allow_off_workload_connectors,
            "rare_query_ratio": args.rare_query_ratio,
            "rare_fraction": args.rare_fraction,
            "rare_degree_threshold": args.rare_degree_threshold,
            "rare_weight": args.rare_weight,
            "include_variables": args.include_variables,
            "min_query_nodes": args.min_query_nodes,
            "require_graph_overlap": args.require_graph_overlap,
            "drop_generic_log_queries": args.drop_generic_log_queries,
            "graph_overlap_weight": args.graph_overlap_weight,
            "topk_seed_fraction": args.topk_seed_fraction,
            "rare_seed_fraction": args.rare_seed_fraction,
            "mutation_jump_rate": args.mutation_jump_rate,
            "mutation_swap_count": args.mutation_swap_count,
        }
    }

    if args.mode in ["wbsum", "both"]:
        print("Running WBSumFREQ with forced connectivity...")
        wbsum = WBSumFreq(
            graph,
            workload,
            args.budget,
            structured_bonus=args.structured_bonus,
            rare_bonus=args.rare_bonus,
            overflow_weight=args.overflow_weight,
            connector_weight=args.connector_weight,
            edge_bonus_weight=args.edge_bonus_weight,
            relevance_weight=args.relevance_weight,
            useless_penalty_weight=args.useless_penalty_weight,
            domain_penalty_weight=args.domain_penalty_weight,
            dominant_frac=args.dominant_frac,
            min_node_workload_weight=args.min_node_workload_weight,
            restrict_connectors_to_workload=not args.allow_off_workload_connectors,
            strict_connectivity=args.strict_connectivity,
            connectivity_hard_penalty=args.connectivity_hard_penalty,
            off_domain_connector_penalty=args.off_domain_connector_penalty,
            connector_support_reward=args.connector_support_reward,
            connector_banlist_enabled=not args.disable_connector_banlist,
            min_connector_gain_ratio=args.min_connector_gain_ratio,
            coverage_priority=args.coverage_priority,
            topk_seed_fraction=args.topk_seed_fraction,
            rare_seed_fraction=args.rare_seed_fraction,
            mutation_jump_rate=args.mutation_jump_rate,
            mutation_swap_count=args.mutation_swap_count,
        )
        w = wbsum.run()
        w_parts = w.get("objectives") or summary_fitness(
            graph,
            w["connected"],
            workload,
            budget=args.budget,
            structured_bonus=args.structured_bonus,
            rare_bonus=args.rare_bonus,
            overflow_weight=args.overflow_weight,
            connector_weight=args.connector_weight,
            edge_bonus_weight=args.edge_bonus_weight,
            relevance_weight=args.relevance_weight,
            useless_penalty_weight=args.useless_penalty_weight,
            domain_penalty_weight=args.domain_penalty_weight,
            dominant_frac=args.dominant_frac,
            min_node_workload_weight=args.min_node_workload_weight,
            strict_connectivity=args.strict_connectivity,
            connectivity_hard_penalty=args.connectivity_hard_penalty,
            seed_nodes=w["seed"],
            return_parts=True,
        )
        save_algorithm_outputs(args.out, "wbsum", graph, w["seed"], w["connected"], w_parts, args.budget)
        results["wbsum"] = {
            "score": w_parts["score"],
            "coverage": w_parts["coverage"],
            "relevance": w_parts["relevance"],
            "rare_query_coverage": w_parts["rare_coverage"],
            "seed_nodes": len(w["seed"]),
            "nodes": len(w["connected"]),
            "edges": len(graph.induced_edges(w["connected"])),
            "connected": is_connected_subset(graph, w["connected"]),
            "connector_nodes": max(0, len(w["connected"]) - len(w["seed"])),
            "overflow": max(0, len(w["connected"]) - args.budget),
            "summary_txt": os.path.join(args.out, "wbsum_summary.txt"),
            "seed_nodes_tsv": os.path.join(args.out, "wbsum_seed_nodes.tsv"),
            "nodes_tsv": os.path.join(args.out, "wbsum_nodes.tsv"),
            "edges_tsv": os.path.join(args.out, "wbsum_edges.tsv"),
        }

    if args.mode in ["ga", "both"]:
        print(f"Running GA with rare-aware fitness (pop={args.population}, gen={args.generations})...")
        ga_target_final_size = None
        if args.mode == "both" and "wbsum" in results:
            ga_target_final_size = results["wbsum"]["nodes"]
            print(f"Constraining GA to match WBSUM final node count exactly when possible: target_final_size={ga_target_final_size}")
        ga = GASummarizer(
            graph=graph,
            workload=workload,
            budget=args.budget,
            pop=args.population,
            gen=args.generations,
            seed=args.seed,
            verbose=True,
            overflow_weight=args.overflow_weight,
            connector_weight=args.connector_weight,
            structured_bonus=args.structured_bonus,
            rare_bonus=args.rare_bonus,
            edge_bonus_weight=args.edge_bonus_weight,
            relevance_weight=args.relevance_weight,
            useless_penalty_weight=args.useless_penalty_weight,
            domain_penalty_weight=args.domain_penalty_weight,
            dominant_frac=args.dominant_frac,
            min_node_workload_weight=args.min_node_workload_weight,
            restrict_connectors_to_workload=not args.allow_off_workload_connectors,
            archive_limit=args.archive_limit,
            strict_connectivity=args.strict_connectivity,
            connectivity_hard_penalty=args.connectivity_hard_penalty,
            selection_mode=args.selection_mode,
            off_domain_connector_penalty=args.off_domain_connector_penalty,
            connector_support_reward=args.connector_support_reward,
            connector_banlist_enabled=not args.disable_connector_banlist,
            min_connector_gain_ratio=args.min_connector_gain_ratio,
            coverage_priority=args.coverage_priority,
            target_final_size=ga_target_final_size,
        )
        g = ga.run()
        g_parts = g.get("objectives") or summary_fitness(
            graph,
            g["connected"],
            workload,
            budget=args.budget,
            structured_bonus=args.structured_bonus,
            rare_bonus=args.rare_bonus,
            overflow_weight=args.overflow_weight,
            connector_weight=args.connector_weight,
            edge_bonus_weight=args.edge_bonus_weight,
            relevance_weight=args.relevance_weight,
            useless_penalty_weight=args.useless_penalty_weight,
            domain_penalty_weight=args.domain_penalty_weight,
            dominant_frac=args.dominant_frac,
            min_node_workload_weight=args.min_node_workload_weight,
            strict_connectivity=args.strict_connectivity,
            connectivity_hard_penalty=args.connectivity_hard_penalty,
            seed_nodes=g["seed"],
            return_parts=True,
        )
        save_algorithm_outputs(args.out, "ga", graph, g["seed"], g["connected"], g_parts, args.budget, pareto_front=g.get("pareto_front"))
        results["ga"] = {
            "score": g_parts["score"],
            "coverage": g_parts["coverage"],
            "relevance": g_parts["relevance"],
            "rare_query_coverage": g_parts["rare_coverage"],
            "seed_nodes": len(g["seed"]),
            "nodes": len(g["connected"]),
            "edges": len(graph.induced_edges(g["connected"])),
            "connected": is_connected_subset(graph, g["connected"]),
            "connector_nodes": max(0, len(g["connected"]) - len(g["seed"])),
            "overflow": max(0, len(g["connected"]) - args.budget),
            "summary_txt": os.path.join(args.out, "ga_summary.txt"),
            "seed_nodes_tsv": os.path.join(args.out, "ga_seed_nodes.tsv"),
            "nodes_tsv": os.path.join(args.out, "ga_nodes.tsv"),
            "edges_tsv": os.path.join(args.out, "ga_edges.tsv"),
            "pareto_front_json": os.path.join(args.out, "ga_pareto_front.json"),
        }

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done.")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to: {metrics_path}")

if __name__ == "__main__":
    main()