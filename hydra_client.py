"""
Thin client for HydraDB's HTTP JSON query API.

Not using the Python `neo4j` Bolt driver: it hard-rejects any server whose
agent string doesn't start with "Neo4j/" (HydraDB reports "SlateDBGraph/0.1.0"),
with no config flag to bypass it. The HTTP API works fine and is simpler for
our needs anyway.

Empirically confirmed constraints of this HydraDB build that shape every query
written against it (see README for detail):
  - every node's `id` property must be an integer (not a string)
  - every write (CREATE or MERGE) must be exactly one relationship connecting
    exactly two node patterns -- no bare single-node writes, no multi-hop
    writes in one clause
  - MATCH followed by CREATE in the same query is rejected; MATCH + SET works
  - MERGE correctly reuses an existing node by id on either side of the
    pattern instead of duplicating it -- this is how we "attach a new node to
    an existing one"
  - relationship patterns require exactly one type, always directed
  - request body param is `parameters`, not `params`
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HTTP_URL = os.environ["HYDRA_HTTP_URL"]
AUTH_TOKEN = os.environ["HYDRA_AUTH_TOKEN"]
NAMESPACE = os.environ["HYDRA_NAMESPACE"]
GRAPH = os.environ["HYDRA_GRAPH"]
CELL_ID = os.environ["HYDRA_CELL_ID"]

_QUERY_URL = f"{HTTP_URL}/v1/graphs/{GRAPH}/query"


def run(query: str, **parameters) -> list[dict]:
    """Run a Cypher query against HydraDB, return rows as list of {column: value} dicts."""
    # Introduce a small pacing delay before mutation queries to prevent write locks/concurrency issues
    if any(keyword in query for keyword in ("MERGE", "SET", "CREATE")):
        time.sleep(0.35)

    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(
                _QUERY_URL,
                headers={
                    "Authorization": f"Bearer {AUTH_TOKEN}",
                    "X-Graph-Namespace": NAMESPACE,
                    "Content-Type": "application/json",
                },
                json={"cell_id": CELL_ID, "query": query, "parameters": parameters},
                timeout=30,
            )
            body = resp.json()
        except requests.exceptions.RequestException as e:
            # Transient network/timeout error -- worth retrying.
            last_err = e
            time.sleep(1.0 + attempt * 0.5)
            continue

        if "error" in body:
            # The server understood the request and rejected the query itself (bad
            # Cypher, unsupported pattern, etc.) -- retrying would just fail identically.
            raise RuntimeError(f"HydraDB query failed: {body['error']['message']}\nquery: {query}\nparameters: {parameters}")

        columns = body["columns"]
        rows = []
        for raw_row in body["rows"]:
            rows.append({col: cell.get("value") for col, cell in zip(columns, raw_row)})
        return rows

    raise last_err
