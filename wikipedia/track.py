import csv
import json
import math
import random
import re
from os import getcwd
from os.path import dirname
from typing import Iterator, List

from esrally.driver import runner
from esrally.track.params import ParamSource

QUERIES_DIRNAME: str = dirname(__file__)
QUERIES_FILENAME: str = f"{QUERIES_DIRNAME}/queries.csv"
SAMPLE_IDS_FILENAME: str = f"{QUERIES_DIRNAME}/ids.txt"

SEARCH_APPLICATION_ROOT_ENDPOINT: str = "/_application/search_application"
QUERY_RULES_ENDPOINT: str = "/_query_rules"

QUERY_CLEAN_REXEXP = regexp = re.compile("[^0-9a-zA-Z]+")

# Constants for the highlighting challenge. The absent term is alphanumeric like the
# pre-cleaned clickstream queries and matches no document, forcing a highlight miss
# on every row that already matched the WHERE/query clause.
ABSENT_QUERY_TERM = "qzxvwmbtr"
# Wikipedia has articles longer than index.highlight.max_analyzed_offset (1M chars by
# default). Without a request-level max_analyzed_offset the _search request fails on
# them, whereas ES|QL HIGHLIGHT silently truncates; setting it keeps behavior comparable.
DSL_MAX_ANALYZED_OFFSET = 1000000
HIGHLIGHT_COLUMN_PREFIX = "highlight_"


def query_samples(k: int, random_seed: int = None) -> List[str]:
    with open(QUERIES_FILENAME) as queries_file:
        csv_reader = csv.reader(queries_file)
        next(csv_reader)
        queries_with_probabilities = list(tuple(line) for line in csv_reader)

        queries = [QUERY_CLEAN_REXEXP.sub(" ", query).lower() for query, _ in queries_with_probabilities]
        probabilities = [float(probability) for _, probability in queries_with_probabilities]
        random.seed(random_seed)

        return random.choices(queries, weights=probabilities, k=k)


# ids file was created with the following command: grep _index pages-1k.json | jq .index._id | tr -d '"' | grep -v null > ids.txt
def ids_samples() -> List[str]:
    with open(SAMPLE_IDS_FILENAME, "r") as file:
        ids = {line.strip() for line in file}
    for i in range(100):
        ids.add(f"missing-id-{i}")
    return list(ids)


def parse_fields(params) -> List[str]:
    fields = params.get("fields", "title,content")
    if isinstance(fields, str):
        fields = fields.split(",")
    return [field.strip() for field in fields if field.strip()]


def significant_word(query: str, min_length: int = 5) -> str:
    """Picks the word used to synthesize wildcard/fuzzy queries: the first word with
    at least ``min_length`` characters, falling back to the longest word."""
    words = query.split()
    if not words:
        return "wikipedia"
    return next((word for word in words if len(word) >= min_length), max(words, key=len))


def qstr_query_text(query_type: str, fields: List[str], query: str) -> str:
    """Renders the Lucene query text shared by the ES|QL QSTR and Query DSL query_string
    operations of the highlighting challenge, so both engines run the exact same query.

    The clickstream queries contain no query_string syntax characters, so boolean,
    wildcard and fuzzy queries are synthesized from the sampled query text.
    """
    if query_type == "qstr-bool":
        words = query.split() or ["wikipedia"]
        pattern = f"({words[0]} AND {words[1]})" if len(words) > 1 else words[0]
    elif query_type == "wildcard":
        pattern = f"{significant_word(query)[:4]}*"
    elif query_type == "fuzzy":
        pattern = f"{significant_word(query)}~"
    else:
        raise ValueError("Unknown query_string query type: " + query_type)
    return " OR ".join(f"{field}:{pattern}" for field in fields)


class SearchApplicationParams:
    def __init__(self, track, params):
        self.indices = params.get("indices", track.index_names())
        self.name = params.get("search-application", f"{self.indices[0]}-search-application")


class CreateSearchApplicationParamSource(ParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self.search_application_params = SearchApplicationParams(track, params)

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        return {
            "method": "PUT",
            "path": f"{SEARCH_APPLICATION_ROOT_ENDPOINT}/{self.search_application_params.name}",
            "body": {"indices": self.search_application_params.indices},
        }


class QueryRulesetParams:
    def __init__(self, track, params):
        self.indices = params.get("indices", track.index_names())
        self.ruleset_id = params.get("ruleset_id")
        self.ruleset_size = params.get("ruleset_size")


class QueryIteratorParamSource(ParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self._batch_size = self._params.get("batch_size", 100000)
        self._random_seed = self._params.get("seed", None)
        self._sample_queries = query_samples(self._batch_size, self._random_seed)
        self._queries_iterator = None

    def size(self):
        return None

    def partition(self, partition_index, total_partitions):
        if self._queries_iterator is None:
            # Rotate each client's starting position: with a fixed seed all clients
            # sample the same sequence and would otherwise issue identical queries
            # in lockstep.
            offset = partition_index * len(self._sample_queries) // total_partitions
            self._sample_queries = self._sample_queries[offset:] + self._sample_queries[:offset]
            self._queries_iterator = iter(self._sample_queries)
        return self


class SearchApplicationSearchParamSource(QueryIteratorParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self.search_application_params = SearchApplicationParams(track, params)

    def params(self):
        try:
            query = next(self._queries_iterator)
            return {
                "method": "POST",
                "path": f"{SEARCH_APPLICATION_ROOT_ENDPOINT}/{self.search_application_params.name}/_search",
                "body": {
                    "params": {
                        "query_string": query,
                    },
                },
            }
        except StopIteration:
            self._queries_iterator = iter(self._sample_queries)
            return self.params()


class CreateQueryRulesetParamSource(ParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self.query_ruleset_params = QueryRulesetParams(track, params)

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        ids = ids_samples()
        rules = []
        for i in range(self.query_ruleset_params.ruleset_size):
            rule = {
                "rule_id": "rule_{{i}}",
                "type": random.choice(["pinned", "exclude"]),
                "criteria": [{"type": "exact", "metadata": "rule_key", "values": [random.choice(["match", "no-match"])]}],
                "actions": {"ids": [random.choice(ids)]},
            }
            rules.append(rule)

        return {"method": "PUT", "path": f"{QUERY_RULES_ENDPOINT}/{self.query_ruleset_params.ruleset_id}", "body": {"rules": rules}}


class QueryRulesSearchParamSource(QueryIteratorParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self.query_ruleset_params = QueryRulesetParams(track, params)

    def params(self):
        try:
            query = next(self._queries_iterator)
            return {
                "method": "POST",
                "path": "/_search",
                "body": {
                    "query": {
                        "rule": {
                            "match_criteria": {"rule_key": random.choice(["match", "no-match"])},
                            "ruleset_ids": [self.query_ruleset_params.ruleset_id],
                            "organic": {"query_string": {"query": query, "default_field": self._params["search-fields"]}},
                        }
                    },
                    "size": self._params["size"],
                },
            }
        except StopIteration:
            self._queries_iterator = iter(self._sample_queries)
            return self.params()


class PinnedSearchParamSource(QueryIteratorParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self.query_ruleset_params = QueryRulesetParams(track, params)
        self.ids = ids_samples()

    def params(self):
        try:
            query = next(self._queries_iterator)
            return {
                "method": "POST",
                "path": "/_search",
                "body": {
                    "query": {
                        "pinned": {
                            "organic": {"query_string": {"query": query, "default_field": self._params["search-fields"]}},
                            "ids": [random.choice(self.ids)],
                        }
                    },
                    "size": self._params["size"],
                },
            }
        except StopIteration:
            self._queries_iterator = iter(self._sample_queries)
            return self.params()


class RetrieverParamSource(QueryIteratorParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self._index_name = params.get("index", track.indices[0].name if len(track.indices) == 1 else "_all")
        self._search_fields = self._params["search-fields"]
        self._rerank = params.get("rerank", False)
        self._reranker = params.get("reranker", "random_reranker")
        self._size = params.get("size", 20)

    def params(self):
        standard_retriever = {
            "standard": {"query": {"query_string": {"query": next(self._queries_iterator), "default_field": self._search_fields}}}
        }

        retriever = standard_retriever
        if self._rerank:
            retriever = {self._reranker: {"retriever": standard_retriever, "field": self._search_fields, "rank_window_size": self._size}}

        try:
            return {
                "method": "POST",
                "path": f"/{self._index_name}/_search",
                "body": {"retriever": retriever, "size": self._size},
            }
        except StopIteration:
            self._queries_iterator = iter(self._sample_queries)
            return self.params()


# TODO Add other queries, check default fields for search. Compare them with other DSL queries
class EsqlSearchParamSource(QueryIteratorParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self._index_name = params.get("index", track.indices[0].name if len(track.indices) == 1 else "_all")
        self._search_fields = self._params.get("search-fields", "*")
        self._size = params.get("size", 20)
        self._query_type = self._params["query-type"]
        # Parameters below are used by the highlighting challenge: `source: false`
        # emits a source-free query (`KEEP _id, _score` only) and `highlight: true`
        # additionally appends a HIGHLIGHT command mirroring the WHERE query.
        self._fields = parse_fields(self._params)
        self._source = params.get("source", True)
        self._highlight = params.get("highlight", False)
        self._highlight_placement = params.get("highlight-placement", "post-limit")
        if self._highlight_placement not in ("post-limit", "pre-sort"):
            raise ValueError("Unknown highlight placement: " + self._highlight_placement)
        self._highlight_options = params.get("highlight-options")
        self._highlight_miss = params.get("highlight-miss", False)
        self._detailed_results = params.get("detailed-results", False)

    def _query_expression(self, query):
        if self._query_type == "query-string":
            return f'QSTR("{ query }", {{"default_field": "{ self._search_fields }" }})'
        elif self._query_type == "match":
            return " OR ".join(f'MATCH({field}, "{query}")' for field in self._fields)
        elif self._query_type == "kql":
            return f'KQL("{ self._search_fields }:{ query }")'
        elif self._query_type == "match_phrase":
            return " OR ".join(f'MATCH_PHRASE({field}, "{query}")' for field in self._fields)
        elif self._query_type in ("qstr-bool", "wildcard", "fuzzy"):
            return f'QSTR("{ qstr_query_text(self._query_type, self._fields, query) }")'
        else:
            raise ValueError("Unknown query type: " + self._query_type)

    def _highlight_command(self, query):
        # HIGHLIGHT has no shortcut form reusing the WHERE query, so the query is
        # spelled out a second time (with an absent term for forced-miss operations).
        highlight_query = self._query_expression(ABSENT_QUERY_TERM if self._highlight_miss else query)
        command = f"HIGHLIGHT {highlight_query} ON {', '.join(self._fields)}"
        if self._highlight_options:
            command += f" WITH {json.dumps(self._highlight_options, sort_keys=True)}"
        return command

    def params(self):
        try:
            query = next(self._queries_iterator)
            query_expression = self._query_expression(query)

            if self._source and not self._highlight:
                return {
                    "query": f"FROM {self._index_name} METADATA _id, _score, _source | WHERE { query_expression } | KEEP _id, _score, _source | SORT _score DESC | LIMIT { self._size }",
                }

            # Source-free shape: _source transport would dominate the highlight cost
            # that op-minus-twin deltas are meant to isolate.
            commands = [f"FROM {self._index_name} METADATA _id, _score", f"WHERE {query_expression}"]
            keep_columns = ["_id", "_score"]
            if self._highlight and self._highlight_placement == "pre-sort":
                # HIGHLIGHT before SORT/LIMIT runs on data nodes over every matching
                # row unless the optimizer pushes it below the limit.
                commands.append(self._highlight_command(query))
            commands.append("SORT _score DESC")
            commands.append(f"LIMIT {self._size}")
            if self._highlight:
                if self._highlight_placement == "post-limit":
                    commands.append(self._highlight_command(query))
                keep_columns.extend(f"{HIGHLIGHT_COLUMN_PREFIX}{field}" for field in self._fields)
            commands.append(f"KEEP {', '.join(keep_columns)}")

            return {
                "query": " | ".join(commands),
                "detailed-results": self._detailed_results,
            }

        except StopIteration:
            self._queries_iterator = iter(self._sample_queries)
            return self.params()


class QueryParamSource(QueryIteratorParamSource):
    def __init__(self, track, params, **kwargs):
        super().__init__(track, params, **kwargs)
        self._index_name = params.get("index", track.indices[0].name if len(track.indices) == 1 else "_all")
        self._cache = params.get("cache", False)
        self._query_type = self._params["query-type"]
        self._detailed_results = params.get("detailed-results", False)
        # Parameters below are used by the highlighting challenge; they mirror the
        # equivalent EsqlSearchParamSource parameters so that a DSL operation and its
        # ES|QL counterpart run the same query over the same fields.
        self._fields = parse_fields(self._params)
        self._source = params.get("source", True)
        self._highlight = params.get("highlight", False)
        self._highlight_options = params.get("highlight-options")
        self._highlight_miss = params.get("highlight-miss", False)

    def _query_body(self, query):
        if self._query_type == "query-string":
            return {"query_string": {"query": query, "default_field": self._params["search-fields"]}}
        elif self._query_type == "kql":
            return {"kql": {"query": query, "default_field": self._params["search-fields"]}}
        elif self._query_type in ("match", "multi_match"):
            return {"bool": {"should": [{"match": {field: query}} for field in self._fields]}}
        elif self._query_type == "match_phrase":
            return {"bool": {"should": [{"match_phrase": {field: query}} for field in self._fields]}}
        elif self._query_type in ("qstr-bool", "wildcard", "fuzzy"):
            return {"query_string": {"query": qstr_query_text(self._query_type, self._fields, query)}}
        else:
            raise ValueError("Unknown query type: " + self._query_type)

    def _highlight_body(self):
        highlight_fields = {}
        for field in self._fields:
            field_options = {}
            if self._highlight_miss:
                field_options["highlight_query"] = {"match": {field: ABSENT_QUERY_TERM}}
            highlight_fields[field] = field_options
        highlight_body = {"fields": highlight_fields, "max_analyzed_offset": DSL_MAX_ANALYZED_OFFSET}
        if self._highlight_options:
            highlight_body.update(self._highlight_options)
        return highlight_body

    def params(self):
        try:
            query = next(self._queries_iterator)
            query_body = self._query_body(query)

        except StopIteration:
            self._queries_iterator = iter(self._sample_queries)
            return self.params()

        body = {
            "query": query_body,
            "size": self._params["size"],
        }
        if not self._source:
            body["_source"] = False
        if self._highlight:
            body["highlight"] = self._highlight_body()

        return {
            "body": body,
            "index": self._index_name,
            "cache": self._cache,
            "detailed-results": self._detailed_results,
        }


def register(registry):
    registry.register_param_source("query-search", QueryParamSource)
    registry.register_param_source("create-search-application-param-source", CreateSearchApplicationParamSource)
    registry.register_param_source("search-application-search-param-source", SearchApplicationSearchParamSource)
    registry.register_param_source("create-query-ruleset-param-source", CreateQueryRulesetParamSource)
    registry.register_param_source("query-rules-search-param-source", QueryRulesSearchParamSource)
    registry.register_param_source("pinned-search-param-source", PinnedSearchParamSource)
    registry.register_param_source("retriever-search", RetrieverParamSource)
    registry.register_param_source("esql-search", EsqlSearchParamSource)
