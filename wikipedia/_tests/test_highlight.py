# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import importlib.util
import json
import pathlib
import types

import jinja2
import pytest

TRACK_DIR = pathlib.Path(__file__).parents[1]

spec = importlib.util.spec_from_file_location("wikipedia_track", TRACK_DIR / "track.py")
wikipedia_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wikipedia_track)

TRACK_STUB = types.SimpleNamespace(indices=[types.SimpleNamespace(name="wikipedia")])
SEED = 99
BATCH_SIZE = 5


def sampled_queries():
    return wikipedia_track.query_samples(BATCH_SIZE, SEED)


def esql_params(**op_params):
    params = {"query-type": "match", "batch_size": BATCH_SIZE, "seed": SEED, **op_params}
    return wikipedia_track.EsqlSearchParamSource(TRACK_STUB, params).partition(0, 1).params()


def dsl_params(**op_params):
    params = {"query-type": "match", "size": 10, "batch_size": BATCH_SIZE, "seed": SEED, **op_params}
    return wikipedia_track.QueryParamSource(TRACK_STUB, params).partition(0, 1).params()


class TestQuerySynthesis:
    def test_significant_word_prefers_first_long_word(self):
        assert wikipedia_track.significant_word("the london underground") == "london"

    def test_significant_word_falls_back_to_longest_word(self):
        assert wikipedia_track.significant_word("the of who") == "the"
        assert wikipedia_track.significant_word("") == "wikipedia"

    def test_qstr_bool(self):
        assert wikipedia_track.qstr_query_text("qstr-bool", ["content"], "quick brown fox") == "content:(quick AND brown)"
        assert wikipedia_track.qstr_query_text("qstr-bool", ["content"], "fox") == "content:fox"

    def test_wildcard(self):
        assert wikipedia_track.qstr_query_text("wildcard", ["content"], "sauron rings") == "content:saur*"

    def test_fuzzy(self):
        assert wikipedia_track.qstr_query_text("fuzzy", ["content"], "sauron rings") == "content:sauron~"

    def test_multiple_fields(self):
        assert wikipedia_track.qstr_query_text("wildcard", ["title", "content"], "sauron") == "title:saur* OR content:saur*"


class TestEsqlHighlightParams:
    def test_legacy_shape_is_unchanged(self):
        query = sampled_queries()[0]
        params = esql_params(**{"search-fields": "*", "size": 20})
        assert params == {
            "query": f'FROM wikipedia METADATA _id, _score, _source | WHERE MATCH(title, "{query}") OR MATCH(content, "{query}")'
            f" | KEEP _id, _score, _source | SORT _score DESC | LIMIT 20",
        }

    def test_source_free_twin(self):
        query = sampled_queries()[0]
        params = esql_params(**{"source": False, "size": 10, "detailed-results": True})
        assert params == {
            "query": f'FROM wikipedia METADATA _id, _score | WHERE MATCH(title, "{query}") OR MATCH(content, "{query}")'
            f" | SORT _score DESC | LIMIT 10 | KEEP _id, _score",
            "detailed-results": True,
        }

    def test_highlight_post_limit(self):
        query = sampled_queries()[0]
        params = esql_params(**{"source": False, "highlight": True, "size": 10})
        match = f'MATCH(title, "{query}") OR MATCH(content, "{query}")'
        assert params["query"] == (
            f"FROM wikipedia METADATA _id, _score | WHERE {match} | SORT _score DESC | LIMIT 10"
            f" | HIGHLIGHT {match} ON title, content | KEEP _id, _score, highlight_title, highlight_content"
        )

    def test_highlight_pre_sort_placement(self):
        query = sampled_queries()[0]
        params = esql_params(**{"source": False, "highlight": True, "highlight-placement": "pre-sort", "size": 10})
        match = f'MATCH(title, "{query}") OR MATCH(content, "{query}")'
        assert params["query"] == (
            f"FROM wikipedia METADATA _id, _score | WHERE {match} | HIGHLIGHT {match} ON title, content"
            f" | SORT _score DESC | LIMIT 10 | KEEP _id, _score, highlight_title, highlight_content"
        )

    def test_unknown_placement_is_rejected(self):
        with pytest.raises(ValueError):
            esql_params(**{"source": False, "highlight": True, "highlight-placement": "sideways"})

    def test_forced_miss_highlights_absent_term_but_keeps_where_query(self):
        query = sampled_queries()[0]
        params = esql_params(**{"source": False, "highlight": True, "highlight-miss": True, "fields": "content", "size": 10})
        assert f'WHERE MATCH(content, "{query}")' in params["query"]
        assert f'HIGHLIGHT MATCH(content, "{wikipedia_track.ABSENT_QUERY_TERM}") ON content' in params["query"]

    def test_highlight_options_render_as_with_map(self):
        params = esql_params(
            **{"source": False, "highlight": True, "fields": "content", "highlight-options": {"no_match_size": 100}, "size": 10}
        )
        assert 'ON content WITH {"no_match_size": 100} | KEEP' in params["query"]


class TestDslHighlightParams:
    def test_highlight_body(self):
        params = dsl_params(**{"source": False, "highlight": True})
        body = params["body"]
        assert body["_source"] is False
        assert body["highlight"] == {
            "fields": {"title": {}, "content": {}},
            "max_analyzed_offset": wikipedia_track.DSL_MAX_ANALYZED_OFFSET,
        }

    def test_twin_has_no_highlight_and_no_source(self):
        params = dsl_params(**{"source": False})
        assert "highlight" not in params["body"]
        assert params["body"]["_source"] is False

    def test_legacy_body_is_unchanged(self):
        query = sampled_queries()[0]
        params = dsl_params()
        assert params["body"] == {
            "query": {"bool": {"should": [{"match": {"title": query}}, {"match": {"content": query}}]}},
            "size": 10,
        }

    def test_forced_miss_adds_per_field_highlight_query(self):
        params = dsl_params(**{"source": False, "highlight": True, "highlight-miss": True, "fields": "content"})
        highlight = params["body"]["highlight"]
        assert highlight["fields"]["content"]["highlight_query"] == {"match": {"content": wikipedia_track.ABSENT_QUERY_TERM}}

    def test_options_merge_into_highlight_body(self):
        params = dsl_params(**{"source": False, "highlight": True, "fields": "content", "highlight-options": {"order": "score"}})
        assert params["body"]["highlight"]["order"] == "score"


class TestCrossEngineParity:
    @pytest.mark.parametrize("query_type", ["qstr-bool", "wildcard", "fuzzy"])
    def test_synthesized_queries_are_identical_across_engines(self, query_type):
        esql = esql_params(**{"query-type": query_type, "fields": "content", "source": False, "highlight": True, "size": 10})
        dsl = dsl_params(**{"query-type": query_type, "fields": "content", "source": False, "highlight": True})
        dsl_query_text = dsl["body"]["query"]["query_string"]["query"]
        assert f'QSTR("{dsl_query_text}")' in esql["query"]


def render_json_fragment(directory, template_name, **params):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(directory)))
    rendered = env.get_template(template_name).render(**params)
    return json.loads(f"[{rendered}]")


class TestRenderedTemplates:
    def test_highlight_operations_render(self):
        ops = render_json_fragment(TRACK_DIR / "operations", "highlight.json")
        by_name = {op["name"]: op for op in ops}
        assert len(by_name) == len(ops), "duplicate operation names"

        highlight_op = by_name["esql-match-highlight"]
        assert highlight_op["operation-type"] == "esql"
        assert highlight_op["highlight"] is True
        assert highlight_op["source"] is False
        assert highlight_op["seed"] == by_name["dsl-match-highlight"]["seed"]

        presort = by_name["esql-match-highlight-presort"]
        assert presort["highlight-placement"] == "pre-sort"

        miss = by_name["dsl-highlight-miss-nomatchsize"]
        assert miss["highlight-miss"] is True
        assert miss["highlight-options"] == {"no_match_size": 100}

        profile = by_name["esql-profile-match-highlight"]
        assert profile["operation-type"] == "esql-profile"

        for engine in ("esql", "dsl"):
            assert by_name[f"{engine}-highlight-wholefield"]["highlight-options"] == {"number_of_fragments": 0}
            assert by_name[f"{engine}-highlight-htmlencoder"]["highlight-options"] == {"encoder": "html"}

    def test_highlight_schedule_references_defined_operations(self):
        defined = set()
        for ops_file in ("default.json", "highlight.json"):
            for op in render_json_fragment(TRACK_DIR / "operations", ops_file):
                defined.add(op["name"])
        schedule = render_json_fragment(TRACK_DIR / "challenges" / "common", "highlighting-schedule.json")
        task_names = [task["name"] for task in schedule]
        assert len(set(task_names)) == len(task_names), "duplicate task names"
        for task in schedule:
            assert task["operation"] in defined, f"task {task['name']} references undefined operation {task['operation']}"

    def test_highlight_schedule_passes(self):
        schedule = render_json_fragment(TRACK_DIR / "challenges" / "common", "highlighting-schedule.json")
        latency = [t for t in schedule if "highlight-latency" in t.get("tags", [])]
        throughput = [t for t in schedule if "highlight-throughput" in t.get("tags", [])]
        profile = [t for t in schedule if "esql-profile" in t.get("tags", [])]
        assert all(t["clients"] == 1 and "target-throughput" in t for t in latency)
        assert all(t["clients"] == 20 for t in throughput)
        assert all(t["clients"] == 1 for t in profile)
        # options variations and forced-miss operations run in the latency pass only
        throughput_ops = {t["operation"] for t in throughput}
        assert not any("miss" in op or "1frag" in op or "orderscore" in op for op in throughput_ops)
        # profile twins run last
        assert [t["operation"] for t in schedule[-3:]] == [
            "esql-profile-match-highlight",
            "esql-profile-match-highlight-content",
            "esql-profile-match-highlight-presort",
        ]

    def test_offsets_mapping(self):
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TRACK_DIR)))
        mapping = json.loads(env.get_template("wikipedia-offsets-mapping.json").render())
        for field in ("title", "content"):
            assert mapping["mappings"]["properties"][field]["index_options"] == "offsets"
