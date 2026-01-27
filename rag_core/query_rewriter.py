"""Query rewriting strategies for improved retrieval."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from rag_core.llm import OpenAIChatLLM

logger = logging.getLogger(__name__)


class RewriteStrategy(str, Enum):
    """Available query rewriting strategies."""

    CONTEXTUAL = "contextual"
    EXPANSION = "expansion"
    HYDE = "hyde"
    DECOMPOSITION = "decomposition"
    STEP_BACK = "step_back"
    NONE = "none"


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""

    query: str
    response: str


@dataclass
class RewriteResult:
    """Result of query rewriting."""

    original_query: str
    rewritten_queries: list[str]
    strategy: RewriteStrategy
    metadata: dict = field(default_factory=dict)

    @property
    def primary_query(self) -> str:
        """Return the primary rewritten query."""
        return self.rewritten_queries[0] if self.rewritten_queries else self.original_query


class BaseQueryRewriter(ABC):
    """Abstract base class for query rewriters."""

    strategy: RewriteStrategy = RewriteStrategy.NONE

    @abstractmethod
    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        **kwargs,
    ) -> RewriteResult:
        """Rewrite the query."""
        pass


class ContextualRewriter(BaseQueryRewriter):
    """
    Rewrites queries by incorporating conversation context.

    Handles cases like:
    - "它是什么" -> "Milvus 向量数据库是什么"
    - "怎么安装" -> "如何安装 Milvus 向量数据库"
    - "还有其他方法吗" -> "除了 HNSW 索引，Milvus 还支持哪些其他索引类型"
    """

    strategy = RewriteStrategy.CONTEXTUAL

    SYSTEM_PROMPT = """You are a query rewriter. Your task is to rewrite the user's query to be self-contained and clear, incorporating relevant context from the conversation history.

Rules:
1. Replace pronouns (it, this, that, they, 它, 这个, 那个) with their actual referents
2. Include relevant context that makes the query unambiguous
3. Keep the rewritten query concise and natural
4. Preserve the user's original intent and language
5. If the query is already self-contained, return it unchanged
6. Output ONLY the rewritten query, nothing else

Examples:
History: Q: What is Milvus? A: Milvus is an open-source vector database...
Query: How do I install it?
Rewritten: How do I install Milvus?

History: Q: 介绍一下HNSW索引 A: HNSW是一种基于图的近似最近邻索引...
Query: 性能怎么样?
Rewritten: HNSW索引的性能怎么样?

History: (empty)
Query: What is RAG?
Rewritten: What is RAG?"""

    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        **kwargs,
    ) -> RewriteResult:
        if not conversation_history:
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy=self.strategy,
                metadata={"skipped": "no_history"},
            )

        history_text = self._format_history(conversation_history)

        user_prompt = f"""Conversation history:
{history_text}

Current query: {query}

Rewritten query:"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        rewritten = self.llm.complete(messages, temperature=0.0).strip()
        rewritten = self._clean_output(rewritten, query)

        logger.info(f"Contextual rewrite: '{query}' -> '{rewritten}'")

        return RewriteResult(
            original_query=query,
            rewritten_queries=[rewritten],
            strategy=self.strategy,
            metadata={"history_turns": len(conversation_history)},
        )

    def _format_history(
        self,
        history: Sequence[ConversationTurn],
        max_turns: int = 3,
    ) -> str:
        recent = list(history[-max_turns:])
        lines = []
        for turn in recent:
            lines.append(f"Q: {turn.query}")
            response = turn.response[:500] + "..." if len(turn.response) > 500 else turn.response
            lines.append(f"A: {response}")
        return "\n".join(lines)

    def _clean_output(self, output: str, original: str) -> str:
        """Clean LLM output to extract just the query."""
        prefixes = ["Rewritten:", "Rewritten query:", "改写后:", "改写："]
        for prefix in prefixes:
            if output.lower().startswith(prefix.lower()):
                output = output[len(prefix) :].strip()

        output = output.strip('"\'')

        if not output or len(output) < 2:
            return original

        return output


class QueryExpansionRewriter(BaseQueryRewriter):
    """
    Expands the query with synonyms and related terms.

    Generates multiple query variants to improve recall:
    - Original: "ML model deployment"
    - Expanded: ["ML model deployment", "machine learning model serving",
                 "deploy neural network production", "model inference deployment"]
    """

    strategy = RewriteStrategy.EXPANSION

    SYSTEM_PROMPT = """You are a search query expansion expert. Given a query, generate alternative versions that capture the same intent using different words, synonyms, and related terms.

Rules:
1. Generate 3-4 alternative queries
2. Each query should use different terminology but seek the same information
3. Include technical synonyms and common variations
4. Maintain the original language (Chinese -> Chinese, English -> English)
5. Output one query per line, no numbering or bullets

Example:
Query: How to deploy ML models?
Alternatives:
machine learning model deployment guide
serving neural networks in production
deploying trained models to production
ML model inference setup

Example:
Query: 向量数据库怎么选择
Alternatives:
向量数据库选型指南
如何选择向量检索引擎
向量数据库对比分析
选择合适的向量存储方案"""

    def __init__(self, llm: OpenAIChatLLM, num_expansions: int = 3):
        self.llm = llm
        self.num_expansions = num_expansions

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        **kwargs,
    ) -> RewriteResult:
        user_prompt = f"Query: {query}\nAlternatives:"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.complete(messages, temperature=0.3)
        expanded = self._parse_expansions(response, query)

        all_queries = [query] + [q for q in expanded if q != query]
        all_queries = all_queries[: self.num_expansions + 1]

        logger.info(f"Query expansion: '{query}' -> {all_queries}")

        return RewriteResult(
            original_query=query,
            rewritten_queries=all_queries,
            strategy=self.strategy,
            metadata={"num_expansions": len(all_queries) - 1},
        )

    def _parse_expansions(self, response: str, original: str) -> list[str]:
        """Parse expansion queries from LLM response."""
        lines = response.strip().split("\n")
        expansions = []

        for line in lines:
            cleaned = line.strip()
            cleaned = re.sub(r"^[\d\.\-\*\•]+\s*", "", cleaned)
            cleaned = cleaned.strip()

            if cleaned and len(cleaned) > 2 and cleaned != original:
                expansions.append(cleaned)

        return expansions[: self.num_expansions]


class HyDERewriter(BaseQueryRewriter):
    """
    Generates a hypothetical answer document for the query.

    Instead of searching with the question embedding, we:
    1. Generate what the answer MIGHT look like
    2. Use that hypothetical answer for retrieval

    This often works better because documents contain answers, not questions.

    Example:
    Query: "What is the time complexity of HNSW search?"
    HyDE: "The time complexity of HNSW (Hierarchical Navigable Small World)
           search is O(log N) on average, where N is the number of vectors..."
    """

    strategy = RewriteStrategy.HYDE

    SYSTEM_PROMPT = """You are a technical documentation expert. Given a question, write a short paragraph that would be the ideal answer found in technical documentation.

Rules:
1. Write as if you are the authoritative documentation
2. Be specific and technical
3. Keep it to 2-4 sentences
4. Use the same language as the question
5. Include relevant technical terms that would appear in real documentation
6. Do NOT say "I don't know" - generate a plausible answer

Example:
Question: How does HNSW indexing work?
Answer: HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor search algorithm. It constructs a multi-layer graph where each layer contains a subset of the data points. During search, it starts from the top layer and greedily navigates to find the nearest neighbors, progressively moving to denser layers for refinement.

Example:
Question: Milvus 支持哪些索引类型?
Answer: Milvus 支持多种索引类型以满足不同场景需求。主要包括：FLAT（暴力搜索，精确但慢）、IVF_FLAT（倒排索引，平衡精度和速度）、IVF_SQ8（量化压缩，节省内存）、HNSW（图索引，高性能）等。用户可根据数据规模和精度要求选择合适的索引。"""

    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        **kwargs,
    ) -> RewriteResult:
        user_prompt = f"Question: {query}\nAnswer:"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        hypothetical_doc = self.llm.complete(messages, temperature=0.3).strip()

        logger.info(f"HyDE generated: '{query}' -> '{hypothetical_doc[:100]}...'")

        return RewriteResult(
            original_query=query,
            rewritten_queries=[hypothetical_doc],
            strategy=self.strategy,
            metadata={"hyde_length": len(hypothetical_doc)},
        )


class DecompositionRewriter(BaseQueryRewriter):
    """
    Breaks down complex queries into simpler sub-queries.

    Example:
    Query: "Compare the performance and memory usage of HNSW vs IVF_FLAT indexes"
    Sub-queries:
    - "What is the search performance of HNSW index?"
    - "What is the search performance of IVF_FLAT index?"
    - "What is the memory usage of HNSW index?"
    - "What is the memory usage of IVF_FLAT index?"
    """

    strategy = RewriteStrategy.DECOMPOSITION

    SYSTEM_PROMPT = """You are a query decomposition expert. Break down complex questions into simpler, atomic sub-questions.

Rules:
1. Each sub-question should be answerable independently
2. Generate 2-4 sub-questions
3. Cover all aspects of the original question
4. Keep sub-questions simple and focused
5. Maintain the original language
6. Output one sub-question per line

Example:
Query: Compare Redis and Milvus for vector search in terms of performance and scalability
Sub-questions:
What is the vector search performance of Redis?
What is the vector search performance of Milvus?
How does Redis scale for vector search workloads?
How does Milvus scale for vector search workloads?

Example:
Query: 如何在Python中使用Milvus进行相似度搜索并处理返回结果
Sub-questions:
Python中如何连接Milvus数据库?
如何使用Python在Milvus中执行向量相似度搜索?
如何处理Milvus搜索返回的结果?"""

    def __init__(self, llm: OpenAIChatLLM, max_sub_queries: int = 4):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        **kwargs,
    ) -> RewriteResult:
        if not self._needs_decomposition(query):
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy=self.strategy,
                metadata={"skipped": "simple_query"},
            )

        user_prompt = f"Query: {query}\nSub-questions:"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.llm.complete(messages, temperature=0.2)
        sub_queries = self._parse_sub_queries(response)

        if not sub_queries:
            sub_queries = [query]

        logger.info(f"Query decomposition: '{query}' -> {sub_queries}")

        return RewriteResult(
            original_query=query,
            rewritten_queries=sub_queries,
            strategy=self.strategy,
            metadata={"num_sub_queries": len(sub_queries)},
        )

    def _needs_decomposition(self, query: str) -> bool:
        """Heuristic check if query needs decomposition."""
        comparison_words = [
            "compare",
            "vs",
            "versus",
            "difference",
            "比较",
            "对比",
            "区别",
            r"和.*哪个",
        ]
        for word in comparison_words:
            if re.search(word, query.lower()):
                return True

        multi_aspect_words = ["and", "以及", "并且", "同时", "还有"]
        for word in multi_aspect_words:
            if word in query.lower():
                return True

        if len(query) > 100:
            return True

        return False

    def _parse_sub_queries(self, response: str) -> list[str]:
        """Parse sub-queries from LLM response."""
        lines = response.strip().split("\n")
        sub_queries = []

        for line in lines:
            cleaned = line.strip()
            cleaned = re.sub(r"^[\d\.\-\*\•]+\s*", "", cleaned)
            cleaned = cleaned.strip()

            if cleaned and len(cleaned) > 5:
                sub_queries.append(cleaned)

        return sub_queries[: self.max_sub_queries]


class StepBackRewriter(BaseQueryRewriter):
    """
    Generates a more abstract/general version of the query.

    This helps when the specific query doesn't have direct matches,
    but related general concepts do.

    Example:
    Query: "Why does my HNSW index have low recall at ef=64?"
    Step-back: "What factors affect HNSW index recall and how to tune them?"
    """

    strategy = RewriteStrategy.STEP_BACK

    SYSTEM_PROMPT = """You are a search optimization expert. Given a specific question, generate a more general "step-back" question that would retrieve broader background information.

Rules:
1. The step-back question should be more general/abstract
2. It should help understand the underlying concepts
3. Combine with the original for better coverage
4. Keep the same language as the original
5. Output only the step-back question

Example:
Query: Why is my HNSW search slow with 10 million vectors?
Step-back: What are the factors that affect HNSW search performance and how to optimize them?

Example:
Query: Milvus IVF_FLAT的nprobe参数设置为多少合适?
Step-back: IVF索引的参数调优原则和最佳实践是什么?

Example:
Query: How to fix "collection not found" error in pymilvus?
Step-back: What are common pymilvus errors and their solutions?"""

    def __init__(self, llm: OpenAIChatLLM):
        self.llm = llm

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        **kwargs,
    ) -> RewriteResult:
        user_prompt = f"Query: {query}\nStep-back question:"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        step_back_query = self.llm.complete(messages, temperature=0.2).strip()

        queries = [query, step_back_query] if step_back_query else [query]

        logger.info(f"Step-back: '{query}' -> '{step_back_query}'")

        return RewriteResult(
            original_query=query,
            rewritten_queries=queries,
            strategy=self.strategy,
            metadata={"step_back_query": step_back_query},
        )


class QueryRewriterPipeline:
    """
    Orchestrates multiple rewriting strategies.

    Supports:
    - Single strategy execution
    - Chained strategies (e.g., contextual -> expansion)
    - Parallel strategies with result merging
    """

    def __init__(
        self,
        llm: OpenAIChatLLM,
        strategies: list[RewriteStrategy] | None = None,
        enable_contextual: bool = True,
        enable_expansion: bool = False,
        enable_hyde: bool = False,
        enable_decomposition: bool = False,
        enable_step_back: bool = False,
    ):
        self.llm = llm

        self.rewriters: dict[RewriteStrategy, BaseQueryRewriter] = {
            RewriteStrategy.CONTEXTUAL: ContextualRewriter(llm),
            RewriteStrategy.EXPANSION: QueryExpansionRewriter(llm),
            RewriteStrategy.HYDE: HyDERewriter(llm),
            RewriteStrategy.DECOMPOSITION: DecompositionRewriter(llm),
            RewriteStrategy.STEP_BACK: StepBackRewriter(llm),
        }

        if strategies:
            self.enabled_strategies = strategies
        else:
            self.enabled_strategies = []
            if enable_contextual:
                self.enabled_strategies.append(RewriteStrategy.CONTEXTUAL)
            if enable_expansion:
                self.enabled_strategies.append(RewriteStrategy.EXPANSION)
            if enable_hyde:
                self.enabled_strategies.append(RewriteStrategy.HYDE)
            if enable_decomposition:
                self.enabled_strategies.append(RewriteStrategy.DECOMPOSITION)
            if enable_step_back:
                self.enabled_strategies.append(RewriteStrategy.STEP_BACK)

    def rewrite(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        strategy: RewriteStrategy | None = None,
    ) -> RewriteResult:
        """
        Execute query rewriting.

        If strategy is specified, use only that strategy.
        Otherwise, run all enabled strategies and merge results.
        """
        if strategy and strategy != RewriteStrategy.NONE:
            rewriter = self.rewriters.get(strategy)
            if rewriter:
                return rewriter.rewrite(query, conversation_history)
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy=RewriteStrategy.NONE,
            )

        if not self.enabled_strategies:
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy=RewriteStrategy.NONE,
            )

        current_query = query
        all_queries = [query]
        all_metadata = {}

        for strat in self.enabled_strategies:
            rewriter = self.rewriters.get(strat)
            if not rewriter:
                continue

            result = rewriter.rewrite(current_query, conversation_history)

            for q in result.rewritten_queries:
                if q not in all_queries:
                    all_queries.append(q)

            current_query = result.primary_query

            all_metadata[strat.value] = result.metadata

        return RewriteResult(
            original_query=query,
            rewritten_queries=all_queries,
            strategy=(
                self.enabled_strategies[0]
                if len(self.enabled_strategies) == 1
                else RewriteStrategy.NONE
            ),
            metadata=all_metadata,
        )

    def rewrite_for_retrieval(
        self,
        query: str,
        conversation_history: Sequence[ConversationTurn] | None = None,
        max_queries: int = 5,
    ) -> list[str]:
        """
        Convenience method that returns just the query strings for retrieval.
        """
        result = self.rewrite(query, conversation_history)
        return result.rewritten_queries[:max_queries]
