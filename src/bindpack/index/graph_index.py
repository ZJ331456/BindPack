import json
import os
import logging
from dataclasses import field
from copy import deepcopy
from typing import List, Dict, Tuple, Iterable
from tqdm import tqdm
import igraph as ig
import re
import tempfile
import hashlib

from .openie.openie_quality import validate_triples

from .openie.source_verified_openie import SourceVerifiedOpenIE as OpenIE
from .openie.structural_openie import StructuralOpenIE, STRUCTURAL_VERSION
from ..utils.misc_utils import *
from ..utils.misc_utils import NerRawOutput, TripleRawOutput
from ..utils.embed_utils import retrieve_knn

logger = logging.getLogger(__name__)

class GraphIndexMixin:
    """Graph construction, OpenIE checkpoints and index publication."""

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph


    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')


    def index(self, docs: List[str], *, rebuild_graph: bool = False,
              retry_failed_openie: bool = True):
        """
        Indexes the given documents based on the HippoRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
            rebuild_graph : bool
                Explicitly rebuild edges and prune obsolete fact/entity vectors.
                Use a separate working directory when repairing shared indexes.
            retry_failed_openie : bool
                Re-extract cached rows with an explicit failed/partial status.
                Empty results marked valid are retained.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        rebuild_graph = rebuild_graph or bool(
            getattr(self.global_config, 'force_index_from_scratch', False)
        )

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(
            chunk_to_rows.keys(), retry_failed=retry_failed_openie,
        )
        progress = None
        if isinstance(self.openie, (OpenIE, StructuralOpenIE)):
            from .openie_checkpoint import OpenIECheckpoint
            from .openie.openie_source_evidence import VERIFIER_VERSION
            from .openie_build_queue import openie_row_is_verified_complete
            from .shared_index_builder import quality_profile
            validation_mode = getattr(self.global_config, 'openie_validation_mode',
                                      'structural' if isinstance(self.openie, StructuralOpenIE) else 'source_verified')
            corpus_digest = hashlib.sha256('\n'.join(sorted(chunk_to_rows)).encode()).hexdigest()
            progress = OpenIECheckpoint(os.path.join(
                self.working_dir, f'openie_progress_{corpus_digest}.sqlite'), {
                'llm_name': self.global_config.llm_name,
                'endpoint': self.global_config.llm_base_url,
                'verifier': STRUCTURAL_VERSION if validation_mode == 'structural' else VERIFIER_VERSION,
                'temperature': self.global_config.temperature,
                'seed': self.global_config.seed,
                'ner_max_tokens': 512,
                'triple_max_tokens': 2048,
                'quality_profile': quality_profile(
                    strict=getattr(self.global_config, 'openie_strict', True),
                    prompt_version=getattr(self.global_config, 'openie_prompt_version', 'optimized'),
                    validation_mode=validation_mode),
                'corpus_ids_sha256': corpus_digest,
            })
            all_openie_info[:] = progress.overlay(all_openie_info, chunk_to_rows)
            by_id = {row['idx']: row for row in all_openie_info}
            chunk_keys_to_process = [key for key in chunk_to_rows if key not in by_id
                or not openie_row_is_verified_complete(self.openie, by_id[key])]
            self.openie.initial_rows = by_id
            self.openie.checkpoint = progress.save
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        # Replacing facts for an existing passage requires replacing its graph
        # edges as well.  The incremental path deliberately skips those edges;
        # refuse a stale graph rather than silently updating only the JSON.
        if self.graph.vcount() and not rebuild_graph:
            graph_names = set(self.graph.vs['name'])
            replacing = graph_names.intersection(chunk_keys_to_process)
            invalid_cached = [row['idx'] for row in all_openie_info
                              if self._openie_row_has_invalid_graph_triples(row)]
            if replacing or invalid_cached:
                if progress is not None:
                    progress.close()
                    self.openie.checkpoint = None
                raise RuntimeError(
                    'OpenIE repair changes an existing graph. Use index(..., '
                    'rebuild_graph=True) in a separate index directory; '
                    f'{len(replacing)} passages need extraction and '
                    f'{len(invalid_cached)} cached passages contain invalid graph triples.'
                )

        try:
            if len(chunk_keys_to_process) > 0:
                new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
                self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
        finally:
            if progress is not None:
                progress.close()
                self.openie.checkpoint = None

        incomplete_rows = [row['idx'] for row in all_openie_info if openie_row_needs_retry(row)]
        if isinstance(self.openie, (OpenIE, StructuralOpenIE)):
            for row in all_openie_info:
                verified = openie_row_is_verified_complete(self.openie, row)
                if not verified and row['idx'] not in incomplete_rows:
                    incomplete_rows.append(row['idx'])
        from .publication_policy import apply_publication_policy
        strict = getattr(self.global_config, 'openie_strict', True)
        report = apply_publication_policy(
            all_openie_info, incomplete_rows, strict=strict,
            report_path=os.path.join(self.working_dir, 'openie_publication_report.json'))
        self._openie_file_metadata = getattr(self, '_openie_file_metadata', {})
        self._openie_file_metadata['publication_report'] = report
        if incomplete_rows and strict:
            # Diagnostic checkpoints are saved even when save_openie=False.
            # A partial graph must never be published as a completed index.
            self.save_openie_results(all_openie_info)
            raise RuntimeError(
                f'OpenIE incomplete for {len(incomplete_rows)} chunks; diagnostics saved to '
                f'{self.openie_results_path}. No graph or entity/fact vectors were published. '
                f'First chunks: {incomplete_rows[:5]}'
            )
        if incomplete_rows:
            logger.warning('openie_strict=False: continuing with %d failed chunks; '
                           'their relations are excluded from the graph. First chunks: %s',
                           len(incomplete_rows), incomplete_rows[:5])
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [self._normalize_graph_triples(triple_results_dict[chunk_id].triples)
                         for chunk_id in chunk_ids]
        chunk_digests = self._openie_chunk_digests(chunk_ids, chunk_triples)
        if self.graph.vcount() and not rebuild_graph:
            previous = (self.graph['openie_chunk_digests']
                        if 'openie_chunk_digests' in self.graph.attributes() else {})
            changed = [key for key, digest in chunk_digests.items()
                       if key in previous and previous[key] != digest]
            if changed:
                raise RuntimeError(
                    f'Facts changed for {len(changed)} existing passages. '
                    'Use index(..., rebuild_graph=True) to keep graph and vectors consistent.'
                )

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        if rebuild_graph or self.graph.vcount() == 0:
            logger.info('Building graph from current validated OpenIE results.')
            self.graph = ig.Graph(directed=self.global_config.is_directed_graph)
            self.ready_to_retrieve = False
            self._prune_embedding_store(self.entity_embedding_store, entity_nodes)
            self._prune_embedding_store(self.fact_embedding_store, [str(fact) for fact in facts])

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            self.ready_to_retrieve = False
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.graph['openie_chunk_digests'] = chunk_digests
            self.save_igraph()


    def _resolve_index_text_normalizer(self):
        manifest_path = os.path.join(self.working_dir, 'index_manifest.json')
        schema = None
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding='utf-8') as f:
                schema = json.load(f).get('text_normalization')
        if schema is None or schema == 'ascii_alnum_lower_v1':
            self._text_normalization_schema = 'ascii_alnum_lower_v1'
            return text_processing
        if schema == 'unicode_alnum_casefold_v1':
            self._text_normalization_schema = schema
            return unicode_text_processing
        raise ValueError(f'Unsupported index text_normalization={schema!r}: {manifest_path}')


    def _normalize_index_text(self, text):
        return getattr(self, '_text_normalizer', text_processing)(text)


    def _normalize_graph_triples(self, triples):
        return normalize_graph_triples(triples, normalizer=self._normalize_index_text)


    def _openie_row_has_invalid_graph_triples(self, row: dict) -> bool:
        validation = validate_triples(row.get('extracted_triples', []))
        return bool(validation.invalid_triples or any(
            not all(field.strip() for field in self._normalize_index_text(triple))
            for triple in validation.valid_triples
        ))


    @staticmethod
    def _openie_chunk_digests(chunk_ids, chunk_triples) -> Dict[str, str]:
        return {
            key: compute_mdhash_id(json.dumps(sorted(set(map(tuple, triples))), ensure_ascii=False))
            for key, triples in zip(chunk_ids, chunk_triples)
        }


    @staticmethod
    def _prune_embedding_store(store, current_texts: List[str]):
        """An explicit rebuild retains embeddings only for current facts/entities."""
        desired = {compute_mdhash_id(text, prefix=store.namespace + '-') for text in current_texts}
        obsolete = [key for key in store.get_all_ids() if key not in desired]
        if obsolete:
            logger.info('Removing %s obsolete %s vectors before rebuilding graph.',
                        len(obsolete), store.namespace)
            store.delete(obsolete)


    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the HippoRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(self._normalize_index_text(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[self._normalize_index_text(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False


    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            for triple in triples:
                triple = tuple(triple)

                node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                entities_in_chunk.add(node_key)
                entities_in_chunk.add(node_2_key)

                # Only insert new fact edges when the passage node is absent (reuse-safe).
                if chunk_key not in current_graph_nodes:
                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

            # Always rebuild entity→chunk map (needed when loading an existing graph).
            for node in entities_in_chunk:
                self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))


    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks


    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)
        del entity_embs
        import gc
        gc.collect()

        num_synonym_triple = 0
        node_keys = list(query_node_key2knn_node_keys.keys())
        for node_key in tqdm(node_keys, total=len(node_keys)):
            nn_indices, nn_scores = query_node_key2knn_node_keys.pop(node_key)

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                num_nns = 0
                for nn_idx, score in zip(nn_indices.tolist(), nn_scores.tolist()):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn = entity_node_keys[int(nn_idx)]
                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

        del query_node_key2knn_node_keys, node_keys
        gc.collect()
        logger.info(f"Added {num_synonym_triple} synonymy edges.")


    def load_existing_openie(self, chunk_keys: Iterable[str], *,
                             retry_failed: bool = False) -> Tuple[List[dict], List[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.

        retry_failed is opt-in so retrieval preparation never starts extraction.
        """

        # combine openie_results with contents already in file, if file exists
        # Preserve corpus order.  A set here changes OpenIE request, file, and
        # potentially graph tie-break order across Python hash seeds.
        chunk_keys_to_save = []
        self._openie_file_metadata = {}

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            with open(self.openie_results_path, encoding="utf-8") as openie_file:
                openie_results = json.load(openie_file)
            self._openie_file_metadata = {
                key: deepcopy(value) for key, value in openie_results.items()
                if key not in ('docs', 'avg_ent_chars', 'avg_ent_words')
            }
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = {}
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                # Historical duplicate rows can arise when a failed chunk is
                # reprocessed.  The latest extraction replaces that chunk.
                renamed_openie_info[openie_info['idx']] = openie_info

            all_openie_info = list(renamed_openie_info.values())

            existing_openie_rows = {info['idx']: info for info in all_openie_info}

            for chunk_key in chunk_keys:
                row = existing_openie_rows.get(chunk_key)
                if row is None or (retry_failed and (
                        openie_row_needs_retry(row) or self._openie_row_has_invalid_graph_triples(row))):
                    chunk_keys_to_save.append(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = list(chunk_keys)

        return all_openie_info, chunk_keys_to_save


    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        positions = {row['idx']: position for position, row in enumerate(all_openie_info)}
        replacements = []
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            if chunk_key not in ner_results_dict or chunk_key not in triple_results_dict:
                raise RuntimeError(f'Missing OpenIE stage result for chunk {chunk_key}')
            ner_result = ner_results_dict[chunk_key]
            triple_result = triple_results_dict[chunk_key]
            if ner_result.chunk_id != chunk_key or triple_result.chunk_id != chunk_key:
                raise RuntimeError(f'OpenIE chunk ID mismatch for {chunk_key}')
            validation = validate_triples(triple_result.triples)
            triple_metadata = deepcopy(triple_result.metadata or {})
            if validation.invalid_triples:
                triple_metadata.update({
                    'quality_status': 'partial' if validation.valid_triples else 'failed',
                    'invalid_triple_count': len(validation.invalid_triples),
                    'validation_issues': validation.issues,
                })
            chunk_openie_info = {
                'idx': chunk_key, 'passage': passage,
                'extracted_entities': ner_result.unique_entities,
                'extracted_triples': validation.valid_triples,
                'openie_metadata': {
                    'ner': deepcopy(ner_result.metadata or {}), 'triples': triple_metadata,
                },
                'openie_responses': {'ner': ner_result.response, 'triples': triple_result.response},
            }
            replacements.append((chunk_key, chunk_openie_info))

        # Validate all results before mutating the caller's rows.  A missing
        # stage must not leave a half-merged corpus that looks successful.
        for chunk_key, chunk_openie_info in replacements:
            if chunk_key in positions:
                all_openie_info[positions[chunk_key]] = chunk_openie_info
            else:
                positions[chunk_key] = len(all_openie_info)
                all_openie_info.append(chunk_openie_info)

        return all_openie_info


    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                **deepcopy(getattr(self, '_openie_file_metadata', {})),
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            # os.replace also avoids overwriting other experiment files that
            # happen to share the source OpenIE file through hard links.
            directory = os.path.dirname(os.path.abspath(self.openie_results_path))
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                                 dir=directory, delete=False) as f:
                    temporary_path = f.name
                    json.dump(openie_dict, f, ensure_ascii=False)
                os.replace(temporary_path, self.openie_results_path)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")


    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())


    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)


    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        # Avoid unused adjacency mirrors of node_to_node_stats (RAM peak on large graphs).
        current_node_ids = set(self.graph.vs["name"])
        valid_edges = []
        valid_weights = {"weight": []}
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            source_node_id, target_node_id = edge[0], edge[1]
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )
        del valid_edges, valid_weights
        import gc
        gc.collect()


    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        directory = os.path.dirname(os.path.abspath(self._graph_pickle_filename))
        fd, temporary_path = tempfile.mkstemp(dir=directory, suffix='.pickle')
        os.close(fd)
        try:
            self.graph.write_pickle(temporary_path)
            os.replace(temporary_path, self._graph_pickle_filename)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        logger.info(f"Saving graph completed!")


    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

